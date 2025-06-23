//! The module that implements the `wasmtime run` command.

#![cfg_attr(
    not(feature = "component-model"),
    allow(irrefutable_let_patterns, unreachable_patterns)
)]

use crate::common::{Profile, RunCommon, RunTarget};
use async_trait::async_trait;

use anyhow::{Context as _, Error, Result, anyhow, bail};
use clap::Parser;
use golem_rib_repl::{
    ComponentSource, ReplComponentDependencies, RibDependencyManager, RibRepl, RibReplConfig,
    WorkerFunctionInvoke,
};
use golem_wasm_ast::analysis::AnalysedType;
use golem_wasm_ast::analysis::analysed_type::str;
use golem_wasm_ast::analysis::wit_parser::WitAnalysisContext;
use golem_wasm_rpc::protobuf::typed_result::ResultValue;
use golem_wasm_rpc::protobuf::{TypeAnnotatedValue, type_annotated_value};
use golem_wasm_rpc::{Value, ValueAndType, parse_value_and_type};
use rib::{
    ComponentDependency, ComponentDependencyKey, ParsedFunctionName, ParsedFunctionReference,
};
use std::cell::RefCell;
use std::ffi::OsString;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::vec;
use uuid::Uuid;
use wasi_common::sync::{Dir, TcpListener, WasiCtxBuilder, ambient_authority};
use wasmtime::component::{Component, Instance, ResourceAny};
use wasmtime::{AsContextMut, Engine, Func, Module, Store, StoreLimits, Val, ValType};
use wasmtime_cli_flags::CommonOptions;
use wasmtime_wasi::p2::{IoView, WasiView};

#[cfg(feature = "wasi-nn")]
use wasmtime_wasi_nn::wit::WasiNnView;

#[cfg(feature = "wasi-threads")]
use wasmtime_wasi_threads::WasiThreadsCtx;

#[cfg(feature = "wasi-config")]
use wasmtime_wasi_config::{WasiConfig, WasiConfigVariables};
#[cfg(feature = "wasi-http")]
use wasmtime_wasi_http::{
    DEFAULT_OUTGOING_BODY_BUFFER_CHUNKS, DEFAULT_OUTGOING_BODY_CHUNK_SIZE, WasiHttpCtx,
};
#[cfg(feature = "wasi-keyvalue")]
use wasmtime_wasi_keyvalue::{WasiKeyValue, WasiKeyValueCtx, WasiKeyValueCtxBuilder};

#[cfg(feature = "wasi-tls")]
use wasmtime_wasi_tls::WasiTlsCtx;

fn parse_preloads(s: &str) -> Result<(String, PathBuf)> {
    let parts: Vec<&str> = s.splitn(2, '=').collect();
    if parts.len() != 2 {
        bail!("must contain exactly one equals character ('=')");
    }
    Ok((parts[0].into(), parts[1].into()))
}

///  Runs a REPL for wasmtime
#[derive(Parser)]
pub struct ReplCommand {
    #[command(flatten)]
    #[expect(missing_docs, reason = "don't want to mess with clap doc-strings")]
    pub run: RunCommon,

    /// The WebAssembly module to run and arguments to pass to it.
    ///
    /// Arguments passed to the wasm module will be configured as WASI CLI
    /// arguments unless the `--invoke` CLI argument is passed in which case
    /// arguments will be interpreted as arguments to the function specified.
    #[arg(value_name = "WASM", trailing_var_arg = true, required = true)]
    pub module_and_args: Vec<OsString>,
}

/// A resource marker used for REPL
pub struct GolemResource;

impl ReplCommand {
    /// Execute Repl
    pub async fn execute_repl(mut self) -> Result<()> {
        let path = PathBuf::from(&self.module_and_args[0]);

        let wasmtime_function_invoke = self.get_instance().await?;

        let repl_config = RibReplConfig {
            history_file: None,
            dependency_manager: Arc::new(WasmtimeComponentDependencyManager {}),
            worker_function_invoke: Arc::new(wasmtime_function_invoke),
            printer: None,
            component_source: Some(ComponentSource {
                source_path: path,
                component_name: "singleton".to_string(),
            }),
            prompt: None,
            command_registry: None,
        };
        let mut repl = RibRepl::bootstrap(repl_config).await?;

        repl.run().await;
        Ok(())
    }

    /// Get reusable instance for REPL
    pub async fn get_instance(&mut self) -> Result<WasmtimeFunctionInvoke> {
        let mut config = self.run.common.config(None)?;
        config.async_support(true);

        let engine = Engine::new(&config)?;

        let main = self
            .run
            .load_module(&engine, self.module_and_args[0].as_ref())?;

        let mut linker = match &main {
            RunTarget::Core(_) => bail!("expecting a wasm component"),
            #[cfg(feature = "component-model")]
            RunTarget::Component(_) => {
                CliLinker::Component(wasmtime::component::Linker::new(&engine))
            }
        };

        let host: Host = Host::default();

        let mut store = Store::new(&engine, host);

        let run_command = RunCommand {
            run: self.run.clone(),
            invoke: None,
            preloads: vec![],
            argv0: None,
            module_and_args: self.module_and_args.clone(),
        };

        run_command.populate_with_wasi(&mut linker, &mut store, &main)?;

        let component = match main {
            RunTarget::Core(_) => bail!("expecting a wasm component"),
            #[cfg(feature = "component-model")]
            RunTarget::Component(component) => component,
        };

        match linker {
            CliLinker::Core(_) => bail!("expected component, found core module".to_string()),

            #[cfg(feature = "component-model")]
            CliLinker::Component(linker) => {
                let instance = linker.instantiate_async(&mut store, &component).await?;

                let session = WasmtimeFunctionInvoke {
                    component,
                    instance: Arc::new(instance),
                    store: Arc::new(tokio::sync::Mutex::new(store)),
                    common_options: self.run.common.clone(),
                };

                Ok(session)
            }
        }
    }

    fn convert_to_type_annotated_value(
        type_annotated_value: TypeAnnotatedValue,
        store: &mut Store<Host>,
    ) -> wasmtime::component::Val {
        match type_annotated_value.type_annotated_value.unwrap() {
            type_annotated_value::TypeAnnotatedValue::Bool(bool) => {
                wasmtime::component::Val::Bool(bool)
            }
            type_annotated_value::TypeAnnotatedValue::U8(u8) => {
                wasmtime::component::Val::U8(u8 as u8)
            }
            type_annotated_value::TypeAnnotatedValue::U16(u16) => {
                wasmtime::component::Val::U16(u16 as u16)
            }
            type_annotated_value::TypeAnnotatedValue::U32(u32) => {
                wasmtime::component::Val::U32(u32)
            }
            type_annotated_value::TypeAnnotatedValue::U64(u64) => {
                wasmtime::component::Val::U64(u64)
            }
            type_annotated_value::TypeAnnotatedValue::S8(s8) => {
                wasmtime::component::Val::S8(s8 as i8)
            }
            type_annotated_value::TypeAnnotatedValue::S16(s16) => {
                wasmtime::component::Val::S16(s16 as i16)
            }
            type_annotated_value::TypeAnnotatedValue::S32(s32) => {
                wasmtime::component::Val::S32(s32)
            }
            type_annotated_value::TypeAnnotatedValue::S64(s64) => {
                wasmtime::component::Val::S64(s64)
            }
            type_annotated_value::TypeAnnotatedValue::F32(f32) => {
                wasmtime::component::Val::Float32(f32)
            }
            type_annotated_value::TypeAnnotatedValue::F64(f64) => {
                wasmtime::component::Val::Float64(f64)
            }
            type_annotated_value::TypeAnnotatedValue::Char(char) => {
                wasmtime::component::Val::Char(std::char::from_u32(char as u32).unwrap())
            }
            type_annotated_value::TypeAnnotatedValue::Str(string) => {
                wasmtime::component::Val::String(string)
            }
            type_annotated_value::TypeAnnotatedValue::List(list) => {
                let values: Vec<wasmtime::component::Val> = list
                    .values
                    .into_iter()
                    .map(|x| Self::convert_to_type_annotated_value(x, store))
                    .collect();
                wasmtime::component::Val::List(values)
            }
            type_annotated_value::TypeAnnotatedValue::Tuple(tuple) => {
                let values: Vec<wasmtime::component::Val> = tuple
                    .value
                    .into_iter()
                    .map(|x| Self::convert_to_type_annotated_value(x, store))
                    .collect();
                wasmtime::component::Val::Tuple(values)
            }
            type_annotated_value::TypeAnnotatedValue::Record(record) => {
                let values = record
                    .value
                    .iter()
                    .map(|x| {
                        (
                            x.name.clone(),
                            Self::convert_to_type_annotated_value(x.value.clone().unwrap(), store),
                        )
                    })
                    .collect::<Vec<_>>();

                wasmtime::component::Val::Record(values)
            }
            type_annotated_value::TypeAnnotatedValue::Variant(typed_variant) => {
                let name = typed_variant.case_name;
                let value = typed_variant.case_value.map(|x| {
                    Box::new(Self::convert_to_type_annotated_value(
                        x.deref().clone(),
                        store,
                    ))
                });
                wasmtime::component::Val::Variant(name, value)
            }
            type_annotated_value::TypeAnnotatedValue::Enum(enum_cases) => {
                wasmtime::component::Val::Enum(enum_cases.value)
            }
            type_annotated_value::TypeAnnotatedValue::Flags(typed_flags) => {
                wasmtime::component::Val::Flags(typed_flags.values)
            }
            type_annotated_value::TypeAnnotatedValue::Option(typed_option) => {
                if let Some(value) = typed_option.value {
                    wasmtime::component::Val::Option(Some(Box::new(
                        Self::convert_to_type_annotated_value(value.deref().clone(), store),
                    )))
                } else {
                    wasmtime::component::Val::Option(None)
                }
            }
            type_annotated_value::TypeAnnotatedValue::Result(typed_result) => {
                let ok = typed_result.result_value;

                match ok {
                    None => wasmtime::component::Val::Result(Ok(None)),
                    Some(value) => match value {
                        ResultValue::OkValue(type_annotated_value) => {
                            let val = Self::convert_to_type_annotated_value(
                                type_annotated_value.deref().clone(),
                                store,
                            );

                            wasmtime::component::Val::Result(Ok(Some(Box::new(val))))
                        }

                        ResultValue::ErrorValue(type_annotated_value) => {
                            let val = Self::convert_to_type_annotated_value(
                                type_annotated_value.deref().clone(),
                                store,
                            );

                            wasmtime::component::Val::Result(Err(Some(Box::new(val))))
                        }
                    },
                }
            }
            type_annotated_value::TypeAnnotatedValue::Handle(typed_handle) => {
                let x = typed_handle.resource_id;

                let typed = wasmtime::component::Resource::<GolemResource>::new_borrow(x as u32);

                let any = ResourceAny::try_from_resource(typed, store)
                    .expect("failed to convert to ResourceAny");

                wasmtime::component::Val::Resource(any)
            }
        }
    }

    fn convert_to_wasm_rpc_value(
        value_and_type: ValueAndType,
        store: &mut Store<Host>,
    ) -> wasmtime::component::Val {
        // Unwrapping it as this is a real bug in the dependent library
        let type_annotated_value = TypeAnnotatedValue::try_from(value_and_type).unwrap();

        Self::convert_to_type_annotated_value(type_annotated_value, store)
    }
}

/// Runs a WebAssembly module
#[derive(Parser, Clone)]
pub struct RunCommand {
    #[command(flatten)]
    #[expect(missing_docs, reason = "don't want to mess with clap doc-strings")]
    pub run: RunCommon,

    /// The name of the function to run
    #[arg(long, value_name = "FUNCTION")]
    pub invoke: Option<String>,

    /// Load the given WebAssembly module before the main module
    #[arg(
        long = "preload",
        number_of_values = 1,
        value_name = "NAME=MODULE_PATH",
        value_parser = parse_preloads,
    )]
    pub preloads: Vec<(String, PathBuf)>,

    /// Override the value of `argv[0]`, typically the name of the executable of
    /// the application being run.
    ///
    /// This can be useful to pass in situations where a CLI tool is being
    /// executed that dispatches its functionality on the value of `argv[0]`
    /// without needing to rename the original wasm binary.
    #[arg(long)]
    pub argv0: Option<String>,

    /// The WebAssembly module to run and arguments to pass to it.
    ///
    /// Arguments passed to the wasm module will be configured as WASI CLI
    /// arguments unless the `--invoke` CLI argument is passed in which case
    /// arguments will be interpreted as arguments to the function specified.
    #[arg(value_name = "WASM", trailing_var_arg = true, required = true)]
    pub module_and_args: Vec<OsString>,
}

enum CliLinker {
    Core(wasmtime::Linker<Host>),
    #[cfg(feature = "component-model")]
    Component(wasmtime::component::Linker<Host>),
}

struct WasmtimeComponentDependencyManager {}

#[async_trait]
impl RibDependencyManager for WasmtimeComponentDependencyManager {
    async fn get_dependencies(&self) -> anyhow::Result<ReplComponentDependencies> {
        Ok(ReplComponentDependencies {
            component_dependencies: vec![],
        })
    }

    async fn add_component(
        &self,
        source_path: &Path,
        component_name: String,
    ) -> anyhow::Result<ComponentDependency> {
        let component_data = std::fs::read(source_path)?;

        let wit_analysis =
            WitAnalysisContext::new(&component_data).map_err(|err| anyhow!(err.reason))?;

        let component_exports = wit_analysis
            .get_top_level_exports()
            .map_err(|err| anyhow!(err.reason))?;

        let root_package_name = wit_analysis.root_package_name();

        let root_package_name_str = root_package_name
            .as_ref()
            .map(|x| format!("{}:{}", x.namespace, x.name));

        let root_package_version = root_package_name.and_then(|p| p.version.map(|v| v.to_string()));

        let component_dependency_key = ComponentDependencyKey {
            component_name: component_name.clone(),
            component_id: Uuid::new_v4(),
            root_package_name: root_package_name_str,
            root_package_version,
        };

        let dependency = ComponentDependency::new(component_dependency_key, component_exports);

        Ok(dependency)
    }
}

struct WasmtimeReplSession {
    common_options: CommonOptions,
    instance: Instance,
    store: Store<Host>,
    component: Component,
}

struct WasmtimeFunctionInvoke {
    common_options: CommonOptions,
    instance: Arc<Instance>,
    store: Arc<tokio::sync::Mutex<Store<Host>>>,
    component: Component,
}

impl WasmtimeFunctionInvoke {
    pub async fn invoke(
        &self,
        function_name: &str,
        args: Vec<ValueAndType>,
        return_type: Option<AnalysedType>,
    ) -> Result<Option<ValueAndType>> {
        let mut store = self.store.lock().await;

        dbg!("acquired lock of store");

        let result = self
            .invoke_function_in_instance(function_name, &mut store, args)
            .await?;

        let result = return_type
            .map(|typ| {
                let result_val = result[0].clone();

                match result_val {
                    wasmtime::component::Val::Resource(resource_any) => {
                        let id = resource_any.try_into_resource::<GolemResource>(&mut *store)?;
                        let resource_id = id.rep();

                        let value = Value::Handle {
                            uri: "/dummy".to_string(),
                            resource_id: resource_id as u64,
                        };

                        Ok(ValueAndType::new(value, typ.clone()))
                    }
                    _ => {
                        let result = result_val.to_wave()?;
                        parse_value_and_type(&typ, &result).map_err(|e| anyhow!(e))
                    }
                }
            })
            .transpose()?;

        Ok(result)
    }

    #[cfg(feature = "component-model")]
    async fn invoke_function_in_instance(
        &self,
        invoke: &str,
        mut store: &mut Store<Host>,
        args: Vec<ValueAndType>,
    ) -> Result<Vec<wasmtime::component::Val>> {
        use wasmtime::component::Val;

        let parsed_function_name = ParsedFunctionName::parse(invoke).unwrap();

        let func = self.find_function(&mut store, &parsed_function_name)?;

        let params = args
            .iter()
            .map(|x| ReplCommand::convert_to_wasm_rpc_value(x.clone(), &mut store))
            .collect::<Vec<_>>();

        let mut results: Vec<Val> = vec![Val::Bool(false); 1];
        func.call_async(&mut *store, &params, &mut results).await?;
        func.post_return_async(&mut *store).await?;

        Ok(results)
    }

    fn find_function(
        &self,
        mut store: &mut Store<Host>,
        parsed_function_name: &ParsedFunctionName,
    ) -> Result<wasmtime::component::Func> {
        match &parsed_function_name.site().interface_name() {
            Some(interface_name) => {
                let (_, exported_instance_idx) = self
                    .instance
                    .get_export(&mut store, None, interface_name)
                    .ok_or(anyhow!(
                        "could not load exports for interface {}",
                        interface_name
                    ))?;

                let func = self
                    .instance
                    .get_export(
                        &mut store,
                        Some(&exported_instance_idx),
                        &parsed_function_name.function().function_name(),
                    )
                    .and_then(|(_, idx)| self.instance.get_func(&mut store, idx));

                match func {
                    Some(func) => Ok(func),
                    None => match parsed_function_name.method_as_static() {
                        None => Err(anyhow!(
                            "could not load function {} for interface {}",
                            &parsed_function_name.function().function_name(),
                            interface_name
                        )),
                        Some(parsed_static) => {
                            let result = self
                                .instance
                                .get_export(
                                    &mut store,
                                    Some(&exported_instance_idx),
                                    &parsed_static.function().function_name(),
                                )
                                .and_then(|(_, idx)| self.instance.get_func(store, idx))
                                .ok_or(anyhow!(
                                    "could not load function {} or {} for interface {}",
                                    &parsed_function_name.function().function_name(),
                                    &parsed_static.function().function_name(),
                                    interface_name
                                ))?;

                            Ok(result)
                        }
                    },
                }
            }
            None => self
                .instance
                .get_func(store, parsed_function_name.function().function_name())
                .ok_or(anyhow!(
                    "could not load function {}",
                    &parsed_function_name.function().function_name()
                )),
        }
    }
}

#[async_trait]
impl WorkerFunctionInvoke for WasmtimeFunctionInvoke {
    async fn invoke(
        &self,
        _component_id: Uuid,
        _component_name: &str,
        _worker_name: Option<String>,
        function_name: &str,
        args: Vec<ValueAndType>,
        return_type: Option<AnalysedType>,
    ) -> anyhow::Result<Option<ValueAndType>> {
        self.invoke(function_name, args, return_type).await
    }
}

impl RunCommand {
    /// Invoke Component Function
    pub async fn invoke_component_function(
        mut self,
        function_name: &str,
        args: Vec<ValueAndType>,
        return_type: Option<AnalysedType>,
    ) -> Result<Option<ValueAndType>> {
        //self.run.common.init_logging()?;
        let mut config = self.run.common.config(None)?;
        config.async_support(true);

        let engine = Engine::new(&config)?;

        let main = self
            .run
            .load_module(&engine, self.module_and_args[0].as_ref())?;

        let mut linker = match &main {
            RunTarget::Core(_) => bail!("expecting a wasm component"),
            #[cfg(feature = "component-model")]
            RunTarget::Component(_) => {
                CliLinker::Component(wasmtime::component::Linker::new(&engine))
            }
        };

        let host: Host = Host::default();

        let mut store = Store::new(&engine, host);
        self.populate_with_wasi(&mut linker, &mut store, &main)?;

        let args = args
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        let function_name_without_golem =
            match ParsedFunctionName::parse(function_name).unwrap().function {
                ParsedFunctionReference::Function { function } => function,
                _ => panic!("currently supporting only function types"),
            };

        let function_name = format!("{}({})", function_name_without_golem, args);

        let result = self
            .load_and_invoke_component(&mut store, &mut linker, &main, &function_name)
            .await
            .with_context(|| {
                format!(
                    "failed to load and invoke component function `{}`",
                    self.module_and_args
                        .iter()
                        .map(|v| v.to_string_lossy())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;

        let result = return_type
            .map(|typ| {
                let result = result[0].to_wave()?;
                parse_value_and_type(&typ, &result).map_err(|e| anyhow!(e))
            })
            .transpose()?;

        Ok(result)
    }

    /// Executes the command.
    pub async fn execute(mut self) -> Result<()> {
        self.run.common.init_logging()?;

        let mut config = self.run.common.config(None)?;
        config.async_support(true);

        if self.run.common.wasm.timeout.is_some() {
            config.epoch_interruption(true);
        }
        match self.run.profile {
            Some(Profile::Native(s)) => {
                config.profiler(s);
            }
            Some(Profile::Guest { .. }) => {
                // Further configured down below as well.
                config.epoch_interruption(true);
            }
            None => {}
        }

        let engine = Engine::new(&config)?;

        // Read the wasm module binary either as `*.wat` or a raw binary.
        let main = self
            .run
            .load_module(&engine, self.module_and_args[0].as_ref())?;

        // Validate coredump-on-trap argument
        if let Some(path) = &self.run.common.debug.coredump {
            if path.contains("%") {
                bail!("the coredump-on-trap path does not support patterns yet.")
            }
        }

        let mut linker = match &main {
            RunTarget::Core(_) => CliLinker::Core(wasmtime::Linker::new(&engine)),
            #[cfg(feature = "component-model")]
            RunTarget::Component(_) => {
                CliLinker::Component(wasmtime::component::Linker::new(&engine))
            }
        };
        if let Some(enable) = self.run.common.wasm.unknown_exports_allow {
            match &mut linker {
                CliLinker::Core(l) => {
                    l.allow_unknown_exports(enable);
                }
                #[cfg(feature = "component-model")]
                CliLinker::Component(_) => {
                    bail!("--allow-unknown-exports not supported with components");
                }
            }
        }

        let host = Host {
            #[cfg(feature = "wasi-http")]
            wasi_http_outgoing_body_buffer_chunks: self
                .run
                .common
                .wasi
                .http_outgoing_body_buffer_chunks,
            #[cfg(feature = "wasi-http")]
            wasi_http_outgoing_body_chunk_size: self.run.common.wasi.http_outgoing_body_chunk_size,
            ..Default::default()
        };

        let mut store = Store::new(&engine, host);
        self.populate_with_wasi(&mut linker, &mut store, &main)?;

        store.data_mut().limits = self.run.store_limits();
        store.limiter(|t| &mut t.limits);

        // If fuel has been configured, we want to add the configured
        // fuel amount to this store.
        if let Some(fuel) = self.run.common.wasm.fuel {
            store.set_fuel(fuel)?;
        }

        let dur = self
            .run
            .common
            .wasm
            .timeout
            .unwrap_or(std::time::Duration::MAX);

        let result = tokio::time::timeout(dur, async {
            let mut profiled_modules: Vec<(String, Module)> = Vec::new();
            if let RunTarget::Core(m) = &main {
                profiled_modules.push(("".to_string(), m.clone()));
            }

            // Load the preload wasm modules.
            for (name, path) in self.preloads.iter() {
                // Read the wasm module binary either as `*.wat` or a raw binary
                let preload_target = self.run.load_module(&engine, path)?;
                let preload_module = match preload_target {
                    RunTarget::Core(m) => m,
                    #[cfg(feature = "component-model")]
                    RunTarget::Component(_) => {
                        bail!("components cannot be loaded with `--preload`")
                    }
                };
                profiled_modules.push((name.to_string(), preload_module.clone()));

                // Add the module's functions to the linker.
                match &mut linker {
                    #[cfg(feature = "cranelift")]
                    CliLinker::Core(linker) => {
                        linker
                            .module_async(&mut store, name, &preload_module)
                            .await
                            .context(format!(
                                "failed to process preload `{}` at `{}`",
                                name,
                                path.display()
                            ))?;
                    }
                    #[cfg(not(feature = "cranelift"))]
                    CliLinker::Core(_) => {
                        bail!("support for --preload disabled at compile time");
                    }
                    #[cfg(feature = "component-model")]
                    CliLinker::Component(_) => {
                        bail!("--preload cannot be used with components");
                    }
                }
            }

            self.load_main_module(&mut store, &mut linker, &main, profiled_modules)
                .await
                .with_context(|| {
                    format!(
                        "failed to run main module `{}`",
                        self.module_and_args[0].to_string_lossy()
                    )
                })
        })
        .await;

        // Load the main wasm module.
        match result.unwrap_or_else(|elapsed| {
            Err(anyhow::Error::from(wasmtime::Trap::Interrupt))
                .with_context(|| format!("timed out after {elapsed}"))
        }) {
            Ok(()) => (),
            Err(e) => {
                // Exit the process if Wasmtime understands the error;
                // otherwise, fall back on Rust's default error printing/return
                // code.
                if store.data().preview1_ctx.is_some() {
                    return Err(wasi_common::maybe_exit_on_error(e));
                } else if store.data().preview2_ctx.is_some() {
                    if let Some(exit) = e.downcast_ref::<wasmtime_wasi::I32Exit>() {
                        std::process::exit(exit.0);
                    }
                }
                if e.is::<wasmtime::Trap>() {
                    eprintln!("Error: {e:?}");
                    cfg_if::cfg_if! {
                        if #[cfg(unix)] {
                            std::process::exit(rustix::process::EXIT_SIGNALED_SIGABRT);
                        } else if #[cfg(windows)] {
                            // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/abort?view=vs-2019
                            std::process::exit(3);
                        }
                    }
                }
                return Err(e);
            }
        }

        Ok(())
    }

    fn compute_argv(&self) -> Result<Vec<String>> {
        let mut result = Vec::new();

        for (i, arg) in self.module_and_args.iter().enumerate() {
            // For argv[0], which is the program name. Only include the base
            // name of the main wasm module, to avoid leaking path information.
            let arg = if i == 0 {
                match &self.argv0 {
                    Some(s) => s.as_ref(),
                    None => Path::new(arg).components().next_back().unwrap().as_os_str(),
                }
            } else {
                arg.as_ref()
            };
            result.push(
                arg.to_str()
                    .ok_or_else(|| anyhow!("failed to convert {arg:?} to utf-8"))?
                    .to_string(),
            );
        }

        Ok(result)
    }

    fn setup_epoch_handler(
        &self,
        store: &mut Store<Host>,
        main_target: &RunTarget,
        profiled_modules: Vec<(String, Module)>,
    ) -> Result<Box<dyn FnOnce(&mut Store<Host>)>> {
        if let Some(Profile::Guest { path, interval }) = &self.run.profile {
            #[cfg(feature = "profiling")]
            return Ok(self.setup_guest_profiler(
                store,
                main_target,
                profiled_modules,
                path,
                *interval,
            ));
            #[cfg(not(feature = "profiling"))]
            {
                let _ = (profiled_modules, path, interval, main_target);
                bail!("support for profiling disabled at compile time");
            }
        }

        if let Some(timeout) = self.run.common.wasm.timeout {
            store.set_epoch_deadline(1);
            let engine = store.engine().clone();
            thread::spawn(move || {
                thread::sleep(timeout);
                engine.increment_epoch();
            });
        }

        Ok(Box::new(|_store| {}))
    }

    #[cfg(feature = "profiling")]
    fn setup_guest_profiler(
        &self,
        store: &mut Store<Host>,
        main_target: &RunTarget,
        profiled_modules: Vec<(String, Module)>,
        path: &str,
        interval: std::time::Duration,
    ) -> Box<dyn FnOnce(&mut Store<Host>)> {
        use wasmtime::{AsContext, GuestProfiler, StoreContext, StoreContextMut, UpdateDeadline};

        let module_name = self.module_and_args[0].to_str().unwrap_or("<main module>");
        store.data_mut().guest_profiler = match main_target {
            RunTarget::Core(_m) => Some(Arc::new(GuestProfiler::new(
                module_name,
                interval,
                profiled_modules,
            ))),
            RunTarget::Component(component) => Some(Arc::new(GuestProfiler::new_component(
                module_name,
                interval,
                component.clone(),
                profiled_modules,
            ))),
        };

        fn sample(
            mut store: StoreContextMut<Host>,
            f: impl FnOnce(&mut GuestProfiler, StoreContext<Host>),
        ) {
            let mut profiler = store.data_mut().guest_profiler.take().unwrap();
            f(
                Arc::get_mut(&mut profiler).expect("profiling doesn't support threads yet"),
                store.as_context(),
            );
            store.data_mut().guest_profiler = Some(profiler);
        }

        store.call_hook(|store, kind| {
            sample(store, |profiler, store| profiler.call_hook(store, kind));
            Ok(())
        });

        if let Some(timeout) = self.run.common.wasm.timeout {
            let mut timeout = (timeout.as_secs_f64() / interval.as_secs_f64()).ceil() as u64;
            assert!(timeout > 0);
            store.epoch_deadline_callback(move |store| {
                sample(store, |profiler, store| {
                    profiler.sample(store, std::time::Duration::ZERO)
                });
                timeout -= 1;
                if timeout == 0 {
                    bail!("timeout exceeded");
                }
                Ok(UpdateDeadline::Continue(1))
            });
        } else {
            store.epoch_deadline_callback(move |store| {
                sample(store, |profiler, store| {
                    profiler.sample(store, std::time::Duration::ZERO)
                });
                Ok(UpdateDeadline::Continue(1))
            });
        }

        store.set_epoch_deadline(1);
        let engine = store.engine().clone();
        thread::spawn(move || {
            loop {
                thread::sleep(interval);
                engine.increment_epoch();
            }
        });

        let path = path.to_string();
        return Box::new(move |store| {
            let profiler = Arc::try_unwrap(store.data_mut().guest_profiler.take().unwrap())
                .expect("profiling doesn't support threads yet");
            if let Err(e) = std::fs::File::create(&path)
                .map_err(anyhow::Error::new)
                .and_then(|output| profiler.finish(std::io::BufWriter::new(output)))
            {
                eprintln!("failed writing profile at {path}: {e:#}");
            } else {
                eprintln!();
                eprintln!("Profile written to: {path}");
                eprintln!("View this profile at https://profiler.firefox.com/.");
            }
        });
    }

    async fn load_and_invoke_component(
        &self,
        store: &mut Store<Host>,
        linker: &mut CliLinker,
        main_target: &RunTarget,
        function_call: &str,
    ) -> Result<Vec<wasmtime::component::Val>> {
        let component = main_target.unwrap_component();

        match linker {
            CliLinker::Core(_) => bail!("expected component, found core module".to_string()),

            #[cfg(feature = "component-model")]
            CliLinker::Component(linker) => {
                self.invoke_component_and_get_result(function_call, store, component, linker)
                    .await
            }
        }
    }

    async fn load_main_module(
        &self,
        store: &mut Store<Host>,
        linker: &mut CliLinker,
        main_target: &RunTarget,
        profiled_modules: Vec<(String, Module)>,
    ) -> Result<()> {
        // The main module might be allowed to have unknown imports, which
        // should be defined as traps:
        if self.run.common.wasm.unknown_imports_trap == Some(true) {
            match linker {
                CliLinker::Core(linker) => {
                    linker.define_unknown_imports_as_traps(main_target.unwrap_core())?;
                }
                #[cfg(feature = "component-model")]
                CliLinker::Component(linker) => {
                    linker.define_unknown_imports_as_traps(main_target.unwrap_component())?;
                }
            }
        }

        // ...or as default values.
        if self.run.common.wasm.unknown_imports_default == Some(true) {
            match linker {
                CliLinker::Core(linker) => {
                    linker.define_unknown_imports_as_default_values(
                        store,
                        main_target.unwrap_core(),
                    )?;
                }
                _ => bail!("cannot use `--default-values-unknown-imports` with components"),
            }
        }

        let finish_epoch_handler =
            self.setup_epoch_handler(store, main_target, profiled_modules)?;

        let result = match linker {
            CliLinker::Core(linker) => {
                let module = main_target.unwrap_core();
                let instance = linker
                    .instantiate_async(&mut *store, &module)
                    .await
                    .context(format!(
                        "failed to instantiate {:?}",
                        self.module_and_args[0]
                    ))?;

                // If `_initialize` is present, meaning a reactor, then invoke
                // the function.
                if let Some(func) = instance.get_func(&mut *store, "_initialize") {
                    func.typed::<(), ()>(&store)?
                        .call_async(&mut *store, ())
                        .await?;
                }

                // Look for the specific function provided or otherwise look for
                // "" or "_start" exports to run as a "main" function.
                let func = if let Some(name) = &self.invoke {
                    Some(
                        instance
                            .get_func(&mut *store, name)
                            .ok_or_else(|| anyhow!("no func export named `{}` found", name))?,
                    )
                } else {
                    instance
                        .get_func(&mut *store, "")
                        .or_else(|| instance.get_func(&mut *store, "_start"))
                };

                match func {
                    Some(func) => self.invoke_func(store, func).await,
                    None => Ok(()),
                }
            }
            #[cfg(feature = "component-model")]
            CliLinker::Component(linker) => {
                let component = main_target.unwrap_component();

                match &self.invoke {
                    Some(name) => {
                        self.invoke_component(&mut *store, component, linker, name)
                            .await
                    }

                    None => {
                        let command = wasmtime_wasi::p2::bindings::Command::instantiate_async(
                            &mut *store,
                            component,
                            linker,
                        )
                        .await?;

                        let result = command
                            .wasi_cli_run()
                            .call_run(&mut *store)
                            .await
                            .context("failed to invoke `run` function")
                            .map_err(|e| self.handle_core_dump(&mut *store, e));

                        // Translate the `Result<(),()>` produced by wasm into a feigned
                        // explicit exit here with status 1 if `Err(())` is returned.
                        result.and_then(|wasm_result| match wasm_result {
                            Ok(()) => Ok(()),
                            Err(()) => Err(wasmtime_wasi::I32Exit(1).into()),
                        })
                    }
                }
            }
        };
        finish_epoch_handler(store);

        result
    }

    #[cfg(feature = "component-model")]
    async fn invoke_component(
        &self,
        store: &mut Store<Host>,
        component: &wasmtime::component::Component,
        linker: &mut wasmtime::component::Linker<Host>,
        function_call: &str,
    ) -> Result<()> {
        use wasmtime::component::wasm_wave::wasm::DisplayFuncResults;

        let results = self
            .invoke_component_and_get_result(function_call, store, component, linker)
            .await?;

        println!("{}", DisplayFuncResults(&results));

        Ok(())
    }

    #[cfg(feature = "component-model")]
    async fn invoke_component_and_get_result(
        &self,
        invoke: &str,
        store: &mut Store<Host>,
        component: &wasmtime::component::Component,
        linker: &mut wasmtime::component::Linker<Host>,
    ) -> Result<Vec<wasmtime::component::Val>> {
        use wasmtime::component::{
            Val, types::ComponentItem, wasm_wave::untyped::UntypedFuncCall,
            wasm_wave::wasm::WasmFunc,
        };

        let untyped_call = UntypedFuncCall::parse(invoke).with_context(|| {
                format!(
                    "Failed to parse invoke '{invoke}': See https://docs.wasmtime.dev/cli-options.html#run for syntax",
                )
        })?;

        let name = untyped_call.name();
        let matches = Self::search_component(store.engine(), component.component_type(), name);
        match matches.len() {
            0 => bail!("No export named `{name}` in component."),
            1 => {}
            _ => bail!(
                "Multiple exports named `{name}`: {matches:?}. FIXME: support some way to disambiguate names"
            ),
        };
        let (params, result_len, export) = match &matches[0] {
            (names, ComponentItem::ComponentFunc(func)) => {
                let param_types = WasmFunc::params(func).collect::<Vec<_>>();
                let params = untyped_call.to_wasm_params(&param_types).with_context(|| {
                    format!("while interpreting parameters in invoke \"{invoke}\"")
                })?;
                let mut export = None;
                for name in names {
                    let ix = component
                        .get_export_index(export.as_ref(), name)
                        .expect("export exists");
                    export = Some(ix);
                }
                (
                    params,
                    func.results().len(),
                    export.expect("export has at least one name"),
                )
            }
            (names, ty) => {
                bail!("Cannot invoke export {names:?}: expected ComponentFunc, got type {ty:?}");
            }
        };

        let instance = linker.instantiate_async(&mut *store, component).await?;

        let func = instance
            .get_func(&mut *store, export)
            .expect("found export index");

        let mut results: Vec<Val> = vec![Val::Bool(false); result_len];
        func.call_async(&mut *store, &params, &mut results).await?;

        Ok(results)
    }

    #[cfg(feature = "component-model")]
    fn search_component(
        engine: &Engine,
        component: wasmtime::component::types::Component,
        name: &str,
    ) -> Vec<(Vec<String>, wasmtime::component::types::ComponentItem)> {
        use wasmtime::component::types::ComponentItem as CItem;
        fn collect_exports(
            engine: &Engine,
            item: CItem,
            basename: Vec<String>,
        ) -> Vec<(Vec<String>, CItem)> {
            match item {
                CItem::Component(c) => c
                    .exports(engine)
                    .flat_map(move |(name, item)| {
                        let mut names = basename.clone();
                        names.push(name.to_string());
                        collect_exports(engine, item, names)
                    })
                    .collect::<Vec<_>>(),
                CItem::ComponentInstance(c) => c
                    .exports(engine)
                    .flat_map(move |(name, item)| {
                        let mut names = basename.clone();
                        names.push(name.to_string());
                        collect_exports(engine, item, names)
                    })
                    .collect::<Vec<_>>(),
                _ => vec![(basename, item)],
            }
        }

        collect_exports(engine, CItem::Component(component), Vec::new())
            .into_iter()
            .filter(|(names, item)| names.last().expect("at least one name") == name)
            .collect()
    }

    async fn invoke_func(&self, store: &mut Store<Host>, func: Func) -> Result<()> {
        let ty = func.ty(&store);
        if ty.params().len() > 0 {
            eprintln!(
                "warning: using `--invoke` with a function that takes arguments \
                 is experimental and may break in the future"
            );
        }
        let mut args = self.module_and_args.iter().skip(1);
        let mut values = Vec::new();
        for ty in ty.params() {
            let val = match args.next() {
                Some(s) => s,
                None => {
                    if let Some(name) = &self.invoke {
                        bail!("not enough arguments for `{}`", name)
                    } else {
                        bail!("not enough arguments for command default")
                    }
                }
            };
            let val = val
                .to_str()
                .ok_or_else(|| anyhow!("argument is not valid utf-8: {val:?}"))?;
            values.push(match ty {
                // Supports both decimal and hexadecimal notation (with 0x prefix)
                ValType::I32 => Val::I32(if val.starts_with("0x") || val.starts_with("0X") {
                    i32::from_str_radix(&val[2..], 16)?
                } else {
                    val.parse::<i32>()?
                }),
                ValType::I64 => Val::I64(if val.starts_with("0x") || val.starts_with("0X") {
                    i64::from_str_radix(&val[2..], 16)?
                } else {
                    val.parse::<i64>()?
                }),
                ValType::F32 => Val::F32(val.parse::<f32>()?.to_bits()),
                ValType::F64 => Val::F64(val.parse::<f64>()?.to_bits()),
                t => bail!("unsupported argument type {:?}", t),
            });
        }

        // Invoke the function and then afterwards print all the results that came
        // out, if there are any.
        let mut results = vec![Val::null_func_ref(); ty.results().len()];
        let invoke_res = func
            .call_async(&mut *store, &values, &mut results)
            .await
            .with_context(|| {
                if let Some(name) = &self.invoke {
                    format!("failed to invoke `{name}`")
                } else {
                    format!("failed to invoke command default")
                }
            });

        if let Err(err) = invoke_res {
            return Err(self.handle_core_dump(&mut *store, err));
        }

        if !results.is_empty() {
            eprintln!(
                "warning: using `--invoke` with a function that returns values \
                 is experimental and may break in the future"
            );
        }

        for result in results {
            match result {
                Val::I32(i) => println!("{i}"),
                Val::I64(i) => println!("{i}"),
                Val::F32(f) => println!("{}", f32::from_bits(f)),
                Val::F64(f) => println!("{}", f64::from_bits(f)),
                Val::V128(i) => println!("{}", i.as_u128()),
                Val::ExternRef(None) => println!("<null externref>"),
                Val::ExternRef(Some(_)) => println!("<externref>"),
                Val::FuncRef(None) => println!("<null funcref>"),
                Val::FuncRef(Some(_)) => println!("<funcref>"),
                Val::AnyRef(None) => println!("<null anyref>"),
                Val::AnyRef(Some(_)) => println!("<anyref>"),
            }
        }

        Ok(())
    }

    #[cfg(feature = "coredump")]
    fn handle_core_dump(&self, store: &mut Store<Host>, err: Error) -> Error {
        let coredump_path = match &self.run.common.debug.coredump {
            Some(path) => path,
            None => return err,
        };
        if !err.is::<wasmtime::Trap>() {
            return err;
        }
        let source_name = self.module_and_args[0]
            .to_str()
            .unwrap_or_else(|| "unknown");

        if let Err(coredump_err) = write_core_dump(store, &err, &source_name, coredump_path) {
            eprintln!("warning: coredump failed to generate: {coredump_err}");
            err
        } else {
            err.context(format!("core dumped at {coredump_path}"))
        }
    }

    #[cfg(not(feature = "coredump"))]
    fn handle_core_dump(&self, _store: &mut Store<Host>, err: Error) -> Error {
        err
    }

    /// Populates the given `Linker` with WASI APIs.
    fn populate_with_wasi(
        &self,
        linker: &mut CliLinker,
        store: &mut Store<Host>,
        module: &RunTarget,
    ) -> Result<()> {
        let mut cli = self.run.common.wasi.cli;

        // Accept -Scommon as a deprecated alias for -Scli.
        if let Some(common) = self.run.common.wasi.common {
            if cli.is_some() {
                bail!(
                    "The -Scommon option should not be use with -Scli as it is a deprecated alias"
                );
            } else {
                // In the future, we may add a warning here to tell users to use
                // `-S cli` instead of `-S common`.
                cli = Some(common);
            }
        }

        if cli != Some(false) {
            match linker {
                CliLinker::Core(linker) => {
                    match (self.run.common.wasi.preview2, self.run.common.wasi.threads) {
                        // If preview2 is explicitly disabled, or if threads
                        // are enabled, then use the historical preview1
                        // implementation.
                        (Some(false), _) | (None, Some(true)) => {
                            wasi_common::tokio::add_to_linker(linker, |host| {
                                host.preview1_ctx.as_mut().unwrap()
                            })?;
                            self.set_preview1_ctx(store)?;
                        }
                        // If preview2 was explicitly requested, always use it.
                        // Otherwise use it so long as threads are disabled.
                        //
                        // Note that for now `preview0` is currently
                        // default-enabled but this may turn into
                        // default-disabled in the future.
                        (Some(true), _) | (None, Some(false) | None) => {
                            if self.run.common.wasi.preview0 != Some(false) {
                                wasmtime_wasi::preview0::add_to_linker_async(linker, |t| {
                                    t.preview2_ctx()
                                })?;
                            }
                            wasmtime_wasi::preview1::add_to_linker_async(linker, |t| {
                                t.preview2_ctx()
                            })?;
                            self.set_preview2_ctx(store)?;
                        }
                    }
                }
                #[cfg(feature = "component-model")]
                CliLinker::Component(linker) => {
                    let link_options = self.run.compute_wasi_features();
                    wasmtime_wasi::p2::add_to_linker_with_options_async(linker, &link_options)?;
                    self.set_preview2_ctx(store)?;
                }
            }
        }

        if self.run.common.wasi.nn == Some(true) {
            #[cfg(not(feature = "wasi-nn"))]
            {
                bail!("Cannot enable wasi-nn when the binary is not compiled with this feature.");
            }
            #[cfg(all(feature = "wasi-nn", feature = "component-model"))]
            {
                let (backends, registry) = self.collect_preloaded_nn_graphs()?;
                match linker {
                    CliLinker::Core(linker) => {
                        wasmtime_wasi_nn::witx::add_to_linker(linker, |host| {
                            Arc::get_mut(host.wasi_nn_witx.as_mut().unwrap())
                                .expect("wasi-nn is not implemented with multi-threading support")
                        })?;
                        store.data_mut().wasi_nn_witx = Some(Arc::new(
                            wasmtime_wasi_nn::witx::WasiNnCtx::new(backends, registry),
                        ));
                    }
                    #[cfg(feature = "component-model")]
                    CliLinker::Component(linker) => {
                        wasmtime_wasi_nn::wit::add_to_linker(linker, |h: &mut Host| {
                            let preview2_ctx =
                                h.preview2_ctx.as_mut().expect("wasip2 is not configured");
                            let preview2_ctx = Arc::get_mut(preview2_ctx)
                                .expect("wasmtime_wasi is not compatible with threads")
                                .get_mut()
                                .unwrap();
                            let nn_ctx = Arc::get_mut(h.wasi_nn_wit.as_mut().unwrap())
                                .expect("wasi-nn is not implemented with multi-threading support");
                            WasiNnView::new(preview2_ctx.table(), nn_ctx)
                        })?;
                        store.data_mut().wasi_nn_wit = Some(Arc::new(
                            wasmtime_wasi_nn::wit::WasiNnCtx::new(backends, registry),
                        ));
                    }
                }
            }
        }

        if self.run.common.wasi.config == Some(true) {
            #[cfg(not(feature = "wasi-config"))]
            {
                bail!(
                    "Cannot enable wasi-config when the binary is not compiled with this feature."
                );
            }
            #[cfg(all(feature = "wasi-config", feature = "component-model"))]
            {
                match linker {
                    CliLinker::Core(_) => {
                        bail!("Cannot enable wasi-config for core wasm modules");
                    }
                    CliLinker::Component(linker) => {
                        let vars = WasiConfigVariables::from_iter(
                            self.run
                                .common
                                .wasi
                                .config_var
                                .iter()
                                .map(|v| (v.key.clone(), v.value.clone())),
                        );

                        wasmtime_wasi_config::add_to_linker(linker, |h| {
                            WasiConfig::new(Arc::get_mut(h.wasi_config.as_mut().unwrap()).unwrap())
                        })?;
                        store.data_mut().wasi_config = Some(Arc::new(vars));
                    }
                }
            }
        }

        if self.run.common.wasi.keyvalue == Some(true) {
            #[cfg(not(feature = "wasi-keyvalue"))]
            {
                bail!(
                    "Cannot enable wasi-keyvalue when the binary is not compiled with this feature."
                );
            }
            #[cfg(all(feature = "wasi-keyvalue", feature = "component-model"))]
            {
                match linker {
                    CliLinker::Core(_) => {
                        bail!("Cannot enable wasi-keyvalue for core wasm modules");
                    }
                    CliLinker::Component(linker) => {
                        let ctx = WasiKeyValueCtxBuilder::new()
                            .in_memory_data(
                                self.run
                                    .common
                                    .wasi
                                    .keyvalue_in_memory_data
                                    .iter()
                                    .map(|v| (v.key.clone(), v.value.clone())),
                            )
                            .build();

                        wasmtime_wasi_keyvalue::add_to_linker(linker, |h| {
                            let preview2_ctx =
                                h.preview2_ctx.as_mut().expect("wasip2 is not configured");
                            let preview2_ctx =
                                Arc::get_mut(preview2_ctx).unwrap().get_mut().unwrap();
                            WasiKeyValue::new(
                                Arc::get_mut(h.wasi_keyvalue.as_mut().unwrap()).unwrap(),
                                preview2_ctx.table(),
                            )
                        })?;
                        store.data_mut().wasi_keyvalue = Some(Arc::new(ctx));
                    }
                }
            }
        }

        if self.run.common.wasi.threads == Some(true) {
            #[cfg(not(feature = "wasi-threads"))]
            {
                // Silence the unused warning for `module` as it is only used in the
                // conditionally-compiled wasi-threads.
                let _ = &module;

                bail!(
                    "Cannot enable wasi-threads when the binary is not compiled with this feature."
                );
            }
            #[cfg(feature = "wasi-threads")]
            {
                let linker = match linker {
                    CliLinker::Core(linker) => linker,
                    _ => bail!("wasi-threads does not support components yet"),
                };
                let module = module.unwrap_core();
                wasmtime_wasi_threads::add_to_linker(linker, store, &module, |host| {
                    host.wasi_threads.as_ref().unwrap()
                })?;
                store.data_mut().wasi_threads = Some(Arc::new(WasiThreadsCtx::new(
                    module.clone(),
                    Arc::new(linker.clone()),
                )?));
            }
        }

        if self.run.common.wasi.http == Some(true) {
            #[cfg(not(all(feature = "wasi-http", feature = "component-model")))]
            {
                bail!("Cannot enable wasi-http when the binary is not compiled with this feature.");
            }
            #[cfg(all(feature = "wasi-http", feature = "component-model"))]
            {
                match linker {
                    CliLinker::Core(_) => {
                        bail!("Cannot enable wasi-http for core wasm modules");
                    }
                    CliLinker::Component(linker) => {
                        wasmtime_wasi_http::add_only_http_to_linker_sync(linker)?;
                    }
                }

                store.data_mut().wasi_http = Some(Arc::new(WasiHttpCtx::new()));
            }
        }

        if self.run.common.wasi.tls == Some(true) {
            #[cfg(all(not(all(feature = "wasi-tls", feature = "component-model"))))]
            {
                bail!("Cannot enable wasi-tls when the binary is not compiled with this feature.");
            }
            #[cfg(all(feature = "wasi-tls", feature = "component-model",))]
            {
                match linker {
                    CliLinker::Core(_) => {
                        bail!("Cannot enable wasi-tls for core wasm modules");
                    }
                    CliLinker::Component(linker) => {
                        let mut opts = wasmtime_wasi_tls::LinkOptions::default();
                        opts.tls(true);
                        wasmtime_wasi_tls::add_to_linker(linker, &mut opts, |h| {
                            let preview2_ctx =
                                h.preview2_ctx.as_mut().expect("wasip2 is not configured");
                            let preview2_ctx =
                                Arc::get_mut(preview2_ctx).unwrap().get_mut().unwrap();
                            WasiTlsCtx::new(preview2_ctx.table())
                        })?;
                    }
                }
            }
        }

        Ok(())
    }

    fn set_preview1_ctx(&self, store: &mut Store<Host>) -> Result<()> {
        let mut builder = WasiCtxBuilder::new();
        builder.inherit_stdio().args(&self.compute_argv()?)?;

        if self.run.common.wasi.inherit_env == Some(true) {
            for (k, v) in std::env::vars() {
                builder.env(&k, &v)?;
            }
        }
        for (key, value) in self.run.vars.iter() {
            let value = match value {
                Some(value) => value.clone(),
                None => match std::env::var_os(key) {
                    Some(val) => val
                        .into_string()
                        .map_err(|_| anyhow!("environment variable `{key}` not valid utf-8"))?,
                    None => {
                        // leave the env var un-set in the guest
                        continue;
                    }
                },
            };
            builder.env(key, &value)?;
        }

        let mut num_fd: usize = 3;

        if self.run.common.wasi.listenfd == Some(true) {
            num_fd = ctx_set_listenfd(num_fd, &mut builder)?;
        }

        for listener in self.run.compute_preopen_sockets()? {
            let listener = TcpListener::from_std(listener);
            builder.preopened_socket(num_fd as _, listener)?;
            num_fd += 1;
        }

        for (host, guest) in self.run.dirs.iter() {
            let dir = Dir::open_ambient_dir(host, ambient_authority())
                .with_context(|| format!("failed to open directory '{host}'"))?;
            builder.preopened_dir(dir, guest)?;
        }

        store.data_mut().preview1_ctx = Some(builder.build());
        Ok(())
    }

    fn set_preview2_ctx(&self, store: &mut Store<Host>) -> Result<()> {
        let mut builder = wasmtime_wasi::p2::WasiCtxBuilder::new();
        builder.inherit_stdio().args(&self.compute_argv()?);
        self.run.configure_wasip2(&mut builder)?;
        let ctx = builder.build_p1();
        store.data_mut().preview2_ctx = Some(Arc::new(Mutex::new(ctx)));
        Ok(())
    }

    #[cfg(feature = "wasi-nn")]
    fn collect_preloaded_nn_graphs(
        &self,
    ) -> Result<(Vec<wasmtime_wasi_nn::Backend>, wasmtime_wasi_nn::Registry)> {
        let graphs = self
            .run
            .common
            .wasi
            .nn_graph
            .iter()
            .map(|g| (g.format.clone(), g.dir.clone()))
            .collect::<Vec<_>>();
        wasmtime_wasi_nn::preload(&graphs)
    }
}

#[derive(Default, Clone)]
struct Host {
    preview1_ctx: Option<wasi_common::WasiCtx>,

    // The Mutex is only needed to satisfy the Sync constraint but we never
    // actually perform any locking on it as we use Mutex::get_mut for every
    // access.
    preview2_ctx: Option<Arc<Mutex<wasmtime_wasi::preview1::WasiP1Ctx>>>,

    #[cfg(feature = "wasi-nn")]
    wasi_nn_wit: Option<Arc<wasmtime_wasi_nn::wit::WasiNnCtx>>,
    #[cfg(feature = "wasi-nn")]
    wasi_nn_witx: Option<Arc<wasmtime_wasi_nn::witx::WasiNnCtx>>,

    #[cfg(feature = "wasi-threads")]
    wasi_threads: Option<Arc<WasiThreadsCtx<Host>>>,
    #[cfg(feature = "wasi-http")]
    wasi_http: Option<Arc<WasiHttpCtx>>,
    #[cfg(feature = "wasi-http")]
    wasi_http_outgoing_body_buffer_chunks: Option<usize>,
    #[cfg(feature = "wasi-http")]
    wasi_http_outgoing_body_chunk_size: Option<usize>,
    limits: StoreLimits,
    #[cfg(feature = "profiling")]
    guest_profiler: Option<Arc<wasmtime::GuestProfiler>>,

    #[cfg(feature = "wasi-config")]
    wasi_config: Option<Arc<WasiConfigVariables>>,
    #[cfg(feature = "wasi-keyvalue")]
    wasi_keyvalue: Option<Arc<WasiKeyValueCtx>>,
}

impl Host {
    fn preview2_ctx(&mut self) -> &mut wasmtime_wasi::preview1::WasiP1Ctx {
        let ctx = self
            .preview2_ctx
            .as_mut()
            .expect("wasip2 is not configured");
        Arc::get_mut(ctx)
            .expect("wasmtime_wasi is not compatible with threads")
            .get_mut()
            .unwrap()
    }
}

impl IoView for Host {
    fn table(&mut self) -> &mut wasmtime::component::ResourceTable {
        self.preview2_ctx().table()
    }
}
impl WasiView for Host {
    fn ctx(&mut self) -> &mut wasmtime_wasi::p2::WasiCtx {
        self.preview2_ctx().ctx()
    }
}

#[cfg(feature = "wasi-http")]
impl wasmtime_wasi_http::types::WasiHttpView for Host {
    fn ctx(&mut self) -> &mut WasiHttpCtx {
        let ctx = self.wasi_http.as_mut().unwrap();
        Arc::get_mut(ctx).expect("wasmtime_wasi is not compatible with threads")
    }

    fn outgoing_body_buffer_chunks(&mut self) -> usize {
        self.wasi_http_outgoing_body_buffer_chunks
            .unwrap_or_else(|| DEFAULT_OUTGOING_BODY_BUFFER_CHUNKS)
    }

    fn outgoing_body_chunk_size(&mut self) -> usize {
        self.wasi_http_outgoing_body_chunk_size
            .unwrap_or_else(|| DEFAULT_OUTGOING_BODY_CHUNK_SIZE)
    }
}

#[cfg(not(unix))]
fn ctx_set_listenfd(num_fd: usize, _builder: &mut WasiCtxBuilder) -> Result<usize> {
    Ok(num_fd)
}

#[cfg(unix)]
fn ctx_set_listenfd(mut num_fd: usize, builder: &mut WasiCtxBuilder) -> Result<usize> {
    use listenfd::ListenFd;

    for env in ["LISTEN_FDS", "LISTEN_FDNAMES"] {
        if let Ok(val) = std::env::var(env) {
            builder.env(env, &val)?;
        }
    }

    let mut listenfd = ListenFd::from_env();

    for i in 0..listenfd.len() {
        if let Some(stdlistener) = listenfd.take_tcp_listener(i)? {
            let _ = stdlistener.set_nonblocking(true)?;
            let listener = TcpListener::from_std(stdlistener);
            builder.preopened_socket((3 + i) as _, listener)?;
            num_fd = 3 + i;
        }
    }

    Ok(num_fd)
}

#[cfg(feature = "coredump")]
fn write_core_dump(
    store: &mut Store<Host>,
    err: &anyhow::Error,
    name: &str,
    path: &str,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let core_dump = err
        .downcast_ref::<wasmtime::WasmCoreDump>()
        .expect("should have been configured to capture core dumps");

    let core_dump = core_dump.serialize(store, name);

    let mut core_dump_file =
        File::create(path).context(format!("failed to create file at `{path}`"))?;
    core_dump_file
        .write_all(&core_dump)
        .with_context(|| format!("failed to write core dump file at `{path}`"))?;
    Ok(())
}
