//! Rib language REPL for WebAssembly components.
//!
//! Enabled with the `rib` Cargo feature: `cargo build -p wasmtime-cli --features rib`.

#![cfg(feature = "rib")]

use super::run::{CliLinker, Host, Preloads, RunCommand};
use crate::common::{RunCommon, RunTarget};
use async_trait::async_trait;
use clap::Parser;
use rib_repl::anyhow::Context as _;
use rib_repl::anyhow::{Result, anyhow, bail};
use rib_repl::uuid::Uuid;
use rib_repl::wit_type::*;
use rib_repl::*;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use wasmtime::component::types::{self, ComponentItem as CItem, Type as WType};
use wasmtime::component::{Component, ComponentExportIndex, Instance, Linker, Val};
use wasmtime::{Engine, Store};

/// Start an interactive Rib REPL against a WebAssembly component.
#[derive(Parser)]
pub struct ReplCommand {
    #[command(flatten)]
    #[expect(missing_docs, reason = "reuse run command flags")]
    pub run: RunCommon,

    /// WebAssembly component file (`.wasm`).
    #[arg(value_name = "WASM")]
    pub component: PathBuf,
}

impl ReplCommand {
    /// Run the Rib REPL.
    pub fn execute(mut self) -> wasmtime::Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_time()
            .enable_io()
            .build()
            .map_err(|e| wasmtime::Error::msg(format!("tokio runtime: {e}")))?;

        self.run.common.init_logging()?;

        let ReplCommand {
            run,
            component: wasm_path,
        } = self;

        let component_name = wasm_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("component")
            .to_string();

        runtime.block_on(async move {
            let mut run_cmd = RunCommand {
                run,
                invoke: None,
                preloads: Preloads::default(),
                argv0: None,
                module_bytes: None,
                module_and_args: vec![wasm_path.as_os_str().into()],
            };

            let engine = run_cmd.new_engine()?;
            let main = run_cmd.run.load_module(&engine, &wasm_path, None)?;
            let (store, linker) = run_cmd.new_store_and_linker(&engine, &main)?;
            let RunTarget::Component(component) = main else {
                return Err(wasmtime::Error::msg(
                    "`wasmtime repl` expects a WebAssembly component, not a core module",
                ));
            };

            let cli_linker = match linker {
                CliLinker::Component(l) => l,
                CliLinker::Core(_) => {
                    return Err(wasmtime::Error::msg("expected component linker"));
                }
            };

            let component_id = Uuid::new_v4();

            let store = Arc::new(Mutex::new(store));
            let dep_manager = Arc::new(WasmtimeRibDependencyManager {
                engine: engine.clone(),
                component_id,
            });
            let invoke = Arc::new(WasmtimeWorkerInvoke {
                component,
                linker: cli_linker,
                store,
                instances: Mutex::new(HashMap::new()),
                component_id,
            });

            let mut repl = RibRepl::bootstrap(RibReplConfig {
                history_file: None,
                dependency_manager: dep_manager,
                worker_function_invoke: invoke,
                printer: None,
                component_source: Some(ComponentSource {
                    component_name,
                    source_path: wasm_path.clone(),
                }),
                prompt: None,
                command_registry: None,
            })
            .await
            .map_err(|e| wasmtime::Error::msg(e.to_string()))?;

            repl.run().await;
            Ok(())
        })
    }
}

struct WasmtimeRibDependencyManager {
    engine: Engine,
    component_id: Uuid,
}

#[async_trait]
impl RibDependencyManager for WasmtimeRibDependencyManager {
    async fn get_dependencies(&self) -> Result<ReplComponentBundle> {
        bail!("load a component via `wasmtime repl <component.wasm>` (no multi-project mode yet)")
    }

    async fn add_component(
        &self,
        source_path: &Path,
        component_name: String,
    ) -> Result<ComponentDependency> {
        let bytes = std::fs::read(source_path)
            .with_context(|| format!("failed to read {}", source_path.display()))?;
        // Compile a component from wasm/WAT bytes (same as `Component::new` / `wasmtime run`).
        // `Component::deserialize` is only for precompiled artifacts (ELF), not `.wasm` binaries.
        let comp = Component::new(&self.engine, &bytes).map_err(|e| anyhow!("{e:?}"))?;
        let exports = component_exports(&self.engine, comp.component_type())?;
        ComponentDependency::from_wit_metadata(
            ComponentDependencyKey {
                component_name,
                component_id: self.component_id,
                component_revision: 0,
                root_package_name: None,
                root_package_version: None,
            },
            &exports,
        )
        .map_err(|e| anyhow!("{e}"))
    }
}

struct WasmtimeWorkerInvoke {
    component: Component,
    linker: Linker<Host>,
    store: Arc<Mutex<Store<Host>>>,
    /// One component [`Instance`] per Rib worker name (`instance()`, `instance("x")`, …).
    instances: Mutex<HashMap<String, Instance>>,
    component_id: Uuid,
}

impl WasmtimeWorkerInvoke {
    async fn instance_for(&self, worker_name: &str) -> Result<Instance> {
        if let Some(&i) = self.instances.lock().await.get(worker_name) {
            return Ok(i);
        }

        let mut store = self.store.lock().await;
        let new_inst = self
            .linker
            .instantiate_async(&mut *store, &self.component)
            .await
            .map_err(|e| anyhow!("{e:?}"))?;
        drop(store);

        let mut instances = self.instances.lock().await;
        if let Some(&i) = instances.get(worker_name) {
            return Ok(i);
        }
        instances.insert(worker_name.to_string(), new_inst);
        Ok(new_inst)
    }
}

#[async_trait]
impl ComponentFunctionInvoke for WasmtimeWorkerInvoke {
    async fn invoke(
        &self,
        component_id: Uuid,
        _component_name: &str,
        worker_name: &str,
        function_name: &str,
        args: Vec<RibVal>,
        _return_type: Option<WitType>,
    ) -> Result<Option<RibVal>> {
        if component_id != self.component_id {
            bail!("unexpected component id (only one component is supported)");
        }

        let parsed = ParsedFunctionName::parse(function_name)
            .map_err(|e| anyhow!("invalid function name `{function_name}`: {e}"))?;

        let path = export_path(&parsed);
        let export = resolve_export(&self.component, &path)
            .with_context(|| format!("resolve export for `{function_name}`"))?;

        let instance = self.instance_for(worker_name).await?;

        let mut store = self.store.lock().await;
        let func = instance
            .get_func(&mut *store, export)
            .ok_or_else(|| anyhow!("export is not a function"))?;

        let fty = func.ty(&*store);
        let n_params = fty.params().count();
        let n_results = fty.results().count();

        if n_params != args.len() {
            bail!("expected {} arguments, got {}", n_params, args.len());
        }

        let mut params = Vec::with_capacity(args.len());
        for arg in &args {
            params.push(rib_val_to_val(arg)?);
        }

        let mut results: Vec<Val> = (0..n_results).map(|_| Val::Bool(false)).collect();

        func.call_async(&mut *store, &params, &mut results)
            .await
            .map_err(|e| anyhow!("{e:?}"))?;

        let out = match results.len() {
            0 => None,
            1 => Some(val_to_rib_val(&results[0])?),
            _ => {
                let parts: Result<Vec<RibVal>> = results.iter().map(val_to_rib_val).collect();
                Some(RibVal::Tuple(parts?))
            }
        };

        Ok(out)
    }
}

fn export_path(parsed: &ParsedFunctionName) -> Vec<String> {
    let mut segments: Vec<String> = match &parsed.site {
        ParsedFunctionSite::Global => Vec::new(),
        ParsedFunctionSite::Interface { name } => name.split('/').map(str::to_string).collect(),
        ParsedFunctionSite::PackagedInterface { .. } => parsed
            .site
            .interface_name()
            .expect("packaged interface has name")
            .split('/')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
    };
    segments.push(parsed.function.function_name());
    segments
}

fn resolve_export(component: &Component, path: &[String]) -> Result<ComponentExportIndex> {
    if path.is_empty() {
        bail!("empty export path");
    }
    let mut instance: Option<ComponentExportIndex> = None;
    for name in &path[..path.len() - 1] {
        instance = Some(
            component
                .get_export_index(instance.as_ref(), name)
                .ok_or_else(|| anyhow!("missing instance export `{name}`"))?,
        );
    }
    let last = path.last().unwrap();
    component
        .get_export_index(instance.as_ref(), last)
        .ok_or_else(|| anyhow!("missing function export `{last}`"))
}

fn component_exports(engine: &Engine, component: types::Component) -> Result<Vec<WitExport>> {
    let funcs = collect_component_funcs(engine, component);
    let mut root_funcs = Vec::new();
    let mut by_instance: BTreeMap<String, Vec<WitFunction>> = BTreeMap::new();

    for (path, cf) in funcs {
        if cf.async_() {
            continue;
        }
        let af = component_func_to_wit(&path, &cf)?;
        if path.len() == 1 {
            root_funcs.push(WitExport::Function(af));
        } else {
            let iface = path[..path.len() - 1].join("/");
            by_instance.entry(iface).or_default().push(af);
        }
    }

    let mut out: Vec<WitExport> = root_funcs;
    for (name, functions) in by_instance {
        out.push(WitExport::Interface(WitInterface { name, functions }));
    }
    Ok(out)
}

fn collect_component_funcs(
    engine: &Engine,
    component: types::Component,
) -> Vec<(Vec<String>, types::ComponentFunc)> {
    fn walk(engine: &Engine, item: CItem, prefix: Vec<String>) -> Vec<(Vec<String>, CItem)> {
        match item {
            CItem::Component(c) => c
                .exports(engine)
                .flat_map(|(n, it)| {
                    let mut p = prefix.clone();
                    p.push(n.to_string());
                    walk(engine, it, p)
                })
                .collect(),
            CItem::ComponentInstance(c) => c
                .exports(engine)
                .flat_map(|(n, it)| {
                    let mut p = prefix.clone();
                    p.push(n.to_string());
                    walk(engine, it, p)
                })
                .collect(),
            _ => vec![(prefix, item)],
        }
    }

    walk(engine, CItem::Component(component), Vec::new())
        .into_iter()
        .filter_map(|(names, item)| match item {
            CItem::ComponentFunc(f) => Some((names, f)),
            _ => None,
        })
        .collect()
}

fn component_func_to_wit(path: &[String], f: &types::ComponentFunc) -> Result<WitFunction> {
    let name = path.last().expect("func path").clone();
    let parameters = f
        .params()
        .map(|(n, t)| {
            Ok(WitFunctionParameter {
                name: n.to_string(),
                typ: wasm_type_to_wit(&t)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let result = match f.results().len() {
        0 => None,
        1 => Some(WitFunctionResult {
            typ: wasm_type_to_wit(&f.results().next().unwrap())?,
        }),
        _ => {
            let items: Vec<WitType> = f
                .results()
                .map(|t| wasm_type_to_wit(&t))
                .collect::<Result<_>>()?;
            Some(WitFunctionResult {
                typ: WitType::Tuple(TypeTuple {
                    name: None,
                    owner: None,
                    items,
                }),
            })
        }
    };

    Ok(WitFunction {
        name,
        parameters,
        result,
    })
}

fn wasm_type_to_wit(ty: &WType) -> Result<WitType> {
    Ok(match ty {
        WType::Bool => WitType::Bool(TypeBool),
        WType::S8 => WitType::S8(TypeS8),
        WType::U8 => WitType::U8(TypeU8),
        WType::S16 => WitType::S16(TypeS16),
        WType::U16 => WitType::U16(TypeU16),
        WType::S32 => WitType::S32(TypeS32),
        WType::U32 => WitType::U32(TypeU32),
        WType::S64 => WitType::S64(TypeS64),
        WType::U64 => WitType::U64(TypeU64),
        WType::Float32 => WitType::F32(TypeF32),
        WType::Float64 => WitType::F64(TypeF64),
        WType::Char => WitType::Chr(TypeChr),
        WType::String => WitType::Str(TypeStr),
        WType::List(l) => WitType::List(TypeList {
            name: None,
            owner: None,
            inner: Box::new(wasm_type_to_wit(&l.ty())?),
        }),
        WType::Record(r) => {
            let fields = r
                .fields()
                .map(|fld| {
                    Ok(NameTypePair {
                        name: fld.name.to_string(),
                        typ: wasm_type_to_wit(&fld.ty)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            WitType::Record(TypeRecord {
                name: None,
                owner: None,
                fields,
            })
        }
        WType::Tuple(t) => {
            let items = t
                .types()
                .map(|ty| wasm_type_to_wit(&ty))
                .collect::<Result<Vec<_>>>()?;
            WitType::Tuple(TypeTuple {
                name: None,
                owner: None,
                items,
            })
        }
        WType::Variant(v) => {
            let cases = v
                .cases()
                .map(|c| {
                    Ok(NameOptionTypePair {
                        name: c.name.to_string(),
                        typ: c.ty.map(|t| wasm_type_to_wit(&t)).transpose()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            WitType::Variant(TypeVariant {
                name: None,
                owner: None,
                cases,
            })
        }
        WType::Enum(e) => WitType::Enum(TypeEnum {
            name: None,
            owner: None,
            cases: e.names().map(str::to_string).collect(),
        }),
        WType::Option(o) => WitType::Option(TypeOption {
            name: None,
            owner: None,
            inner: Box::new(wasm_type_to_wit(&o.ty())?),
        }),
        WType::Result(r) => WitType::Result(TypeResult {
            name: None,
            owner: None,
            ok: r
                .ok()
                .map(|t| wasm_type_to_wit(&t))
                .transpose()?
                .map(Box::new),
            err: r
                .err()
                .map(|t| wasm_type_to_wit(&t))
                .transpose()?
                .map(Box::new),
        }),
        WType::Flags(fl) => WitType::Flags(TypeFlags {
            name: None,
            owner: None,
            names: fl.names().map(str::to_string).collect(),
        }),
        WType::Own(_) => WitType::Handle(TypeHandle {
            name: None,
            owner: None,
            resource_id: AnalysedResourceId(0),
            mode: AnalysedResourceMode::Owned,
        }),
        WType::Borrow(_) => WitType::Handle(TypeHandle {
            name: None,
            owner: None,
            resource_id: AnalysedResourceId(0),
            mode: AnalysedResourceMode::Borrowed,
        }),
        WType::Map(_) => {
            bail!("Rib metadata does not support WIT `map` yet")
        }
        WType::Future(_) | WType::Stream(_) | WType::ErrorContext => {
            bail!("async component types are not supported in Rib metadata yet")
        }
    })
}

/// [`RibVal`] ↔ Wasmtime [`Val`] by shape only — no WIT / `expected` type.
fn rib_val_to_val(rv: &RibVal) -> Result<Val> {
    use RibVal as R;
    Ok(match rv {
        R::Bool(b) => Val::Bool(*b),
        R::S8(x) => Val::S8(*x),
        R::U8(x) => Val::U8(*x),
        R::S16(x) => Val::S16(*x),
        R::U16(x) => Val::U16(*x),
        R::S32(x) => Val::S32(*x),
        R::U32(x) => Val::U32(*x),
        R::S64(x) => Val::S64(*x),
        R::U64(x) => Val::U64(*x),
        R::Float32(x) => Val::Float32(*x),
        R::Float64(x) => Val::Float64(*x),
        R::Char(c) => Val::Char(*c),
        R::String(s) => Val::String(s.clone()),
        R::List(items) => Val::List(items.iter().map(rib_val_to_val).collect::<Result<_>>()?),
        R::Record(pairs) => Val::Record(
            pairs
                .iter()
                .map(|(n, v)| Ok((n.clone(), rib_val_to_val(v)?)))
                .collect::<Result<_>>()?,
        ),
        R::Tuple(items) => Val::Tuple(items.iter().map(rib_val_to_val).collect::<Result<_>>()?),
        R::Variant(name, payload) => {
            let p = match payload {
                None => None,
                Some(b) => Some(Box::new(rib_val_to_val(b)?)),
            };
            Val::Variant(name.clone(), p)
        }
        R::Enum(name) => Val::Enum(name.clone()),
        R::Option(inner) => Val::Option(match inner {
            None => None,
            Some(b) => Some(Box::new(rib_val_to_val(b)?)),
        }),
        R::Result(inner) => Val::Result(match inner {
            Ok(v) => Ok(match v {
                None => None,
                Some(b) => Some(Box::new(rib_val_to_val(b)?)),
            }),
            Err(v) => Err(match v {
                None => None,
                Some(b) => Some(Box::new(rib_val_to_val(b)?)),
            }),
        }),
        R::Flags(names) => Val::Flags(names.clone()),
        R::Handle { .. } => bail!("resource handles are not supported in `wasmtime repl` yet"),
    })
}

fn val_to_rib_val(v: &Val) -> Result<RibVal> {
    use RibVal as R;
    Ok(match v {
        Val::Bool(b) => R::Bool(*b),
        Val::S8(x) => R::S8(*x),
        Val::U8(x) => R::U8(*x),
        Val::S16(x) => R::S16(*x),
        Val::U16(x) => R::U16(*x),
        Val::S32(x) => R::S32(*x),
        Val::U32(x) => R::U32(*x),
        Val::S64(x) => R::S64(*x),
        Val::U64(x) => R::U64(*x),
        Val::Float32(x) => R::Float32(*x),
        Val::Float64(x) => R::Float64(*x),
        Val::Char(c) => R::Char(*c),
        Val::String(s) => R::String(s.clone()),
        Val::List(items) => R::List(items.iter().map(val_to_rib_val).collect::<Result<_>>()?),
        Val::Record(pairs) => R::Record(
            pairs
                .iter()
                .map(|(n, v)| Ok((n.clone(), val_to_rib_val(v)?)))
                .collect::<Result<_>>()?,
        ),
        Val::Tuple(items) => R::Tuple(items.iter().map(val_to_rib_val).collect::<Result<_>>()?),
        Val::Variant(name, payload) => R::Variant(
            name.clone(),
            match payload {
                None => None,
                Some(b) => Some(Box::new(val_to_rib_val(b)?)),
            },
        ),
        Val::Enum(name) => R::Enum(name.clone()),
        Val::Option(inner) => R::Option(match inner {
            None => None,
            Some(b) => Some(Box::new(val_to_rib_val(b)?)),
        }),
        Val::Result(inner) => R::Result(match inner {
            Ok(v) => Ok(match v {
                None => None,
                Some(b) => Some(Box::new(val_to_rib_val(b)?)),
            }),
            Err(v) => Err(match v {
                None => None,
                Some(b) => Some(Box::new(val_to_rib_val(b)?)),
            }),
        }),
        Val::Flags(names) => R::Flags(names.clone()),
        Val::Resource(_) => bail!("component resources are not supported in `wasmtime repl` yet"),
        Val::Map(_) => bail!("WIT maps are not supported in `wasmtime repl` yet"),
        Val::Future(_) | Val::Stream(_) | Val::ErrorContext(_) => {
            bail!("async values are not supported in `wasmtime repl` yet")
        }
    })
}
