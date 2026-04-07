//! Rib language REPL for WebAssembly components.
//!
//! Enabled with the `rib` Cargo feature: `cargo build -p wasmtime-cli --features rib`.

#![cfg(feature = "rib")]

use super::run::{CliLinker, Host, Preloads, RunCommand};
use crate::common::{RunCommon, RunTarget};
use async_trait::async_trait;
use clap::Parser;
use rib::analysis::{
    AnalysedExport, AnalysedFunction, AnalysedFunctionParameter, AnalysedFunctionResult,
    AnalysedInstance, AnalysedResourceId, AnalysedResourceMode, AnalysedType, NameOptionTypePair,
    NameTypePair, TypeBool, TypeChr, TypeEnum, TypeF32, TypeF64, TypeFlags, TypeHandle, TypeList,
    TypeOption, TypeRecord, TypeResult, TypeS16, TypeS32, TypeS64, TypeS8, TypeStr, TypeTuple,
    TypeU16, TypeU32, TypeU64, TypeU8, TypeVariant,
};
use rib::{
    ComponentDependency, ComponentDependencyKey, ParsedFunctionName, ParsedFunctionSite, Value,
    ValueAndType,
};
use rib_repl::{
    self as rib_repl_crate, anyhow::Context as _, ComponentSource, ReplComponentDependencies,
    RibDependencyManager, RibRepl, RibReplConfig, WorkerFunctionInvoke,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use wasmtime::component::types::{self, ComponentItem as CItem, Type as WType};
use wasmtime::component::{Component, ComponentExportIndex, Func, Instance, Val};
use wasmtime::{Engine, Result, Store};

/// Start an interactive Rib REPL against a WebAssembly component.
#[derive(Parser, Clone)]
pub struct RibCommand {
    #[command(flatten)]
    #[expect(missing_docs, reason = "reuse run command flags")]
    pub run: RunCommon,

    /// Logical component name in Rib (default: file stem).
    #[arg(long = "rib-name")]
    pub name: Option<String>,

    /// WebAssembly component file (`.wasm`).
    #[arg(value_name = "WASM")]
    pub component: PathBuf,
}

impl RibCommand {
    /// Run the Rib REPL.
    pub fn execute(mut self) -> Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_time()
            .enable_io()
            .build()
            .map_err(|e| wasmtime::Error::msg(format!("tokio runtime: {e}")))?;

        runtime.block_on(async {
            self.run.common.init_logging()?;

            let component_name = self.name.clone().unwrap_or_else(|| {
                self.component
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("component")
                    .to_string()
            });

            let mut run_cmd = RunCommand {
                run: self.run.clone(),
                invoke: None,
                preloads: Preloads::default(),
                argv0: None,
                module_bytes: None,
                module_and_args: vec![self.component.as_os_str().into()],
            };

            let engine = run_cmd.new_engine()?;
            let main = run_cmd.run.load_module(&engine, &self.component, None)?;
            let (mut store, linker) = run_cmd.new_store_and_linker(&engine, &main)?;
            let RunTarget::Component(component) = main else {
                return Err(wasmtime::Error::msg(
                    "`wasmtime rib` expects a WebAssembly component, not a core module",
                ));
            };

            let cli_linker = match linker {
                CliLinker::Component(l) => l,
                CliLinker::Core(_) => {
                    return Err(wasmtime::Error::msg("expected component linker"));
                }
            };

            let instance = cli_linker
                .instantiate_async(&mut store, &component)
                .await
                .map_err(|e| wasmtime::Error::msg(format!("{e:?}")))?;

            let component_id = rib_repl_crate::uuid::Uuid::new_v4();

            let store = Arc::new(Mutex::new(store));
            let dep_manager = Arc::new(WasmtimeRibDependencyManager {
                engine: engine.clone(),
                component_id,
            });
            let invoke = Arc::new(WasmtimeWorkerInvoke {
                component,
                instance,
                store,
                component_id,
            });

            let mut repl = RibRepl::bootstrap(RibReplConfig {
                history_file: None,
                dependency_manager: dep_manager,
                worker_function_invoke: invoke,
                printer: None,
                component_source: Some(ComponentSource {
                    component_name,
                    source_path: self.component.clone(),
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
    component_id: rib_repl_crate::uuid::Uuid,
}

#[async_trait]
impl RibDependencyManager for WasmtimeRibDependencyManager {
    async fn get_dependencies(&self) -> rib_repl_crate::anyhow::Result<ReplComponentDependencies> {
        rib_repl_crate::anyhow::bail!(
            "load a component via `wasmtime rib <component.wasm>` (no multi-project mode yet)"
        )
    }

    async fn add_component(
        &self,
        source_path: &Path,
        component_name: String,
    ) -> rib_repl_crate::anyhow::Result<ComponentDependency> {
        let bytes = std::fs::read(source_path)
            .with_context(|| format!("failed to read {}", source_path.display()))?;
        let comp = Component::new(&self.engine, &bytes)
            .map_err(|e| rib_repl_crate::anyhow::anyhow!("{e:?}"))?;
        let exports = component_exports(&self.engine, comp.component_type())?;
        Ok(ComponentDependency::new(
            ComponentDependencyKey {
                component_name,
                component_id: self.component_id,
                component_revision: 0,
                root_package_name: None,
                root_package_version: None,
            },
            exports,
        ))
    }
}

struct WasmtimeWorkerInvoke {
    component: Component,
    instance: Instance,
    store: Arc<Mutex<Store<Host>>>,
    component_id: rib_repl_crate::uuid::Uuid,
}

#[async_trait]
impl WorkerFunctionInvoke for WasmtimeWorkerInvoke {
    async fn invoke(
        &self,
        component_id: rib_repl_crate::uuid::Uuid,
        _component_name: &str,
        _worker_name: &str,
        function_name: &str,
        args: Vec<ValueAndType>,
        return_type: Option<AnalysedType>,
    ) -> rib_repl_crate::anyhow::Result<Option<ValueAndType>> {
        if component_id != self.component_id {
            rib_repl_crate::anyhow::bail!("unexpected component id (only one component is supported)");
        }

        let parsed = ParsedFunctionName::parse(function_name).map_err(|e| {
            rib_repl_crate::anyhow::anyhow!("invalid function name `{function_name}`: {e}")
        })?;

        let path = export_path(&parsed);
        let export = resolve_export(&self.component, &path)
            .with_context(|| format!("resolve export for `{function_name}`"))?;

        let mut store = self.store.lock().await;
        let func = self
            .instance
            .get_func(&mut *store, export)
            .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("export is not a function"))?;

        let param_tys: Vec<WType> = func.ty(&*store).params().map(|(_, t)| t).collect();
        let result_tys: Vec<WType> = func.ty(&*store).results().collect();

        if param_tys.len() != args.len() {
            rib_repl_crate::anyhow::bail!(
                "expected {} arguments, got {}",
                param_tys.len(),
                args.len()
            );
        }

        let mut params = Vec::with_capacity(args.len());
        for (arg, ty) in args.iter().zip(&param_tys) {
            params.push(value_and_type_to_val(ty, arg)?);
        }

        let mut results: Vec<Val> = result_tys
            .iter()
            .map(|_| Val::Bool(false))
            .collect();

        call_func(&mut store, func, &params, &mut results).await?;

        let out = match results.len() {
            0 => None,
            1 => {
                let rt = return_type.as_ref().ok_or_else(|| {
                    rib_repl_crate::anyhow::anyhow!("missing return type for non-unit function")
                })?;
                Some(val_to_value_and_type(rt, &results[0])?)
            }
            _ => {
                let tuple_ty = return_type.ok_or_else(|| {
                    rib_repl_crate::anyhow::anyhow!("missing return type for multi-return function")
                })?;
                let vals: rib_repl_crate::anyhow::Result<Vec<ValueAndType>> = results
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let elem_ty = tuple_element_type(&tuple_ty, i)?;
                        val_to_value_and_type(&elem_ty, v)
                    })
                    .collect();
                let parts = vals?;
                let inner: Vec<Value> = parts.into_iter().map(|v| v.value).collect();
                Some(ValueAndType::new(
                    Value::Tuple(inner),
                    tuple_ty.clone(),
                ))
            }
        };

        Ok(out)
    }
}

fn tuple_element_type(tuple_ty: &AnalysedType, i: usize) -> rib_repl_crate::anyhow::Result<AnalysedType> {
    match tuple_ty {
        AnalysedType::Tuple(t) => t
            .items
            .get(i)
            .cloned()
            .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("tuple arity mismatch")),
        _ => rib_repl_crate::anyhow::bail!("expected tuple return type for multi-value return"),
    }
}

async fn call_func(
    store: &mut Store<Host>,
    func: Func,
    params: &[Val],
    results: &mut [Val],
) -> rib_repl_crate::anyhow::Result<()> {
    func.call_async(store, params, results)
        .await
        .map_err(|e| rib_repl_crate::anyhow::anyhow!("{e:?}"))
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

fn resolve_export(
    component: &Component,
    path: &[String],
) -> rib_repl_crate::anyhow::Result<ComponentExportIndex> {
    if path.is_empty() {
        rib_repl_crate::anyhow::bail!("empty export path");
    }
    let mut instance: Option<ComponentExportIndex> = None;
    for name in &path[..path.len() - 1] {
        instance = Some(
            component
                .get_export_index(instance.as_ref(), name)
                .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("missing instance export `{name}`"))?,
        );
    }
    let last = path.last().unwrap();
    component
        .get_export_index(instance.as_ref(), last)
        .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("missing function export `{last}`"))
}

fn component_exports(
    engine: &Engine,
    component: types::Component,
) -> rib_repl_crate::anyhow::Result<Vec<AnalysedExport>> {
    let funcs = collect_component_funcs(engine, component);
    let mut root_funcs = Vec::new();
    let mut by_instance: BTreeMap<String, Vec<AnalysedFunction>> = BTreeMap::new();

    for (path, cf) in funcs {
        if cf.async_() {
            continue;
        }
        let af = component_func_to_analysed(&path, &cf)?;
        if path.len() == 1 {
            root_funcs.push(AnalysedExport::Function(af));
        } else {
            let iface = path[..path.len() - 1].join("/");
            by_instance.entry(iface).or_default().push(af);
        }
    }

    let mut out: Vec<AnalysedExport> = root_funcs;
    for (name, functions) in by_instance {
        out.push(AnalysedExport::Instance(AnalysedInstance { name, functions }));
    }
    Ok(out)
}

fn collect_component_funcs(
    engine: &Engine,
    component: types::Component,
) -> Vec<(Vec<String>, types::ComponentFunc)> {
    fn walk(
        engine: &Engine,
        item: CItem,
        prefix: Vec<String>,
    ) -> Vec<(Vec<String>, CItem)> {
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

fn component_func_to_analysed(
    path: &[String],
    f: &types::ComponentFunc,
) -> rib_repl_crate::anyhow::Result<AnalysedFunction> {
    let name = path.last().expect("func path").clone();
    let parameters = f
        .params()
        .map(|(n, t)| {
            Ok(AnalysedFunctionParameter {
                name: n.to_string(),
                typ: wasm_type_to_analysed(&t)?,
            })
        })
        .collect::<rib_repl_crate::anyhow::Result<Vec<_>>>()?;

    let result = match f.results().len() {
        0 => None,
        1 => Some(AnalysedFunctionResult {
            typ: wasm_type_to_analysed(&f.results().next().unwrap())?,
        }),
        _ => {
            let items: Vec<AnalysedType> = f
                .results()
                .map(|t| wasm_type_to_analysed(&t))
                .collect::<rib_repl_crate::anyhow::Result<_>>()?;
            Some(AnalysedFunctionResult {
                typ: AnalysedType::Tuple(TypeTuple {
                    name: None,
                    owner: None,
                    items,
                }),
            })
        }
    };

    Ok(AnalysedFunction {
        name,
        parameters,
        result,
    })
}

fn wasm_type_to_analysed(ty: &WType) -> rib_repl_crate::anyhow::Result<AnalysedType> {
    Ok(match ty {
        WType::Bool => AnalysedType::Bool(TypeBool),
        WType::S8 => AnalysedType::S8(TypeS8),
        WType::U8 => AnalysedType::U8(TypeU8),
        WType::S16 => AnalysedType::S16(TypeS16),
        WType::U16 => AnalysedType::U16(TypeU16),
        WType::S32 => AnalysedType::S32(TypeS32),
        WType::U32 => AnalysedType::U32(TypeU32),
        WType::S64 => AnalysedType::S64(TypeS64),
        WType::U64 => AnalysedType::U64(TypeU64),
        WType::Float32 => AnalysedType::F32(TypeF32),
        WType::Float64 => AnalysedType::F64(TypeF64),
        WType::Char => AnalysedType::Chr(TypeChr),
        WType::String => AnalysedType::Str(TypeStr),
        WType::List(l) => AnalysedType::List(TypeList {
            name: None,
            owner: None,
            inner: Box::new(wasm_type_to_analysed(&l.ty())?),
        }),
        WType::Record(r) => {
            let fields = r
                .fields()
                .map(|fld| {
                    Ok(NameTypePair {
                        name: fld.name.to_string(),
                        typ: wasm_type_to_analysed(&fld.ty)?,
                    })
                })
                .collect::<rib_repl_crate::anyhow::Result<Vec<_>>>()?;
            AnalysedType::Record(TypeRecord {
                name: None,
                owner: None,
                fields,
            })
        }
        WType::Tuple(t) => {
            let items = t
                .types()
                .map(|ty| wasm_type_to_analysed(&ty))
                .collect::<rib_repl_crate::anyhow::Result<Vec<_>>>()?;
            AnalysedType::Tuple(TypeTuple {
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
                        typ: c.ty.map(|t| wasm_type_to_analysed(&t)).transpose()?,
                    })
                })
                .collect::<rib_repl_crate::anyhow::Result<Vec<_>>>()?;
            AnalysedType::Variant(TypeVariant {
                name: None,
                owner: None,
                cases,
            })
        }
        WType::Enum(e) => AnalysedType::Enum(TypeEnum {
            name: None,
            owner: None,
            cases: e.names().map(str::to_string).collect(),
        }),
        WType::Option(o) => AnalysedType::Option(TypeOption {
            name: None,
            owner: None,
            inner: Box::new(wasm_type_to_analysed(&o.ty())?),
        }),
        WType::Result(r) => AnalysedType::Result(TypeResult {
            name: None,
            owner: None,
            ok: r
                .ok()
                .map(|t| wasm_type_to_analysed(&t))
                .transpose()?
                .map(Box::new),
            err: r
                .err()
                .map(|t| wasm_type_to_analysed(&t))
                .transpose()?
                .map(Box::new),
        }),
        WType::Flags(fl) => AnalysedType::Flags(TypeFlags {
            name: None,
            owner: None,
            names: fl.names().map(str::to_string).collect(),
        }),
        WType::Own(_) => AnalysedType::Handle(TypeHandle {
            name: None,
            owner: None,
            resource_id: AnalysedResourceId(0),
            mode: AnalysedResourceMode::Owned,
        }),
        WType::Borrow(_) => AnalysedType::Handle(TypeHandle {
            name: None,
            owner: None,
            resource_id: AnalysedResourceId(0),
            mode: AnalysedResourceMode::Borrowed,
        }),
        WType::Map(_) => rib_repl_crate::anyhow::bail!("Rib metadata does not support WIT `map` yet"),
        WType::Future(_) | WType::Stream(_) | WType::ErrorContext => {
            rib_repl_crate::anyhow::bail!("async component types are not supported in Rib metadata yet")
        }
    })
}

fn value_and_type_to_val(expected: &WType, v: &ValueAndType) -> rib_repl_crate::anyhow::Result<Val> {
    use rib::analysis::AnalysedType as AT;
    match (expected, &v.value) {
        (WType::Bool, Value::Bool(b)) => Ok(Val::Bool(*b)),
        (WType::S8, Value::S8(x)) => Ok(Val::S8(*x)),
        (WType::U8, Value::U8(x)) => Ok(Val::U8(*x)),
        (WType::S16, Value::S16(x)) => Ok(Val::S16(*x)),
        (WType::U16, Value::U16(x)) => Ok(Val::U16(*x)),
        (WType::S32, Value::S32(x)) => Ok(Val::S32(*x)),
        (WType::U32, Value::U32(x)) => Ok(Val::U32(*x)),
        (WType::S64, Value::S64(x)) => Ok(Val::S64(*x)),
        (WType::U64, Value::U64(x)) => Ok(Val::U64(*x)),
        (WType::Float32, Value::F32(x)) => Ok(Val::Float32(*x)),
        (WType::Float64, Value::F64(x)) => Ok(Val::Float64(*x)),
        (WType::Char, Value::Char(c)) => Ok(Val::Char(*c)),
        (WType::String, Value::String(s)) => Ok(Val::String(s.clone())),
        (WType::List(_), Value::List(items)) => {
            let WType::List(l) = expected else { unreachable!() };
            let elem = l.ty();
            let inner = if let AT::List(lt) = &v.typ {
                items
                    .iter()
                    .map(|x| {
                        value_and_type_to_val(
                            &elem,
                            &ValueAndType::new(x.clone(), (*lt.inner).clone()),
                        )
                    })
                    .collect::<rib_repl_crate::anyhow::Result<_>>()?
            } else {
                rib_repl_crate::anyhow::bail!("list type mismatch");
            };
            Ok(Val::List(inner))
        }
        (WType::Record(_), Value::Record(items)) => {
            let WType::Record(r) = expected else { unreachable!() };
            let AT::Record(rec_ty) = &v.typ else {
                rib_repl_crate::anyhow::bail!("record type mismatch");
            };
            if rec_ty.fields.len() != items.len() {
                rib_repl_crate::anyhow::bail!("record field count mismatch");
            }
            let pairs: rib_repl_crate::anyhow::Result<Vec<(String, Val)>> = r
                .fields()
                .zip(items.iter())
                .zip(rec_ty.fields.iter())
                .map(|((fld, val), spec)| {
                    Ok((
                        fld.name.to_string(),
                        value_and_type_to_val(
                            &fld.ty,
                            &ValueAndType::new(val.clone(), spec.typ.clone()),
                        )?,
                    ))
                })
                .collect();
            Ok(Val::Record(pairs?))
        }
        (WType::Tuple(_), Value::Tuple(items)) => {
            let WType::Tuple(t) = expected else { unreachable!() };
            let AT::Tuple(tup_ty) = &v.typ else {
                rib_repl_crate::anyhow::bail!("tuple type mismatch");
            };
            let inner = t
                .types()
                .zip(items.iter())
                .zip(tup_ty.items.iter())
                .map(|((wt, val), at)| value_and_type_to_val(&wt, &ValueAndType::new(val.clone(), at.clone())))
                .collect::<rib_repl_crate::anyhow::Result<Vec<_>>>()?;
            Ok(Val::Tuple(inner))
        }
        (WType::Variant(_), Value::Variant { case_idx, case_value }) => {
            let WType::Variant(wasm_var) = expected else { unreachable!() };
            let cases: Vec<_> = wasm_var.cases().collect();
            let case = cases
                .get(*case_idx as usize)
                .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("invalid variant case index"))?;
            let payload = match (&case.ty, case_value) {
                (None, None) => None,
                (Some(wt), Some(boxed)) => {
                    let AT::Variant(var_ty) = &v.typ else {
                        rib_repl_crate::anyhow::bail!("variant type mismatch");
                    };
                    let case_ty = var_ty
                        .cases
                        .get(*case_idx as usize)
                        .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("bad variant case"))?;
                    let inner_ty = case_ty
                        .typ
                        .as_ref()
                        .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("expected payload type"))?;
                    Some(Box::new(value_and_type_to_val(
                        wt,
                        &ValueAndType::new((**boxed).clone(), inner_ty.clone()),
                    )?))
                }
                _ => rib_repl_crate::anyhow::bail!("variant payload mismatch"),
            };
            Ok(Val::Variant(case.name.to_string(), payload))
        }
        (WType::Enum(_), Value::Enum(idx)) => {
            let WType::Enum(e) = expected else { unreachable!() };
            let name = e
                .names()
                .nth(*idx as usize)
                .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("invalid enum discriminant"))?;
            Ok(Val::Enum(name.to_string()))
        }
        (WType::Option(_), Value::Option(inner)) => {
            let WType::Option(o) = expected else { unreachable!() };
            let AT::Option(opt_ty) = &v.typ else {
                rib_repl_crate::anyhow::bail!("option type mismatch");
            };
            let mapped = match inner {
                None => None,
                Some(b) => Some(Box::new(value_and_type_to_val(
                    &o.ty(),
                    &ValueAndType::new((**b).clone(), (*opt_ty.inner).clone()),
                )?)),
            };
            Ok(Val::Option(mapped))
        }
        (WType::Result(_), Value::Result(inner)) => {
            let WType::Result(r) = expected else { unreachable!() };
            let AT::Result(res_ty) = &v.typ else {
                rib_repl_crate::anyhow::bail!("result type mismatch");
            };
            let mapped = match inner {
                Ok(v) => Ok(match v {
                    None => None,
                    Some(b) => {
                        let wt = r
                            .ok()
                            .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("result ok type missing"))?;
                        let at = res_ty
                            .ok
                            .as_deref()
                            .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("result ok type missing"))?;
                        Some(Box::new(value_and_type_to_val(
                            &wt,
                            &ValueAndType::new((**b).clone(), at.clone()),
                        )?))
                    }
                }),
                Err(v) => Err(match v {
                    None => None,
                    Some(b) => {
                        let wt = r
                            .err()
                            .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("result err type missing"))?;
                        let at = res_ty
                            .err
                            .as_deref()
                            .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("result err type missing"))?;
                        Some(Box::new(value_and_type_to_val(
                            &wt,
                            &ValueAndType::new((**b).clone(), at.clone()),
                        )?))
                    }
                }),
            };
            Ok(Val::Result(mapped))
        }
        (WType::Flags(_), Value::Flags(bits)) => {
            let WType::Flags(f) = expected else { unreachable!() };
            let names: Vec<String> = f
                .names()
                .enumerate()
                .filter_map(|(i, n)| bits.get(i).copied().unwrap_or(false).then_some(n.to_string()))
                .collect();
            Ok(Val::Flags(names))
        }
        _ => rib_repl_crate::anyhow::bail!(
            "cannot convert Rib value {:?} to Wasmtime value for type {:?}",
            v.value,
            expected
        ),
    }
}

fn val_to_value_and_type(ty: &AnalysedType, v: &Val) -> rib_repl_crate::anyhow::Result<ValueAndType> {
    use AnalysedType as AT;
    Ok(match (ty, v) {
        (AT::Bool(_), Val::Bool(b)) => ValueAndType::new(Value::Bool(*b), ty.clone()),
        (AT::S8(_), Val::S8(x)) => ValueAndType::new(Value::S8(*x), ty.clone()),
        (AT::U8(_), Val::U8(x)) => ValueAndType::new(Value::U8(*x), ty.clone()),
        (AT::S16(_), Val::S16(x)) => ValueAndType::new(Value::S16(*x), ty.clone()),
        (AT::U16(_), Val::U16(x)) => ValueAndType::new(Value::U16(*x), ty.clone()),
        (AT::S32(_), Val::S32(x)) => ValueAndType::new(Value::S32(*x), ty.clone()),
        (AT::U32(_), Val::U32(x)) => ValueAndType::new(Value::U32(*x), ty.clone()),
        (AT::S64(_), Val::S64(x)) => ValueAndType::new(Value::S64(*x), ty.clone()),
        (AT::U64(_), Val::U64(x)) => ValueAndType::new(Value::U64(*x), ty.clone()),
        (AT::F32(_), Val::Float32(x)) => ValueAndType::new(Value::F32(*x), ty.clone()),
        (AT::F64(_), Val::Float64(x)) => ValueAndType::new(Value::F64(*x), ty.clone()),
        (AT::Chr(_), Val::Char(c)) => ValueAndType::new(Value::Char(*c), ty.clone()),
        (AT::Str(_), Val::String(s)) => ValueAndType::new(Value::String(s.clone()), ty.clone()),
        (AT::List(lt), Val::List(items)) => {
            let inner: rib_repl_crate::anyhow::Result<Vec<Value>> = items
                .iter()
                .map(|x| Ok(val_to_value_and_type(&lt.inner, x)?.value))
                .collect();
            ValueAndType::new(Value::List(inner?), ty.clone())
        }
        (AT::Record(rt), Val::Record(pairs)) => {
            if rt.fields.len() != pairs.len() {
                rib_repl_crate::anyhow::bail!("record field mismatch");
            }
            let vals: rib_repl_crate::anyhow::Result<Vec<Value>> = rt
                .fields
                .iter()
                .zip(pairs.iter())
                .map(|(f, (n, val))| {
                    if f.name != *n {
                        rib_repl_crate::anyhow::bail!("record field name mismatch");
                    }
                    Ok(val_to_value_and_type(&f.typ, val)?.value)
                })
                .collect();
            ValueAndType::new(Value::Record(vals?), ty.clone())
        }
        (AT::Tuple(tt), Val::Tuple(items)) => {
            if tt.items.len() != items.len() {
                rib_repl_crate::anyhow::bail!("tuple arity mismatch");
            }
            let vals: rib_repl_crate::anyhow::Result<Vec<Value>> = tt
                .items
                .iter()
                .zip(items.iter())
                .map(|(t, v)| Ok(val_to_value_and_type(t, v)?.value))
                .collect();
            ValueAndType::new(Value::Tuple(vals?), ty.clone())
        }
        (AT::Variant(vt), Val::Variant(name, payload)) => {
            let (idx, case_ty) = vt
                .cases
                .iter()
                .enumerate()
                .find(|(_, c)| c.name == *name)
                .map(|(i, c)| (i as u32, &c.typ))
                .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("unknown variant case `{name}`"))?;
            let case_value = match (case_ty, payload) {
                (None, None) => None,
                (Some(inner), Some(p)) => Some(Box::new(val_to_value_and_type(inner, p)?.value)),
                _ => rib_repl_crate::anyhow::bail!("variant payload mismatch"),
            };
            ValueAndType::new(
                Value::Variant {
                    case_idx: idx,
                    case_value,
                },
                ty.clone(),
            )
        }
        (AT::Enum(et), Val::Enum(name)) => {
            let idx = et
                .cases
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("unknown enum case `{name}`"))? as u32;
            ValueAndType::new(Value::Enum(idx), ty.clone())
        }
        (AT::Option(ot), Val::Option(inner)) => {
            let v = match inner {
                None => Value::Option(None),
                Some(b) => Value::Option(Some(Box::new(
                    val_to_value_and_type(&ot.inner, b)?.value,
                ))),
            };
            ValueAndType::new(v, ty.clone())
        }
        (AT::Result(rt), Val::Result(inner)) => {
            let v = match inner {
                Ok(x) => Value::Result(Ok(match x {
                    None => None,
                    Some(b) => Some(Box::new(
                        val_to_value_and_type(
                            rt.ok.as_deref().ok_or_else(|| rib_repl_crate::anyhow::anyhow!("ok type"))?,
                            b,
                        )?
                        .value,
                    )),
                })),
                Err(x) => Value::Result(Err(match x {
                    None => None,
                    Some(b) => Some(Box::new(
                        val_to_value_and_type(
                            rt.err.as_deref().ok_or_else(|| rib_repl_crate::anyhow::anyhow!("err type"))?,
                            b,
                        )?
                        .value,
                    )),
                })),
            };
            ValueAndType::new(v, ty.clone())
        }
        (AT::Flags(ft), Val::Flags(names)) => {
            let mut bits = vec![false; ft.names.len()];
            for n in names {
                let i = ft
                    .names
                    .iter()
                    .position(|x| x == n)
                    .ok_or_else(|| rib_repl_crate::anyhow::anyhow!("unknown flag `{n}`"))?;
                bits[i] = true;
            }
            ValueAndType::new(Value::Flags(bits), ty.clone())
        }
        _ => rib_repl_crate::anyhow::bail!("cannot lift Wasmtime value to Rib for type {ty:?}"),
    })
}
