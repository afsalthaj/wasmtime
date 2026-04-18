//! Rib language REPL for WebAssembly components.
//!
//! Enabled with the `rib` Cargo feature: `cargo build -p wasmtime-cli --features rib`.

#![cfg(feature = "rib")]

use super::run::{CliLinker, Host, Preloads, RunCommand};
use crate::common::{RunCommon, RunTarget};
use async_trait::async_trait;
use clap::Parser;
use rib::wit_type::{
    AnalysedResourceId, AnalysedResourceMode, NameOptionTypePair, NameTypePair, TypeBool, TypeChr,
    TypeEnum, TypeF32, TypeF64, TypeFlags, TypeHandle, TypeList, TypeOption, TypeRecord,
    TypeResult, TypeS8, TypeS16, TypeS32, TypeS64, TypeStr, TypeTuple, TypeU8, TypeU16, TypeU32,
    TypeU64, TypeVariant, WitExport, WitFunction, WitFunctionParameter, WitFunctionResult,
    WitInterface, WitType,
};
use rib::{
    ComponentDependency, ComponentDependencyKey, ParsedFunctionName, ParsedFunctionSite, Value,
    ValueAndType,
};
use rib_repl::anyhow::{anyhow, bail, Result};
use rib_repl::anyhow::Context as _;
use rib_repl::rib;
use rib_repl::uuid::Uuid;
use rib_repl::{
    ComponentFunctionInvoke, ComponentSource, ReplComponentBundle, RibDependencyManager, RibRepl,
    RibReplConfig,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use wasmtime::component::types::{self, ComponentItem as CItem, Type as WType};
use wasmtime::component::{Component, ComponentExportIndex, Func, Instance, Val};
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
            let (mut store, linker) = run_cmd.new_store_and_linker(&engine, &main)?;
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

            let instance = cli_linker
                .instantiate_async(&mut store, &component)
                .await
                .map_err(|e| wasmtime::Error::msg(format!("{e:?}")))?;

            let component_id = Uuid::new_v4();

            let store = Arc::new(Mutex::new(store));
            let dep_manager = Arc::new(WasmtimeRibDependencyManager {
                engine: engine.clone(),
                component_id,
            });
            let invoke = Arc::new(WasmtimeComponentFunctionInvoke {
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
bail!(
            "load a component via `wasmtime repl <component.wasm>` (no multi-project mode yet)"
        )
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
        let comp = Component::new(&self.engine, &bytes)
            .map_err(|e| anyhow!("{e:?}"))?;
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

struct WasmtimeComponentFunctionInvoke {
    component: Component,
    instance: Instance,
    store: Arc<Mutex<Store<Host>>>,
    component_id: Uuid,
}

#[async_trait]
impl ComponentFunctionInvoke for WasmtimeComponentFunctionInvoke {
    async fn invoke(
        &self,
        component_id: Uuid,
        _component_name: &str,
        _worker_name: &str,
        function_name: &str,
        args: Vec<ValueAndType>,
        return_type: Option<WitType>,
    ) -> Result<Option<ValueAndType>> {
        if component_id != self.component_id {
bail!(
                "unexpected component id (only one component is supported)"
            );
        }

        let parsed = ParsedFunctionName::parse(function_name).map_err(|e| {
anyhow!("invalid function name `{function_name}`: {e}")
        })?;

        let path = export_path(&parsed);
        let export = resolve_export(&self.component, &path)
            .with_context(|| format!("resolve export for `{function_name}`"))?;

        let mut store = self.store.lock().await;
        let func = self
            .instance
            .get_func(&mut *store, export)
            .ok_or_else(|| anyhow!("export is not a function"))?;

        let param_tys: Vec<WType> = func.ty(&*store).params().map(|(_, t)| t).collect();
        let result_tys: Vec<WType> = func.ty(&*store).results().collect();

        if param_tys.len() != args.len() {
bail!(
                "expected {} arguments, got {}",
                param_tys.len(),
                args.len()
            );
        }

        let mut params = Vec::with_capacity(args.len());
        for (arg, ty) in args.iter().zip(&param_tys) {
            params.push(value_and_type_to_val(ty, arg)?);
        }

        let mut results: Vec<Val> = result_tys.iter().map(|_| Val::Bool(false)).collect();

        call_func(&mut store, func, &params, &mut results).await?;

        let out = match results.len() {
            0 => None,
            1 => {
                let rt = return_type.as_ref().ok_or_else(|| {
anyhow!("missing return type for non-unit function")
                })?;
                Some(val_to_value_and_type(rt, &results[0])?)
            }
            _ => {
                let tuple_ty = return_type.ok_or_else(|| {
anyhow!("missing return type for multi-return function")
                })?;
                let vals: Result<Vec<ValueAndType>> = results
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let elem_ty = tuple_element_type(&tuple_ty, i)?;
                        val_to_value_and_type(&elem_ty, v)
                    })
                    .collect();
                let parts = vals?;
                let inner: Vec<Value> = parts.into_iter().map(|v| v.value).collect();
                Some(ValueAndType::new(Value::Tuple(inner), tuple_ty.clone()))
            }
        };

        Ok(out)
    }
}

fn tuple_element_type(tuple_ty: &WitType, i: usize) -> Result<WitType> {
    match tuple_ty {
        WitType::Tuple(t) => t
            .items
            .get(i)
            .cloned()
            .ok_or_else(|| anyhow!("tuple arity mismatch")),
        _ => bail!("expected tuple return type for multi-value return"),
    }
}

async fn call_func(
    store: &mut Store<Host>,
    func: Func,
    params: &[Val],
    results: &mut [Val],
) -> Result<()> {
    func.call_async(store, params, results)
        .await
        .map_err(|e| anyhow!("{e:?}"))
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
) -> Result<ComponentExportIndex> {
    if path.is_empty() {
bail!("empty export path");
    }
    let mut instance: Option<ComponentExportIndex> = None;
    for name in &path[..path.len() - 1] {
        instance = Some(
            component
                .get_export_index(instance.as_ref(), name)
                .ok_or_else(|| {
anyhow!("missing instance export `{name}`")
                })?,
        );
    }
    let last = path.last().unwrap();
    component
        .get_export_index(instance.as_ref(), last)
        .ok_or_else(|| anyhow!("missing function export `{last}`"))
}

fn component_exports(
    engine: &Engine,
    component: types::Component,
) -> Result<Vec<WitExport>> {
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

fn component_func_to_wit(
    path: &[String],
    f: &types::ComponentFunc,
) -> Result<WitFunction> {
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
bail!(
                "async component types are not supported in Rib metadata yet"
            )
        }
    })
}

fn value_and_type_to_val(
    expected: &WType,
    v: &ValueAndType,
) -> Result<Val> {
    use rib::wit_type::WitType as WT;
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
            let WType::List(l) = expected else {
                unreachable!()
            };
            let elem = l.ty();
            let inner = if let WT::List(lt) = &v.typ {
                items
                    .iter()
                    .map(|x| {
                        value_and_type_to_val(
                            &elem,
                            &ValueAndType::new(x.clone(), (*lt.inner).clone()),
                        )
                    })
                    .collect::<Result<_>>()?
            } else {
bail!("list type mismatch");
            };
            Ok(Val::List(inner))
        }
        (WType::Record(_), Value::Record(items)) => {
            let WType::Record(r) = expected else {
                unreachable!()
            };
            let WT::Record(rec_ty) = &v.typ else {
bail!("record type mismatch");
            };
            if rec_ty.fields.len() != items.len() {
bail!("record field count mismatch");
            }
            let pairs: Result<Vec<(String, Val)>> = r
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
            let WType::Tuple(t) = expected else {
                unreachable!()
            };
            let WT::Tuple(tup_ty) = &v.typ else {
bail!("tuple type mismatch");
            };
            let inner = t
                .types()
                .zip(items.iter())
                .zip(tup_ty.items.iter())
                .map(|((wt, val), at)| {
                    value_and_type_to_val(&wt, &ValueAndType::new(val.clone(), at.clone()))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Val::Tuple(inner))
        }
        (
            WType::Variant(_),
            Value::Variant {
                case_idx,
                case_value,
            },
        ) => {
            let WType::Variant(wasm_var) = expected else {
                unreachable!()
            };
            let cases: Vec<_> = wasm_var.cases().collect();
            let case = cases
                .get(*case_idx as usize)
                .ok_or_else(|| anyhow!("invalid variant case index"))?;
            let payload = match (&case.ty, case_value) {
                (None, None) => None,
                (Some(wt), Some(boxed)) => {
                    let WT::Variant(var_ty) = &v.typ else {
bail!("variant type mismatch");
                    };
                    let case_ty = var_ty
                        .cases
                        .get(*case_idx as usize)
                        .ok_or_else(|| anyhow!("bad variant case"))?;
                    let inner_ty = case_ty
                        .typ
                        .as_ref()
                        .ok_or_else(|| anyhow!("expected payload type"))?;
                    Some(Box::new(value_and_type_to_val(
                        wt,
                        &ValueAndType::new((**boxed).clone(), inner_ty.clone()),
                    )?))
                }
                _ => bail!("variant payload mismatch"),
            };
            Ok(Val::Variant(case.name.to_string(), payload))
        }
        (WType::Enum(_), Value::Enum(idx)) => {
            let WType::Enum(e) = expected else {
                unreachable!()
            };
            let name = e
                .names()
                .nth(*idx as usize)
                .ok_or_else(|| anyhow!("invalid enum discriminant"))?;
            Ok(Val::Enum(name.to_string()))
        }
        (WType::Option(_), Value::Option(inner)) => {
            let WType::Option(o) = expected else {
                unreachable!()
            };
            let WT::Option(opt_ty) = &v.typ else {
bail!("option type mismatch");
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
            let WType::Result(r) = expected else {
                unreachable!()
            };
            let WT::Result(res_ty) = &v.typ else {
bail!("result type mismatch");
            };
            let mapped = match inner {
                Ok(v) => Ok(match v {
                    None => None,
                    Some(b) => {
                        let wt = r.ok().ok_or_else(|| {
anyhow!("result ok type missing")
                        })?;
                        let at = res_ty.ok.as_deref().ok_or_else(|| {
anyhow!("result ok type missing")
                        })?;
                        Some(Box::new(value_and_type_to_val(
                            &wt,
                            &ValueAndType::new((**b).clone(), at.clone()),
                        )?))
                    }
                }),
                Err(v) => Err(match v {
                    None => None,
                    Some(b) => {
                        let wt = r.err().ok_or_else(|| {
anyhow!("result err type missing")
                        })?;
                        let at = res_ty.err.as_deref().ok_or_else(|| {
anyhow!("result err type missing")
                        })?;
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
            let WType::Flags(f) = expected else {
                unreachable!()
            };
            let names: Vec<String> = f
                .names()
                .enumerate()
                .filter_map(|(i, n)| {
                    bits.get(i)
                        .copied()
                        .unwrap_or(false)
                        .then_some(n.to_string())
                })
                .collect();
            Ok(Val::Flags(names))
        }
        _ => bail!(
            "cannot convert Rib value {:?} to Wasmtime value for type {:?}",
            v.value,
            expected
        ),
    }
}

fn val_to_value_and_type(ty: &WitType, v: &Val) -> Result<ValueAndType> {
    use WitType as WT;
    Ok(match (ty, v) {
        (WT::Bool(_), Val::Bool(b)) => ValueAndType::new(Value::Bool(*b), ty.clone()),
        (WT::S8(_), Val::S8(x)) => ValueAndType::new(Value::S8(*x), ty.clone()),
        (WT::U8(_), Val::U8(x)) => ValueAndType::new(Value::U8(*x), ty.clone()),
        (WT::S16(_), Val::S16(x)) => ValueAndType::new(Value::S16(*x), ty.clone()),
        (WT::U16(_), Val::U16(x)) => ValueAndType::new(Value::U16(*x), ty.clone()),
        (WT::S32(_), Val::S32(x)) => ValueAndType::new(Value::S32(*x), ty.clone()),
        (WT::U32(_), Val::U32(x)) => ValueAndType::new(Value::U32(*x), ty.clone()),
        (WT::S64(_), Val::S64(x)) => ValueAndType::new(Value::S64(*x), ty.clone()),
        (WT::U64(_), Val::U64(x)) => ValueAndType::new(Value::U64(*x), ty.clone()),
        (WT::F32(_), Val::Float32(x)) => ValueAndType::new(Value::F32(*x), ty.clone()),
        (WT::F64(_), Val::Float64(x)) => ValueAndType::new(Value::F64(*x), ty.clone()),
        (WT::Chr(_), Val::Char(c)) => ValueAndType::new(Value::Char(*c), ty.clone()),
        (WT::Str(_), Val::String(s)) => ValueAndType::new(Value::String(s.clone()), ty.clone()),
        (WT::List(lt), Val::List(items)) => {
            let inner: Result<Vec<Value>> = items
                .iter()
                .map(|x| Ok(val_to_value_and_type(&lt.inner, x)?.value))
                .collect();
            ValueAndType::new(Value::List(inner?), ty.clone())
        }
        (WT::Record(rt), Val::Record(pairs)) => {
            if rt.fields.len() != pairs.len() {
bail!("record field mismatch");
            }
            let vals: Result<Vec<Value>> = rt
                .fields
                .iter()
                .zip(pairs.iter())
                .map(|(f, (n, val))| {
                    if f.name != *n {
bail!("record field name mismatch");
                    }
                    Ok(val_to_value_and_type(&f.typ, val)?.value)
                })
                .collect();
            ValueAndType::new(Value::Record(vals?), ty.clone())
        }
        (WT::Tuple(tt), Val::Tuple(items)) => {
            if tt.items.len() != items.len() {
bail!("tuple arity mismatch");
            }
            let vals: Result<Vec<Value>> = tt
                .items
                .iter()
                .zip(items.iter())
                .map(|(t, v)| Ok(val_to_value_and_type(t, v)?.value))
                .collect();
            ValueAndType::new(Value::Tuple(vals?), ty.clone())
        }
        (WT::Variant(vt), Val::Variant(name, payload)) => {
            let (idx, case_ty) = vt
                .cases
                .iter()
                .enumerate()
                .find(|(_, c)| c.name == *name)
                .map(|(i, c)| (i as u32, &c.typ))
                .ok_or_else(|| anyhow!("unknown variant case `{name}`"))?;
            let case_value = match (case_ty, payload) {
                (None, None) => None,
                (Some(inner), Some(p)) => Some(Box::new(val_to_value_and_type(inner, p)?.value)),
                _ => bail!("variant payload mismatch"),
            };
            ValueAndType::new(
                Value::Variant {
                    case_idx: idx,
                    case_value,
                },
                ty.clone(),
            )
        }
        (WT::Enum(et), Val::Enum(name)) => {
            let idx = et
                .cases
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| anyhow!("unknown enum case `{name}`"))?
                as u32;
            ValueAndType::new(Value::Enum(idx), ty.clone())
        }
        (WT::Option(ot), Val::Option(inner)) => {
            let v = match inner {
                None => Value::Option(None),
                Some(b) => {
                    Value::Option(Some(Box::new(val_to_value_and_type(&ot.inner, b)?.value)))
                }
            };
            ValueAndType::new(v, ty.clone())
        }
        (WT::Result(rt), Val::Result(inner)) => {
            let v = match inner {
                Ok(x) => Value::Result(Ok(match x {
                    None => None,
                    Some(b) => Some(Box::new(
                        val_to_value_and_type(
                            rt.ok
                                .as_deref()
                                .ok_or_else(|| anyhow!("ok type"))?,
                            b,
                        )?
                        .value,
                    )),
                })),
                Err(x) => Value::Result(Err(match x {
                    None => None,
                    Some(b) => Some(Box::new(
                        val_to_value_and_type(
                            rt.err
                                .as_deref()
                                .ok_or_else(|| anyhow!("err type"))?,
                            b,
                        )?
                        .value,
                    )),
                })),
            };
            ValueAndType::new(v, ty.clone())
        }
        (WT::Flags(ft), Val::Flags(names)) => {
            let mut bits = vec![false; ft.names.len()];
            for n in names {
                let i = ft
                    .names
                    .iter()
                    .position(|x| x == n)
                    .ok_or_else(|| anyhow!("unknown flag `{n}`"))?;
                bits[i] = true;
            }
            ValueAndType::new(Value::Flags(bits), ty.clone())
        }
        _ => bail!("cannot lift Wasmtime value to Rib for type {ty:?}"),
    })
}
