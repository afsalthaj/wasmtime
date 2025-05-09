use crate::p2::{bindings::cli::exit, WasiImpl, WasiView};
use crate::I32Exit;

impl<T> exit::Host for WasiImpl<T>
where
    T: WasiView,
{
    fn exit(&mut self, status: Result<(), ()>) -> anyhow::Result<()> {
        let status = match status {
            Ok(()) => 0,
            Err(()) => 1,
        };
        Err(anyhow::anyhow!(I32Exit(status)))
    }

    fn exit_with_code(&mut self, status_code: u8) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(I32Exit(status_code.into())))
    }
}
