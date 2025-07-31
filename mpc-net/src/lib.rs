pub mod mpc_star;
pub mod rep3;

pub(crate) use color_eyre::eyre::Result;
use std::{sync::Mutex, time::Duration};
use tokio::runtime::Runtime;

pub mod channel;
pub mod codecs;
pub mod config;

pub use rep3::quic::MpcNetworkHandlerWorker as MpcNetworkHandler;

pub(crate) const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// A warapper for a runtime and a network handler for MPC protocols.
/// Ensures a gracefull shutdown on drop
#[derive(Debug)]
pub struct MpcNetworkHandlerWrapper<H: MpcNetworkHandlerShutdown = MpcNetworkHandler> {
    /// The runtime used by the network handler
    pub runtime: Runtime,
    /// The wrapped network handler
    pub inner: H,
}

impl<H: MpcNetworkHandlerShutdown> MpcNetworkHandlerWrapper<H> {
    /// Create a new wrapper  
    pub fn new(runtime: Runtime, inner: H) -> Self {
        Self { runtime, inner }
    }
}

impl<H: MpcNetworkHandlerShutdown> Drop for MpcNetworkHandlerWrapper<H> {
    fn drop(&mut self) {
        // ignore errors in drop
        let _ = self.runtime.block_on(self.inner.shutdown());
    }
}

trait MpcNetworkHandlerShutdown: Send + Sync {
    async fn shutdown(&self) -> std::io::Result<()>;
}

/// A warapper for a runtime and a network handler for MPC protocols.
/// Ensures a gracefull shutdown on drop
#[derive(Debug)]
pub struct MpcNetworkHandlerWrapperMut<H: MpcNetworkHandlerShutdown = MpcNetworkHandler> {
    /// The runtime used by the network handler
    pub runtime: Runtime,
    /// The wrapped network handler
    pub inner: Mutex<H>,
}

impl<H: MpcNetworkHandlerShutdown> MpcNetworkHandlerWrapperMut<H> {
    /// Create a new wrapper  
    pub fn new(runtime: Runtime, inner: H) -> Self {
        Self { runtime, inner: Mutex::new(inner) }
    }
}

impl<H: MpcNetworkHandlerShutdown> Drop for MpcNetworkHandlerWrapperMut<H> {
    fn drop(&mut self) {
        // ignore errors in drop
        let _ = self.runtime.block_on(self.inner.lock().unwrap().shutdown());
    }
}
