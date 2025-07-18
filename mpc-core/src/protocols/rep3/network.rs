use mpc_net::mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker};

pub use mpc_core::protocols::rep3::network::*;
use mpc_net::rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator};

pub trait Rep3NetworkWorker: Rep3Network + MpcStarNetWorker + 'static {}
pub trait Rep3NetworkCoordinator: MpcStarNetCoordinator + 'static {}

impl Rep3NetworkWorker for Rep3QuicMpcNetWorker {

}

impl Rep3NetworkCoordinator for Rep3QuicNetCoordinator {

}
