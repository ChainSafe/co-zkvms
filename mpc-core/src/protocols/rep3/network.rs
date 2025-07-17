use mpc_net::mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker};

pub use mpc_core::protocols::rep3::network::*;

pub trait Rep3NetworkWorker: Rep3Network + MpcStarNetWorker + 'static {}
pub trait Rep3NetworkCoordinator: MpcStarNetCoordinator + 'static {}
