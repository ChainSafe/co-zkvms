//! Data structures and helpers for the network configuration.
use color_eyre::eyre;
use quinn::rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fmt::Formatter,
    net::{IpAddr, SocketAddr, ToSocketAddrs},
    num::ParseIntError,
    path::PathBuf,
    str::FromStr,
    time::Duration,
};

use crate::rep3::PartyWorkerID;

/// A network address wrapper.
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Address {
    /// The hostname of the address, will be DNS resolved. This hostname is also checked to be contained in the certificate for the party.
    pub hostname: String,
    /// The port of the address.
    pub port: u16,
}

impl Address {
    /// Construct a new [`Address`] type.
    pub fn new(hostname: String, port: u16) -> Self {
        Self { hostname, port }
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.hostname, self.port)
    }
}

/// An error for parsing [`Address`]es.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseAddressError {
    /// Must be hostname:port
    InvalidFormat,
    /// Invalid port
    InvalidPort(ParseIntError),
}

impl std::error::Error for ParseAddressError {}

impl std::fmt::Display for ParseAddressError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseAddressError::InvalidFormat => {
                write!(f, "invalid format, expected hostname:port")
            }
            ParseAddressError::InvalidPort(e) => write!(f, "cannot parse port: {e}"),
        }
    }
}

impl FromStr for Address {
    type Err = ParseAddressError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err(ParseAddressError::InvalidFormat);
        }
        let hostname = parts[0].to_string();
        let port = parts[1].parse().map_err(ParseAddressError::InvalidPort)?;
        Ok(Address { hostname, port })
    }
}

impl ToSocketAddrs for Address {
    type Iter = std::vec::IntoIter<SocketAddr>;
    fn to_socket_addrs(&self) -> std::io::Result<Self::Iter> {
        let mut addrs: Vec<SocketAddr> = format!("{}:{}", self.hostname, self.port)
            .to_socket_addrs()?
            .collect();
        // Sort IPv4 addresses first so that connections to servers bound on
        // 0.0.0.0 succeed even when the OS resolves "localhost" to ::1 first.
        addrs.sort_by_key(|a| matches!(a, SocketAddr::V6(_)));
        Ok(addrs.into_iter())
    }
}

impl Serialize for Address {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("{}:{}", self.hostname, self.port))
    }
}

impl<'de> Deserialize<'de> for Address {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Address::from_str(&s).map_err(serde::de::Error::custom)
    }
}

/// A party in the network config file.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct NetworkWorkerConfig {
    /// The id of the party, 0-based indexing.
    pub id: usize,
    /// The index of the worker in the party.
    #[serde(default)]
    pub worker: usize,
    /// The DNS name of the party.
    pub dns_name: Address,
    /// The path to the public certificate of the party.
    pub cert_path: PathBuf,
}

/// Protocol used for the worker↔coordinator connection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "lowercase")]
pub enum CoordinatorProtocol {
    /// QUIC transport (default). Requires `cert_path` for pre-shared coordinator cert.
    #[default]
    Quic,
    /// Raw TLS over TCP. Used in TEE mode where the coordinator has an ephemeral cert
    /// verified via attestation. `cert_path` is not required.
    Tls,
}

/// A coordinator in the network config file.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct NetworkCoordinatorConfig {
    /// The DNS name of the party.
    pub dns_name: Address,
    /// Protocol for the coordinator connection.
    #[serde(default)]
    pub protocol: CoordinatorProtocol,
    /// The path to the public certificate of the party.
    /// Required for QUIC mode, optional for TLS/TEE mode.
    #[serde(default)]
    pub cert_path: Option<PathBuf>,
}

/// A party in the network.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NetworkParty {
    /// The id of the party, 0-based indexing.
    pub id: usize,
    /// The index of the worker in the party.
    pub worker: usize,
    /// The DNS name of the party.
    pub dns_name: Address,
    /// The public certificate of the party.
    pub cert: CertificateDer<'static>,
    /// Connection protocol (only meaningful for coordinators).
    pub protocol: CoordinatorProtocol,
}

impl NetworkParty {
    /// Construct a new [`NetworkParty`] type.
    pub fn new(id: usize, worker: usize, address: Address, cert: CertificateDer<'static>) -> Self {
        Self {
            id,
            worker,
            dns_name: address,
            cert,
            protocol: CoordinatorProtocol::default(),
        }
    }
}

impl TryFrom<NetworkWorkerConfig> for NetworkParty {
    type Error = std::io::Error;
    fn try_from(value: NetworkWorkerConfig) -> Result<Self, Self::Error> {
        let cert = CertificateDer::from(std::fs::read(value.cert_path)?).into_owned();
        Ok(NetworkParty {
            id: value.id,
            worker: value.worker,
            dns_name: value.dns_name,
            cert,
            protocol: CoordinatorProtocol::default(),
        })
    }
}

impl TryFrom<NetworkCoordinatorConfig> for NetworkParty {
    type Error = std::io::Error;
    fn try_from(value: NetworkCoordinatorConfig) -> Result<Self, Self::Error> {
        let cert = match value.cert_path {
            Some(path) => CertificateDer::from(std::fs::read(path)?).into_owned(),
            None => CertificateDer::from(vec![]),
        };
        Ok(NetworkParty {
            id: usize::MAX,
            worker: usize::MAX,
            dns_name: value.dns_name,
            cert,
            protocol: value.protocol,
        })
    }
}

/// The network configuration file.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct NetworkConfigFile {
    /// The list of parties in the network.
    pub parties: Vec<NetworkWorkerConfig>,
    /// The coordinator of the network.
    #[serde(default)]
    pub coordinator: Option<NetworkCoordinatorConfig>,
    /// Our own id in the network.
    #[serde(default)]
    pub my_id: usize,
    /// The is coordinator flag.
    #[serde(default)]
    pub is_coordinator: bool,
    /// The worker id of the party.
    #[serde(default)]
    pub worker: usize,
    /// The [SocketAddr] we bind to.
    pub bind_addr: SocketAddr,
    /// The path to our private key file.
    pub key_path: PathBuf,
    /// The connect timeout in seconds.
    pub timeout_secs: Option<u64>,
    /// Optional TLS listen address for accepting user proof requests.
    #[serde(default)]
    pub user_listen_addr: Option<SocketAddr>,
}

/// The network configuration.
#[derive(Debug, Eq, PartialEq)]
pub struct NetworkConfig {
    /// The list of parties in the network.
    pub parties: Vec<NetworkParty>,
    /// The coordinator of the network.
    pub coordinator: Option<NetworkParty>,
    /// The worker id of the party.
    pub worker: usize,
    /// Our own id in the network.
    pub my_id: usize,
    /// The is coordinator flag.
    pub is_coordinator: bool,
    /// The [SocketAddr] we bind to.
    pub bind_addr: SocketAddr,
    /// The private key.
    pub key: PrivateKeyDer<'static>,
    /// The connect timeout.
    pub timeout: Option<Duration>,
    /// Optional TLS listen address for accepting user proof requests.
    pub user_listen_addr: Option<SocketAddr>,
}

impl NetworkConfig {
    /// Construct a new [`NetworkConfig`] type.
    pub fn new_party(
        id: usize,
        worker: usize,
        bind_addr: SocketAddr,
        key: PrivateKeyDer<'static>,
        parties: Vec<NetworkParty>,
        timeout: Option<Duration>,
    ) -> Self {
        Self {
            parties,
            coordinator: None,
            is_coordinator: false,
            my_id: id,
            worker,
            bind_addr,
            key,
            timeout,
            user_listen_addr: None,
        }
    }

    pub fn for_worker(&self, worker: usize) -> NetworkConfig {
        let mut config = self.clone();
        config.worker = worker;
        config
            .bind_addr
            .set_port(config.bind_addr.port() + 10 * worker as u16);
        config.parties.iter_mut().for_each(|party| {
            party.worker = worker;
            party.dns_name.port = party.dns_name.port + 10 * worker as u16;
        });
        config
    }

    pub fn extend_with_workers(&self, num_workers: usize) -> NetworkConfig {
        let mut config = self.clone();
        let mut new_workers = vec![];
        for worker in 1..num_workers {
            new_workers.extend(config.parties.iter().map(|party| {
                let mut party = party.clone();
                party.worker = worker;
                party.dns_name.port = party.dns_name.port + 10 * worker as u16;
                party
            }));
        }
        config.parties.extend(new_workers);

        config
    }
}

impl TryFrom<NetworkConfigFile> for NetworkConfig {
    type Error = std::io::Error;
    fn try_from(value: NetworkConfigFile) -> Result<Self, Self::Error> {
        let parties = value
            .parties
            .into_iter()
            .map(NetworkParty::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(std::fs::read(value.key_path)?))
            .clone_key();
        Ok(NetworkConfig {
            parties,
            is_coordinator: value.is_coordinator,
            coordinator: value.coordinator.map(NetworkParty::try_from).transpose()?,
            my_id: value.my_id,
            worker: value.worker,
            bind_addr: value.bind_addr,
            key,
            timeout: value.timeout_secs.map(Duration::from_secs),
            user_listen_addr: value.user_listen_addr,
        })
    }
}

impl Clone for NetworkConfig {
    fn clone(&self) -> Self {
        Self {
            parties: self.parties.clone(),
            coordinator: self.coordinator.clone(),
            is_coordinator: self.is_coordinator,
            my_id: self.my_id,
            worker: self.worker,
            bind_addr: self.bind_addr,
            key: self.key.clone_key(),
            timeout: self.timeout,
            user_listen_addr: self.user_listen_addr,
        }
    }
}

impl NetworkConfig {
    /// Basic sanity checks for the configuration.
    pub fn check_config(&self) -> eyre::Result<()> {
        // sanity check config
        // 1. check that my_id is in the list of parties
        self.parties
            .iter()
            .find(|p| p.id == self.my_id)
            .ok_or_else(|| {
                eyre::eyre!(
                    "my_id {} not found in list of parties: {:?}",
                    self.my_id,
                    self.parties
                )
            })?;
        // 2. check that all parties have a unique id
        let mut ids = self.parties.iter().map(|p| p.id).collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != self.parties.len() {
            return Err(eyre::eyre!("duplicate party ids found"));
        }
        Ok(())
    }

    pub fn generate_worker_configs(
        num_workers: usize,
    ) -> (
        BTreeMap<PartyWorkerID, NetworkConfigFile>,
        NetworkConfigFile,
    ) {
        Self::generate_worker_configs_with_dir(num_workers, "data")
    }

    pub fn generate_worker_configs_with_dir(
        num_workers: usize,
        data_dir: &str,
    ) -> (
        BTreeMap<PartyWorkerID, NetworkConfigFile>,
        NetworkConfigFile,
    ) {
        let mut parties = vec![
            NetworkWorkerConfig {
                id: 0,
                worker: 0,
                dns_name: "localhost:10000".parse().unwrap(),
                cert_path: format!("{data_dir}/cert0_0.der").into(),
            },
            NetworkWorkerConfig {
                id: 1,
                worker: 0,
                dns_name: "localhost:10001".parse().unwrap(),
                cert_path: format!("{data_dir}/cert0_1.der").into(),
            },
            NetworkWorkerConfig {
                id: 2,
                worker: 0,
                dns_name: "localhost:10002".parse().unwrap(),
                cert_path: format!("{data_dir}/cert0_2.der").into(),
            },
        ];
        let coordinator = NetworkCoordinatorConfig {
            dns_name: "localhost:20000".parse().unwrap(),
            protocol: CoordinatorProtocol::default(),
            cert_path: Some(format!("{data_dir}/cert_coordinator.der").into()),
        };
        let mut workers = BTreeMap::new();

        for worker in 0..num_workers {
            let worker_port_offset = 1000 * worker as u16;

            for party in 0..=2usize {
                parties[party].worker = worker;
                parties[party].dns_name.port += worker_port_offset;
                parties[party].cert_path = format!("{data_dir}/cert{worker}_{party}.der").into();
            }

            for party in 0..=2usize {
                workers.insert(
                    PartyWorkerID::new(party, worker),
                    NetworkConfigFile {
                        my_id: party,
                        worker,
                        bind_addr: SocketAddr::new(
                            IpAddr::from_str("0.0.0.0").unwrap(),
                            10000 + party as u16 + worker_port_offset,
                        ),
                        key_path: format!("{data_dir}/key{worker}_{party}.der").into(),
                        parties: parties.clone(),
                        coordinator: Some(coordinator.clone()),
                        is_coordinator: false,
                        timeout_secs: None,
                        user_listen_addr: None,
                    },
                );
            }
        }

        let coordinator_config = NetworkConfigFile {
            is_coordinator: true,
            my_id: 0,
            worker: 0,
            bind_addr: SocketAddr::new(IpAddr::from_str("0.0.0.0").unwrap(), 20000),
            key_path: format!("{data_dir}/key_coordinator.der").into(),
            parties: (0..num_workers)
                .flat_map(|i| {
                    workers
                        .get(&PartyWorkerID::new(0, i))
                        .unwrap()
                        .parties
                        .clone()
                })
                .collect(),
            coordinator: Some(coordinator),
            timeout_secs: None,
            user_listen_addr: None,
        };

        (workers, coordinator_config)
    }
}
