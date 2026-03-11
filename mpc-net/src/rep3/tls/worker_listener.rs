//! TLS server for accepting user proof request connections on workers.
//!
//! Each worker exposes a TLS listener port (configured via `user_listen_addr`)
//! where the user's `ProvingClient` connects to send trace shares and receive
//! the assembled proof.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;

use eyre::Context;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

type TlsStream = rustls::StreamOwned<rustls::ServerConnection, TcpStream>;

/// A TLS server that accepts user connections on a worker node.
pub struct TlsWorkerListener {
    listener: TcpListener,
    tls_config: Arc<rustls::ServerConfig>,
}

impl TlsWorkerListener {
    /// Bind a TLS listener on the given address using the worker's cert and key.
    pub fn bind(
        addr: SocketAddr,
        cert: CertificateDer<'static>,
        key: PrivateKeyDer<'static>,
    ) -> eyre::Result<Self> {
        let tls_config = Arc::new(
            rustls::ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(vec![cert], key)
                .context("building TLS server config for user listener")?,
        );

        let listener = TcpListener::bind(addr)
            .with_context(|| format!("binding user TLS listener on {addr}"))?;

        Ok(Self {
            listener,
            tls_config,
        })
    }

    /// Accept a single user connection. Blocks until a connection arrives.
    pub fn accept(&self) -> eyre::Result<TlsUserConnection> {
        let (tcp_stream, peer_addr) = self
            .listener
            .accept()
            .context("accepting user TCP connection")?;
        tcp_stream.set_nodelay(true)?;

        let tls_conn = rustls::ServerConnection::new(Arc::clone(&self.tls_config))
            .context("creating TLS server connection for user")?;
        let stream = rustls::StreamOwned::new(tls_conn, tcp_stream);

        Ok(TlsUserConnection {
            stream,
            peer_addr,
        })
    }
}

/// An established TLS connection from a user's `ProvingClient`.
pub struct TlsUserConnection {
    stream: TlsStream,
    peer_addr: SocketAddr,
}

impl std::fmt::Debug for TlsUserConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TlsUserConnection")
            .field("peer_addr", &self.peer_addr)
            .finish()
    }
}

impl TlsUserConnection {
    /// Peer address of the connected user.
    pub fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// Receive a length-prefixed message from the user.
    pub fn recv(&mut self) -> io::Result<Vec<u8>> {
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf)?;
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        self.stream.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Send a length-prefixed message to the user.
    pub fn send(&mut self, data: &[u8]) -> io::Result<()> {
        self.stream.write_all(&(data.len() as u32).to_be_bytes())?;
        self.stream.write_all(data)?;
        self.stream.flush()
    }
}
