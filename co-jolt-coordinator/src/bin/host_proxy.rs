//! Untrusted TCP↔vsock relay for bridging MPC workers to the enclave coordinator.
//!
//! Runs on the EC2 host (NOT inside the enclave). For each incoming TCP
//! connection, opens a vsock connection to the enclave and pipes bytes
//! bidirectionally. The proxy never sees plaintext — TLS terminates inside
//! the enclave.
//!
//! Configuration via environment variables:
//! - `TCP_LISTEN_ADDR` — TCP address to bind (default: `0.0.0.0:9000`)
//! - `VSOCK_CID` — enclave vsock CID (default: `3`, standard Nitro enclave CID)
//! - `VSOCK_PORT` — enclave vsock port (default: `9000`)

use std::io;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use vsock::VsockStream;

fn relay(id: usize, tcp: TcpStream, vsock_cid: u32, vsock_port: u32, active: Arc<AtomicUsize>) {
    let peer = tcp.peer_addr().ok();
    eprintln!("[conn {id}] accepted from {peer:?}");
    let _ = tcp.set_nodelay(true);

    let vsock = match VsockStream::connect_with_cid_port(vsock_cid, vsock_port) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[conn {id}] vsock connect failed: {e}");
            return;
        }
    };

    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
    eprintln!("[conn {id}] connected to enclave (active: {count})");

    let mut tcp_r = tcp.try_clone().unwrap();
    let mut vs_r = vsock.try_clone().unwrap();
    let mut tcp_w = tcp;
    let mut vs_w = vsock;
    let active2 = active.clone();

    // TCP → vsock
    let t1 = thread::spawn(move || {
        let bytes = io::copy(&mut tcp_r, &mut vs_w).unwrap_or(0);
        eprintln!("[conn {id}] tcp→vsock done ({bytes} bytes)");
        let _ = vs_w.shutdown(std::net::Shutdown::Write);
    });

    // vsock → TCP
    let bytes = io::copy(&mut vs_r, &mut tcp_w).unwrap_or(0);
    eprintln!("[conn {id}] vsock→tcp done ({bytes} bytes)");
    let _ = tcp_w.shutdown(std::net::Shutdown::Write);

    let _ = t1.join();
    let remaining = active2.fetch_sub(1, Ordering::SeqCst) - 1;
    eprintln!("[conn {id}] closed (active: {remaining})");
}

fn main() -> io::Result<()> {
    let listen_addr = std::env::var("TCP_LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:9000".into());
    let vsock_cid: u32 =
        std::env::var("VSOCK_CID").unwrap_or_else(|_| "3".into()).parse().expect("VSOCK_CID must be a valid u32");
    let vsock_port: u32 =
        std::env::var("VSOCK_PORT").unwrap_or_else(|_| "9000".into()).parse().expect("VSOCK_PORT must be a valid u32");

    let listener = TcpListener::bind(&listen_addr)?;
    eprintln!("host_proxy listening on {listen_addr} → vsock {vsock_cid}:{vsock_port}");

    let active = Arc::new(AtomicUsize::new(0));

    for (id, stream) in listener.incoming().enumerate() {
        match stream {
            Ok(tcp) => {
                let active = active.clone();
                thread::spawn(move || relay(id, tcp, vsock_cid, vsock_port, active));
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}
