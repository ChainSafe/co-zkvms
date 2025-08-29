use bytes::{Buf, BufMut, BytesMut};
use futures::TryStreamExt;
use quinn::{Connection, RecvStream};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    io,
    sync::Arc,
};
use tokio::{
    io::AsyncReadExt,
    runtime::Handle,
    sync::{mpsc, oneshot, Semaphore},
};
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};

const PROTOCOL: &'static [u8] = b"REP3";
const VER: u8 = 1;

pub(crate) const HEADER_LEN: usize = 17;

#[inline]
pub fn write_header(dst: &mut BytesMut, fork_id: u32, seq: u64) {
    dst.reserve(HEADER_LEN);
    dst.put(PROTOCOL);
    dst.put_u8(VER);
    dst.put_u32(fork_id);
    dst.put_u64(seq);
}

pub struct ResvJob {
    pub fork: u32,
    pub tx: oneshot::Sender<Result<BytesMut, io::Error>>,
}

pub struct RecvDemux {}

impl RecvDemux {
    pub fn handle(
        conn: Connection,
        codec: LengthDelimitedCodec,
        rt: Handle,
        inflight_max: usize,
    ) -> mpsc::Sender<ResvJob> {
        let (req_tx, mut req_rx) = mpsc::channel::<ResvJob>(1024);
        rt.spawn(async move {
            let inflight = Arc::new(Semaphore::new(inflight_max));
            let mut forks: HashMap<u32, ForkState> = HashMap::new();

            let serve_ready = |fork_id: u32, forks: &mut HashMap<u32, ForkState>| {
                if let Some(fs) = forks.get_mut(&fork_id) {
                    while let Some(payload) = fs.pending.remove(&fs.expected) {
                        let Some(reply) = fs.waiters.pop_front() else {
                            // no waiter yet; put stream back and stop
                            fs.pending.insert(fs.expected, payload);
                            break;
                        };
                        let permit = inflight.clone();
                        tokio::spawn(async move {
                            let _permit = permit.acquire_owned().await.ok();
                            let _ = reply.send(Ok(payload));
                        });
                        fs.expected = fs.expected.wrapping_add(1);
                    }
                }
            };

            loop {
                tokio::select! {
                    accepted = conn.accept_uni() => {
                        let rs = match accepted {
                            Ok(rs) => rs,
                            Err(e) => {
                                Self::notify_all_waiters(&mut forks, e);
                                break;
                            }
                        };

                        let mut payload = match FramedRead::new(rs, codec.clone()).try_next().await {
                            Ok(Some(buf)) => buf,
                            Ok(None) => {
                                panic!("Failed to read from stream");
                            }
                            Err(e) => {
                                panic!("Failed to read from stream: {}", e);
                            }
                        };

                        let (fork, seq) = match Self::read_header(&mut payload).await {
                            Ok((fork, seq)) => (fork, seq),
                            Err(e) => {
                                panic!("Failed to read header: {}", e);
                                // continue;
                            }
                        };

                        let fs = forks.entry(fork).or_insert_with(|| ForkState {
                            expected: 0, pending: BTreeMap::new(), waiters: VecDeque::new()
                        });
                        if fs.pending.insert(seq, payload).is_some() {
                            tracing::warn!("Duplicate sequence number received");
                            // duplicate seq; last wins
                        }
                        serve_ready(fork, &mut forks);
                    }
                    req = req_rx.recv() => {
                        let Some(ResvJob{ fork, tx }) = req else { Self::notify_all_waiters(&mut forks, "req_rx dropped"); break; };
                        let fs = forks.entry(fork).or_insert_with(|| ForkState {
                            expected: 0, pending: BTreeMap::new(), waiters: VecDeque::new()
                        });
                        if let Some(payload) = fs.pending.remove(&fs.expected) {
                            let permit = inflight.clone();
                            tokio::spawn(async move {
                                let _permit = permit.acquire_owned().await.ok();
                                let _ = tx.send(Ok(payload));
                            });
                            fs.expected = fs.expected.wrapping_add(1);
                            serve_ready(fork, &mut forks);
                        } else {
                            fs.waiters.push_back(tx); // if receiver drops, send() will fail later; fine
                        }
                    }
                }
            }
        });
        req_tx
    }

    #[inline]
    async fn read_header(rs: &mut BytesMut) -> io::Result<(u32, u64)> {
        let mut header = rs.split_to(HEADER_LEN);
        let protocol = header.get_u32().to_be_bytes();
        if protocol != PROTOCOL {
            return Err(io_err(format!(
                "unexpected protocol, expected: {:?}, got: {:?}",
                PROTOCOL, protocol,
            )));
        }
        let ver = header.get_u8();
        if ver != VER {
            return Err(io_err(format!(
                "unexpected version, expected: {}, got: {}",
                VER, ver,
            )));
        }
        let fork = header.get_u32();
        let seq = header.get_u64();
        Ok((fork, seq))
    }

    fn notify_all_waiters<E: std::fmt::Display + Clone>(forks: &mut HashMap<u32, ForkState>, e: E) {
        tracing::warn!("recv hub closed");
        for (_, mut fs) in forks.drain() {
            while let Some(tx) = fs.waiters.pop_front() {
                let _ = tx.send(Err(io_err(e.clone())));
            }
        }
    }
}

// Per-fork state kept local to the hub task (no locking needed).
struct ForkState {
    expected: u64,
    pending: BTreeMap<u64, BytesMut>, // seq -> stream (body unread)
    waiters: VecDeque<oneshot::Sender<Result<BytesMut, io::Error>>>,
}

#[inline]
fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
}
