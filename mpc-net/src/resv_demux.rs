use bytes::{Buf, BufMut, BytesMut};
use futures::TryStreamExt;
use quinn::Connection;
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    io,
    sync::Arc,
};
use tokio::{
    runtime::Handle,
    sync::{mpsc, oneshot, Semaphore},
};
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};

const PROTOCOL: u32 = u32::from_be_bytes(*b"REP3");
const VER: u8 = 1;
pub const HEADER_LEN: usize = 4 + 1 + 4 + 8;
const PER_FORK_WINDOW: usize = 4; // tune

#[inline]
pub fn write_header(dst: &mut BytesMut, fork_id: u32, seq: u64) {
    dst.reserve(HEADER_LEN);
    dst.put_u32(PROTOCOL);
    dst.put_u8(VER);
    dst.put_u32(fork_id);
    dst.put_u64(seq);
}

fn parse_header(frame: &mut BytesMut) -> io::Result<(u32, u64)> {
    if frame.len() < HEADER_LEN {
        return Err(ioe("short"));
    }
    let mut h = frame.split_to(HEADER_LEN);
    let m = h.get_u32();
    let v = h.get_u8();
    let fork = h.get_u32();
    let seq = h.get_u64();
    if m != PROTOCOL || v != VER {
        return Err(ioe("bad hdr"));
    }
    Ok((fork, seq))
}

#[derive(Debug)]
pub struct RecvJob {
    pub fork: u32,
    pub tx: oneshot::Sender<Result<BytesMut, io::Error>>,
}

struct ForkState {
    next_assign: u64,
    pending: BTreeMap<u64, BytesMut>,
    waiters: BTreeMap<u64, oneshot::Sender<Result<BytesMut, io::Error>>>,
    backlog: VecDeque<oneshot::Sender<Result<BytesMut, io::Error>>>,
}
impl ForkState {
    fn new() -> Self {
        Self {
            next_assign: 0,
            pending: BTreeMap::new(),
            waiters: BTreeMap::new(),
            backlog: VecDeque::new(),
        }
    }
    fn outstanding(&self) -> usize {
        self.waiters.len()
    }
}

fn ioe<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
}

pub struct RecvDemux;
impl RecvDemux {
    pub fn handle(
        conn_prev: Connection,
        codec: LengthDelimitedCodec,
        rt: Handle,
        inflight_reads: usize,
    ) -> mpsc::Sender<RecvJob> {
        let (tx, mut rx) = mpsc::channel::<RecvJob>(2048);

        // accept/read task → sends HubMsg::Data into a local mpsc
        let (dt, mut dr) = mpsc::channel::<(u32, u64, BytesMut)>(2048);
        let inflight = Arc::new(Semaphore::new(inflight_reads));
        let codec_read = codec.clone();
        let infl = inflight.clone();
        let _rt = rt.clone();
        rt.spawn(async move {
            loop {
                let Ok(rs) = conn_prev.accept_uni().await else {
                    break;
                };
                let codec = codec_read.clone();
                let dt = dt.clone();
                let infl = infl.clone();
                _rt.spawn(async move {
                    let _p = infl.acquire_owned().await.ok();
                    let mut fr = FramedRead::new(rs, codec);
                    if let Ok(Some(mut frame)) = fr.try_next().await {
                        if let Ok((fork, seq)) = parse_header(&mut frame) {
                            let _ = dt.send((fork, seq, frame)).await;
                        }
                    }
                });
            }
        });

        // main actor: match reqs↔data by seq per fork
        rt.spawn(async move {
            let mut forks: HashMap<u32, ForkState> = HashMap::new();

            // helper: after a delivery, backfill from backlog while window allows
            let try_assign_from_backlog = |fs: &mut ForkState| {
                while fs.outstanding() < PER_FORK_WINDOW {
                    let Some(tx) = fs.backlog.pop_front() else {
                        break;
                    };
                    let seq = fs.next_assign;
                    fs.next_assign += 1;
                    if let Some(payload) = fs.pending.remove(&seq) {
                        let _ = tx.send(Ok(payload));
                    } else {
                        fs.waiters.insert(seq, tx);
                    }
                }
            };

            loop {
                tokio::select! {
                    Some((fork, seq, payload)) = dr.recv() => {
                        let fs = forks.entry(fork).or_insert_with(ForkState::new);
                        if let Some(w) = fs.waiters.remove(&seq) {
                            let _ = w.send(Ok(payload));
                            // a waiter was satisfied → window slot freed
                            try_assign_from_backlog(fs);
                        } else {
                            // data ahead of assigned seqs → park it
                            fs.pending.insert(seq, payload);
                        }
                    }
                    Some(RecvJob { fork, tx }) = rx.recv() => {
                        let fs = forks.entry(fork).or_insert_with(ForkState::new);
                        if fs.outstanding() < PER_FORK_WINDOW {
                            let seq = fs.next_assign; fs.next_assign += 1;
                            if let Some(payload) = fs.pending.remove(&seq) {
                                let _ = tx.send(Ok(payload));
                                // delivered immediately; window slot used+freed → refill from backlog if any
                                try_assign_from_backlog(fs);
                            } else {
                                fs.waiters.insert(seq, tx);
                            }
                        } else {
                            // window full; queue requester
                            fs.backlog.push_back(tx);
                        }
                    }
                    else => break,
                }
            }

            // shutdown: fail outstanding waiters
            for (_, mut fs) in forks.drain() {
                for (_, tx) in fs.waiters {
                    let _ = tx.send(Err(ioe("hub closed")));
                }
                for tx in fs.backlog.drain(..) {
                    let _ = tx.send(Err(ioe("hub closed")));
                }
            }
        });

        tx
    }
}
