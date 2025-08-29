//! A channel abstraction for sending and receiving messages.
use crate::{
    rep3::quic::codec_cfg,
    resv_demux::{header, RecvHub},
};
use bytes::{Bytes, BytesMut};
use futures::{Sink, SinkExt, Stream, StreamExt, TryStreamExt};
use quinn::{Connection, ConnectionError, RecvStream, SendStream};
use std::{
    io,
    marker::{PhantomData, Unpin},
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    runtime::Handle,
    sync::{mpsc, oneshot, Semaphore},
};
use tokio_util::codec::{Decoder, Encoder, FramedRead, FramedWrite, LengthDelimitedCodec};

// use crate::rep3::quic::RUNTIME;

/// A read end of the channel, just a type alias for [`FramedRead`].
pub type ReadChannel<T, D> = FramedRead<T, D>;
/// A write end of the channel, just a type alias for [`FramedWrite`].
pub type WriteChannel<T, E> = FramedWrite<T, E>;

/// A channel that uses a [`Encoder`] and [`Decoder`] to send and receive messages.
#[derive(Debug)]
pub struct Channel<R, W, C> {
    read_conn: ReadChannel<R, C>,
    write_conn: WriteChannel<W, C>,
}

/// A channel that uses a [`LengthDelimitedCodec`] to send and receive messages.
pub type BytesChannel<R, W> = Channel<R, W, LengthDelimitedCodec>;

impl<R, W, C> Channel<R, W, C> {
    /// Create a new [`Channel`], backed by a read and write half. Read and write buffers
    /// are automatically handled by [`LengthDelimitedCodec`].
    pub fn new<MSend>(read_half: R, write_half: W, codec: C) -> Self
    where
        C: Clone + Decoder + Encoder<MSend>,
        R: AsyncReadExt,
        W: AsyncWriteExt,
    {
        Channel {
            write_conn: FramedWrite::new(write_half, codec.clone()),
            read_conn: FramedRead::new(read_half, codec),
        }
    }

    /// Split Connection into a ([`WriteChannel`],[`ReadChannel`]) pair.
    pub fn split(self) -> (WriteChannel<W, C>, ReadChannel<R, C>) {
        (self.write_conn, self.read_conn)
    }

    /// Join ([`WriteChannel`],[`ReadChannel`]) pair back into a [`Channel`].
    pub fn join(write_conn: WriteChannel<W, C>, read_conn: ReadChannel<R, C>) -> Self {
        Self {
            write_conn,
            read_conn,
        }
    }

    /// Returns mutable reference to the ([`WriteChannel`],[`ReadChannel`]) pair.
    pub fn inner_ref(&mut self) -> (&mut WriteChannel<W, C>, &mut ReadChannel<R, C>) {
        (&mut self.write_conn, &mut self.read_conn)
    }

    /// Closes the channel, flushing the write buffer and checking that there is no unread data.
    pub async fn close<MSend>(self) -> Result<(), io::Error>
    where
        C: Encoder<MSend, Error = std::io::Error> + Decoder<Error = std::io::Error>,
        R: AsyncReadExt + Unpin,
        W: AsyncWriteExt + Unpin,
    {
        let Channel {
            mut read_conn,
            mut write_conn,
            ..
        } = self;
        write_conn.flush().await?;
        write_conn.close().await?;
        if let Some(x) = read_conn.next().await {
            match x {
                Ok(_) => {
                    return Err(io::Error::other(
                        "Unexpected data on read channel when closing connections",
                    ));
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        Ok(())
    }
}
impl<R, W: AsyncWriteExt + Unpin, MSend, C: Encoder<MSend, Error = io::Error>> Sink<MSend>
    for Channel<R, W, C>
where
    Self: Unpin,
{
    type Error = <C as Encoder<MSend>>::Error;

    fn poll_ready(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.write_conn.poll_ready_unpin(cx)
    }

    fn start_send(mut self: std::pin::Pin<&mut Self>, item: MSend) -> Result<(), Self::Error> {
        self.write_conn.start_send_unpin(item)
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.write_conn.poll_flush_unpin(cx)
    }

    fn poll_close(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.write_conn.poll_close_unpin(cx)
    }
}
impl<R: AsyncReadExt + Unpin, W, MRecv, C: Decoder<Item = MRecv, Error = io::Error>> Stream
    for Channel<R, W, C>
where
    Self: Unpin,
{
    type Item = Result<MRecv, <C as Decoder>::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.read_conn.poll_next_unpin(cx)
    }
}

struct WriteJob<MSend> {
    data: MSend,
    ret: oneshot::Sender<Result<(), io::Error>>,
}

struct ReadJob<MRecv> {
    ret: oneshot::Sender<Result<MRecv, io::Error>>,
}

/// A handle to a channel that allows sending and receiving messages.
#[derive(Debug, Clone)]
pub struct ChannelHandle<MSend, MRecv> {
    write_job_queue: mpsc::Sender<WriteJob<MSend>>,
    read_job_queue: mpsc::Sender<ReadJob<MRecv>>,
}

impl<MSend, MRecv> ChannelHandle<MSend, MRecv>
where
    MRecv: Send + std::fmt::Debug + 'static,
    MSend: Send + std::fmt::Debug + 'static,
{
    /// Create a new [`ChannelHandle`] from a [`Channel`]. This spawns a new tokio task that handles the read and write jobs so they can happen concurrently.
    pub fn manage<R, W, C>(chan: Channel<R, W, C>) -> ChannelHandle<MSend, MRecv>
    where
        C: 'static,
        R: AsyncReadExt + Unpin + 'static,
        W: AsyncWriteExt + Unpin + 'static,
        FramedRead<R, C>: Stream<Item = Result<MRecv, io::Error>> + Send,
        FramedWrite<W, C>: Sink<MSend, Error = io::Error> + Send,
    {
        let (write_send, mut write_recv) = mpsc::channel::<WriteJob<MSend>>(1024);
        let (read_send, mut read_recv) = mpsc::channel::<ReadJob<MRecv>>(1024);

        let (mut write, mut read) = chan.split();

        const RESET_LIMIT: usize = 1 << 30; // 1 GiB
        const BASE_CAP: usize = 8 * 1024;

        tokio::spawn(async move {
            while let Some(frame) = read.next().await {
                let job = read_recv.recv().await;
                match job {
                    Some(job) => {
                        if job.ret.send(frame).is_err() {
                            tracing::warn!("Warning: Read Job finished but receiver is gone!");
                        }

                        // free capacity after receiving large frames (witness shares)
                        if read.read_buffer().is_empty()
                            && read.read_buffer().capacity() > RESET_LIMIT
                        {
                            *read.read_buffer_mut() = BytesMut::with_capacity(BASE_CAP);
                        }
                    }
                    None => {
                        if frame.is_ok() {
                            tracing::warn!("Warning: received Ok frame but receiver is gone!");
                        }
                        break;
                    }
                }
            }
        });
        tokio::spawn(async move {
            while let Some(write_job) = write_recv.recv().await {
                match write.send(write_job.data).await {
                    Ok(_) => {
                        // we don't really care if the receiver for a write job is gone, as this is a common case
                        // therefore we only emit a trace message
                        if write_job.ret.send(Ok(())).is_err() {
                            tracing::trace!("Debug: Write Job finished but receiver is gone!");
                        }

                        // workaround to free capacity after sending large frames (witness shares)
                        let buf = write.write_buffer();
                        if buf.is_empty() && buf.capacity() > RESET_LIMIT {
                            *write.write_buffer_mut() = BytesMut::with_capacity(BASE_CAP);
                        }
                    }
                    Err(err) => {
                        tracing::error!("Write job failed: {err}");
                    }
                }
            }
        });

        ChannelHandle {
            write_job_queue: write_send,
            read_job_queue: read_send,
        }
    }

    /// Instructs the channel to send a message. Returns a [oneshot::Receiver] that will return the result of the send operation.
    pub async fn send(&self, data: MSend) -> oneshot::Receiver<Result<(), io::Error>> {
        let (ret, recv) = oneshot::channel();
        let job = WriteJob { data, ret };
        match self.write_job_queue.send(job).await {
            Ok(_) => {}
            Err(job) => job
                .0
                .ret
                .send(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "ChannelHandle: send Channel is gone",
                )))
                .unwrap(),
        }
        recv
    }

    /// Instructs the channel to receive a message. Returns a [oneshot::Receiver] that will return the result of the receive operation.
    pub async fn recv(&self) -> oneshot::Receiver<Result<MRecv, io::Error>> {
        let (ret, recv) = oneshot::channel();
        let job = ReadJob { ret };
        match self.read_job_queue.send(job).await {
            Ok(_) => {}
            Err(job) => job
                .0
                .ret
                .send(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "ChannelHandle: recv Channel is gone",
                )))
                .unwrap(),
        }
        recv
    }

    /// A blocking version of [ChannelHandle::send]. This will block until the send operation is complete.
    pub fn blocking_send(&self, data: MSend) -> oneshot::Receiver<Result<(), io::Error>> {
        let (ret, recv) = oneshot::channel();
        let job = WriteJob { data, ret };
        match self.write_job_queue.blocking_send(job) {
            Ok(_) => {}
            Err(job) => job
                .0
                .ret
                .send(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "ChannelHandle: send Channel is gone",
                )))
                .unwrap(),
        }
        recv
    }

    /// A blocking version of [ChannelHandle::recv]. This will block until the receive operation is complete.
    pub fn blocking_recv(&self) -> oneshot::Receiver<Result<MRecv, io::Error>> {
        let (ret, recv) = oneshot::channel();
        let job = ReadJob { ret };
        match self.read_job_queue.blocking_send(job) {
            Ok(_) => {}
            Err(job) => job
                .0
                .ret
                .send(Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "ChannelHandle: recv Channel is gone",
                )))
                .unwrap(),
        }
        recv
    }
}

fn ioerr(s: &str) -> io::Error {
    io::Error::new(io::ErrorKind::Other, s)
}

#[derive(Debug, Clone)]
pub struct PerOpChannelHandle {
    conn: Connection,
    codec: LengthDelimitedCodec,
    rt: Handle,
    send_limit: Arc<Semaphore>, // bound concurrent sends
    fork_id: u32,
    seq: Arc<AtomicU64>,
    resv_demux: Option<Arc<RecvHub>>,
    inbox: Option<Arc<tokio::sync::Mutex<mpsc::Receiver<(u64, BytesMut)>>>>, // from RecvHub::register(...)
}

impl PerOpChannelHandle {
    pub fn new(
        conn_next: Connection,
        resv_demux: Option<Arc<RecvHub>>,
        codec: LengthDelimitedCodec,
        rt: Handle,
        fork_id: u32,
        per_conn_streams: usize,
    ) -> Self {
        let inbox = if let Some(resv_demux) = resv_demux.as_ref() {
            resv_demux.start(); // idempotent multi-calls ok
            Some(Arc::new(tokio::sync::Mutex::new(
                resv_demux.register(fork_id),
            )))
        } else {
            None
        };

        Self {
            conn: conn_next,
            codec,
            rt,
            send_limit: Arc::new(Semaphore::new(per_conn_streams)),
            fork_id,
            resv_demux,
            seq: Arc::new(AtomicU64::new(0)),
            inbox,
        }
    }

    pub fn fork(&self, fork_id: u32) -> Self {
        let inbox = if let Some(resv_demux) = self.resv_demux.as_ref() {
            resv_demux.start(); // idempotent multi-calls ok
            Some(Arc::new(tokio::sync::Mutex::new(
                resv_demux.register(fork_id),
            )))
        } else {
            None
        };

        Self {
            conn: self.conn.clone(),
            codec: codec_cfg(),
            rt: self.rt.clone(),
            send_limit: self.send_limit.clone(),
            fork_id,
            seq: Arc::new(AtomicU64::new(0)),
            inbox,
            resv_demux: self.resv_demux.clone(),
        }
    }

    // --- SAME API as your ChannelHandle ---

    /// Async: schedule a SEND to next party (per-op UNI stream). Returns oneshot for send result.
    pub async fn send(&self, data: Bytes) -> oneshot::Receiver<Result<(), io::Error>> {
        self.spawn_send(data)
    }

    /// Async: schedule a RECV from prev party (accept next UNI). Returns oneshot for received message.
    pub async fn recv(&self) -> oneshot::Receiver<Result<BytesMut, io::Error>> {
        self.spawn_recv()
    }

    /// Blocking variants return a oneshot Receiver you can `.blocking_recv()`.
    pub fn blocking_send(&self, data: Bytes) -> oneshot::Receiver<Result<(), io::Error>> {
        self.spawn_send(data)
    }
    pub fn blocking_recv(&self) -> oneshot::Receiver<Result<BytesMut, io::Error>> {
        self.spawn_recv()
    }

    fn spawn_send(&self, payload: Bytes) -> oneshot::Receiver<Result<(), io::Error>> {
        let (tx, rx) = oneshot::channel();
        let conn = self.conn.clone();
        let codec = self.codec.clone();
        let fork_id = self.fork_id;
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let out_sem = self.send_limit.clone();

        self.rt.spawn(async move {
            let _p = match out_sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    let _ = tx.send(err("sem closed"));
                    return;
                }
            };
            // prepend header inside one LDC frame
            let mut buf = BytesMut::with_capacity(13 + payload.len());
            buf.extend_from_slice(&header(fork_id, seq));
            buf.extend_from_slice(&payload);
            // open per-op UNI and send framed
            let send: SendStream = match conn.open_uni().await {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(io_err(e)));
                    return;
                }
            };
            let mut fw = FramedWrite::new(send, codec);
            let res = async {
                fw.send(buf.freeze()).await?;
                fw.flush().await?;
                let mut s = fw.into_inner();
                s.finish().map_err(io_err)
            }
            .await;
            let _ = tx.send(res);
        });
        rx
    }

    fn spawn_recv(&self) -> oneshot::Receiver<Result<BytesMut, io::Error>> {
        let inbox = self.inbox.as_ref().expect("send-only connection").clone();
        let (tx, rx) = oneshot::channel();
        self.rt.spawn(async move {
            let mut rxq = inbox.lock().await;
            match rxq.recv().await {
                Some((_seq, bytes)) => {
                    let _ = tx.send(Ok(bytes));
                }
                None => {
                    let _ = tx.send(Err(io_err("inbox closed")));
                }
            }
        });
        rx
    }
}

#[inline]
fn err(msg: &str) -> Result<(), io::Error> {
    Err(io::Error::new(io::ErrorKind::Other, msg))
}

#[inline]
fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(std::io::ErrorKind::Other, e.to_string())
}
