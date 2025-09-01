//! A channel abstraction for sending and receiving messages.
use crate::rep3::{quic::codec_cfg, PartyWorkerID};
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
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    runtime::Handle,
    sync::{mpsc, oneshot, Semaphore},
};
use tokio_util::codec::{Decoder, Encoder, FramedRead, FramedWrite, LengthDelimitedCodec};
use tracing::Instrument;

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

        tokio::spawn(
            async move {
                while let Some(frame) = read.next().await {
                    let job = read_recv.recv().await;
                    match job {
                        Some(job) => {
                            if job.ret.send(frame).is_err() {
                                // tracing::warn!("Warning: Read Job finished but receiver is gone!");
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
                                // tracing::warn!("Warning: received Ok frame but receiver is gone!");
                            }
                            break;
                        }
                    }
                }
            }
            .instrument(tracing::Span::none()),
        );
        tokio::spawn(
            async move {
                while let Some(write_job) = write_recv.recv().await {
                    match write.send(write_job.data).await {
                        Ok(_) => {
                            // we don't really care if the receiver for a write job is gone, as this is a common case
                            // therefore we only emit a trace message
                            if write_job.ret.send(Ok(())).is_err() {
                                // tracing::trace!("Debug: Write Job finished but receiver is gone!");
                            }

                            // workaround to free capacity after sending large frames (witness shares)
                            let buf = write.write_buffer();
                            if buf.is_empty() && buf.capacity() > RESET_LIMIT {
                                *write.write_buffer_mut() = BytesMut::with_capacity(BASE_CAP);
                            }
                        }
                        Err(err) => {
                            // tracing::error!("Write job failed: {err}");
                        }
                    }
                }
            }
            .instrument(tracing::Span::none()),
        );

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

// TODO: remove?
impl ChannelHandle<Bytes, BytesMut> {
    /// Optimized manager for Bytes over QUIC: chunked length-prefixed I/O without LengthDelimitedCodec buffering.
    /// This keeps message semantics the same for callers while reducing large-frame burstiness.
    pub fn manage_bytes_quic(
        chan: Channel<RecvStream, SendStream, LengthDelimitedCodec>,
    ) -> ChannelHandle<Bytes, BytesMut> {
        // Chunking and framing params
        const LEN_BYTES: usize = 5; // match existing LengthDelimitedCodec config
        const WRITE_CHUNK: usize = 16 * 1024; // 16 KiB fairness chunks

        // Small helpers for 5-byte big-endian length prefix
        #[inline]
        async fn write_len_prefix(w: &mut SendStream, len: usize) -> io::Result<()> {
            let v = len as u64;
            let mut buf = [0u8; LEN_BYTES];
            for i in (0..LEN_BYTES).rev() {
                buf[i] = (v >> (8 * (LEN_BYTES - 1 - i)) & 0xFF) as u8;
            }
            w.write_all(&buf).await.map_err(io_err)
        }

        #[inline]
        async fn read_len_prefix(r: &mut RecvStream) -> io::Result<usize> {
            let mut buf = [0u8; LEN_BYTES];
            r.read_exact(&mut buf).await.map_err(io_err)?;
            let mut v: u64 = 0;
            for b in buf {
                v = (v << 8) | b as u64;
            }
            Ok(v as usize)
        }

        // job queues remain bounded; outer pipeline enforces per-actor sequencing
        let (write_send, mut write_recv) = mpsc::channel::<WriteJob<Bytes>>(1024);
        let (read_send, mut read_recv) = mpsc::channel::<ReadJob<BytesMut>>(1024);

        // Prefetch buffer: keep reading frames to free QUIC conn window.
        // Bound memory with a byte-budget semaphore + small item queue.
        const READ_BUF_BYTES: usize = 8 * 1024 * 1024; // 8 MiB per stream
        const READ_CHAN_CAP: usize = 16; // small item queue; bytes are bounded by semaphore
        let read_byte_budget = Arc::new(Semaphore::new(READ_BUF_BYTES));
        let (frames_tx, mut frames_rx) = mpsc::channel::<BytesMut>(READ_CHAN_CAP);

        // Extract raw streams and spawn lightweight tasks around them
        let Channel {
            read_conn,
            write_conn,
        } = chan;
        let mut read = read_conn.into_inner();
        let mut write = write_conn.into_inner();

        // Reader task: prefetch frames into bounded queue.
        {
            let read_byte_budget = read_byte_budget.clone();
            let mut read = read;
            let frames_tx = frames_tx.clone();
            tokio::spawn(async move {
                loop {
                    let len = match read_len_prefix(&mut read).await {
                        Ok(l) => l,
                        Err(e) => {
                            // Signal end to responder by closing channel
                            // tracing::debug!("reader len err: {e}");
                            break;
                        }
                    };

                    // Acquire byte budget before buffering the frame
                    if len <= READ_BUF_BYTES {
                        // Normal path: bound by semaphore
                        if let Err(_) = read_byte_budget.acquire_many(len as u32).await {
                            break; // semaphore closed; shutdown
                        }
                    }
                    // else: oversize frame; skip budget to avoid deadlock

                    // Read body
                    let mut buf = BytesMut::with_capacity(len);
                    while buf.len() < len {
                        buf.reserve(len - buf.len());
                        match read.read_buf(&mut buf).await {
                            Ok(0) => {
                                tracing::warn!("eof while reading frame body");
                                break;
                            }
                            Ok(_) => {}
                            Err(e) => {
                                tracing::warn!("read body error: {e}");
                                break;
                            }
                        }
                    }
                    if buf.len() != len {
                        // on error, release any taken budget and stop
                        if len <= READ_BUF_BYTES {
                            read_byte_budget.add_permits(len);
                        }
                        break;
                    }

                    // Send to buffer; if receiver dropped, stop
                    if frames_tx.send(buf).await.is_err() {
                        break;
                    }
                }
            });
        }

        // Responder task: match recv jobs to buffered frames, releasing byte permits after delivery
        {
            let read_byte_budget = read_byte_budget.clone();
            tokio::spawn(async move {
                while let Some(job) = read_recv.recv().await {
                    match frames_rx.recv().await {
                        Some(buf) => {
                            let len = buf.len();
                            let _ = job.ret.send(Ok(buf));
                            if len <= READ_BUF_BYTES {
                                read_byte_budget.add_permits(len);
                            }
                        }
                        None => {
                            let _ = job.ret.send(Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                "channel closed",
                            )));
                            break;
                        }
                    }
                }
            });
        }

        // Writer: write length prefix then body in small chunks to promote interleaving
        tokio::spawn(async move {
            while let Some(write_job) = write_recv.recv().await {
                let res = async {
                    let data = write_job.data;
                    write_len_prefix(&mut write, data.len()).await?;
                    let mut off = 0;
                    while off < data.len() {
                        let end = (off + WRITE_CHUNK).min(data.len());
                        write
                            .write_all(&data.slice(off..end))
                            .await
                            .map_err(io_err)?;
                        off = end;
                        // Cooperate to allow other tasks/streams to progress
                        tokio::task::yield_now().await;
                    }
                    Ok::<(), io::Error>(())
                }
                .await;

                match res {
                    Ok(()) => {
                        // Notify caller; ignore if dropped
                        let _ = write_job.ret.send(Ok(()));
                    }
                    Err(err) => {
                        let _ = write_job.ret.send(Err(err));
                        break;
                    }
                }
            }
        });

        ChannelHandle {
            write_job_queue: write_send,
            read_job_queue: read_send,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerOpChannelHandle {
    id: PartyWorkerID,
    conn: Connection,
    codec: LengthDelimitedCodec,
    rt: Handle,
    send_limit: Arc<Semaphore>, // bound concurrent sends
    pub fork_id: u32,
    pub seq: Arc<AtomicU64>,
}

impl PerOpChannelHandle {
    pub fn new(
        id: PartyWorkerID,
        conn_next: Connection,
        codec: LengthDelimitedCodec,
        rt: Handle,
        fork_id: u32,
        per_conn_streams: usize,
    ) -> Self {
        Self {
            id,
            conn: conn_next,
            codec,
            rt,
            send_limit: Arc::new(Semaphore::new(per_conn_streams)),
            fork_id,
            seq: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn fork(&self, fork_id: u32) -> Self {
        Self {
            id: self.id.clone(),
            conn: self.conn.clone(),
            codec: codec_cfg(),
            rt: self.rt.clone(),
            send_limit: self.send_limit.clone(),
            fork_id,
            seq: Arc::new(AtomicU64::new(0)),
        }
    }

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
                    // tracing::warn!("limit reached");
                    let _ = tx.send(err("limit reached"));
                    return;
                }
            };
            // prepend header inside one LDC frame

            let mut buf = BytesMut::with_capacity(payload.len());
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
        let (tx, rx) = oneshot::channel();
        let conn = self.conn.clone();
        let codec = self.codec.clone();

        self.rt.spawn(async move {
            // accept next inbound UNI stream from prev party
            let recv: RecvStream = match conn.accept_uni().await {
                Ok(r) => r,
                Err(e) => {
                    panic!("accept_uni failed")
                }
            };
            let mut fr = FramedRead::new(recv, codec.clone());
            let out = match fr.try_next().await {
                Ok(Some(m)) => Ok(m),
                Ok(None) => panic!("peer closed without frame"),
                Err(e) => Err(e),
            };
            let _ = tx.send(out);
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
