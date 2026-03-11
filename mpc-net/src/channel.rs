//! A channel abstraction for sending and receiving messages.
use bytes::{Bytes, BytesMut};
use futures::{Sink, SinkExt, Stream, StreamExt, TryStreamExt};
use quinn::{RecvStream, SendStream};
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
    sync::{mpsc, oneshot, Mutex, OwnedSemaphorePermit, Semaphore},
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
    write_permit: Option<OwnedSemaphorePermit>,
    write_chunk: Option<usize>,
}

struct ReadJob<MRecv> {
    ret: oneshot::Sender<Result<MRecv, io::Error>>,
}

/// A handle to a channel that allows sending and receiving messages.
#[derive(Debug, Clone)]
pub struct ChannelHandle<MSend, MRecv> {
    write_job_queue: mpsc::Sender<WriteJob<MSend>>,
    read_job_queue: mpsc::Sender<ReadJob<MRecv>>,
    write_byte_budget: Option<Arc<Semaphore>>,
    write_byte_budget_max: usize,
    write_len_fn: Option<fn(&MSend) -> usize>,
}

#[derive(Debug, Clone)]
pub struct BulkBytesChannelHandle {
    read: Arc<Mutex<RecvStream>>,
    write: Arc<Mutex<SendStream>>,
    write_byte_budget: Arc<Semaphore>,
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
        let (write_send, mut write_recv) = mpsc::channel::<WriteJob<MSend>>(8192);
        let (read_send, mut read_recv) = mpsc::channel::<ReadJob<MRecv>>(8192);

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
            }
            .instrument(tracing::Span::none()),
        );

        ChannelHandle {
            write_job_queue: write_send,
            read_job_queue: read_send,
            write_byte_budget: None,
            write_byte_budget_max: 0,
            write_len_fn: None,
        }
    }

    fn write_len(&self, data: &MSend) -> Option<usize> {
        self.write_len_fn.map(|f| f(data))
    }

    async fn acquire_write_permit_async(
        &self,
        data: &MSend,
    ) -> io::Result<Option<OwnedSemaphorePermit>> {
        let (Some(semaphore), Some(len)) = (&self.write_byte_budget, self.write_len(data)) else {
            return Ok(None);
        };
        // Skip semaphore for messages larger than the budget capacity —
        // the semaphore can never grant that many permits.
        if len > self.write_byte_budget_max {
            return Ok(None);
        }
        let permits = u32::try_from(len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("message too large for write permit accounting: {len} bytes"),
            )
        })?;
        semaphore
            .clone()
            .acquire_many_owned(permits)
            .await
            .map(Some)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "write byte budget closed"))
    }

    fn acquire_write_permit_blocking(
        &self,
        data: &MSend,
    ) -> io::Result<Option<OwnedSemaphorePermit>> {
        let (Some(semaphore), Some(len)) = (&self.write_byte_budget, self.write_len(data)) else {
            return Ok(None);
        };
        // Skip semaphore for messages larger than the budget capacity —
        // the semaphore can never grant that many permits.
        if len > self.write_byte_budget_max {
            return Ok(None);
        }
        let permits = u32::try_from(len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("message too large for write permit accounting: {len} bytes"),
            )
        })?;

        loop {
            match semaphore.clone().try_acquire_many_owned(permits) {
                Ok(permit) => return Ok(Some(permit)),
                Err(tokio::sync::TryAcquireError::NoPermits) => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(tokio::sync::TryAcquireError::Closed) => {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "write byte budget closed",
                    ));
                }
            }
        }
    }

    /// Instructs the channel to send a message. Returns a [oneshot::Receiver] that will return the result of the send operation.
    pub async fn send(&self, data: MSend) -> oneshot::Receiver<Result<(), io::Error>> {
        let (ret, recv) = oneshot::channel();
        let write_permit = match self.acquire_write_permit_async(&data).await {
            Ok(permit) => permit,
            Err(err) => {
                let _ = ret.send(Err(err));
                return recv;
            }
        };
        let job = WriteJob {
            data,
            ret,
            write_permit,
            write_chunk: None,
        };
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
        let write_permit = match self.acquire_write_permit_blocking(&data) {
            Ok(permit) => permit,
            Err(err) => {
                let _ = ret.send(Err(err));
                return recv;
            }
        };
        let job = WriteJob {
            data,
            ret,
            write_permit,
            write_chunk: None,
        };
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
    fn manage_bytes_quic_impl(
        chan: Channel<RecvStream, SendStream, LengthDelimitedCodec>,
        default_write_chunk: usize,
        prefetch_reads: bool,
    ) -> ChannelHandle<Bytes, BytesMut> {
        const LEN_BYTES: usize = 5;

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

        const WRITE_CHAN_CAP: usize = 16;
        let (write_send, mut write_recv) = mpsc::channel::<WriteJob<Bytes>>(WRITE_CHAN_CAP);
        let (read_send, mut read_recv) = mpsc::channel::<ReadJob<BytesMut>>(8192);
        let write_byte_budget = Arc::new(Semaphore::new(quic_write_buf_bytes()));

        let Channel {
            read_conn,
            write_conn,
        } = chan;
        let mut read = read_conn.into_inner();
        let mut write = write_conn.into_inner();

        if prefetch_reads {
            let read_buf_bytes = quic_read_buf_bytes();
            const READ_CHAN_CAP: usize = 16;
            let read_byte_budget = Arc::new(Semaphore::new(read_buf_bytes));
            let (frames_tx, mut frames_rx) =
                mpsc::channel::<(BytesMut, Option<OwnedSemaphorePermit>)>(READ_CHAN_CAP);

            {
                let read_byte_budget = read_byte_budget.clone();
                let mut read = read;
                let frames_tx = frames_tx.clone();
                tokio::spawn(async move {
                    loop {
                        let len = match read_len_prefix(&mut read).await {
                            Ok(l) => l,
                            Err(_) => break,
                        };

                        let read_permit = if len <= read_buf_bytes {
                            match read_byte_budget
                                .clone()
                                .acquire_many_owned(len as u32)
                                .await
                            {
                                Ok(permit) => Some(permit),
                                Err(_) => break,
                            }
                        } else {
                            None
                        };

                        let mut buf = BytesMut::with_capacity(len);
                        while buf.len() < len {
                            buf.reserve(len - buf.len());
                            match read.read_buf(&mut buf).await {
                                Ok(0) => break,
                                Ok(_) => {}
                                Err(_) => break,
                            }
                        }
                        if buf.len() != len {
                            break;
                        }
                        if frames_tx.send((buf, read_permit)).await.is_err() {
                            break;
                        }
                    }
                });
            }

            tokio::spawn(async move {
                while let Some(job) = read_recv.recv().await {
                    match frames_rx.recv().await {
                        Some((buf, read_permit)) => {
                            let _ = job.ret.send(Ok(buf));
                            drop(read_permit);
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
        } else {
            tokio::spawn(async move {
                while let Some(job) = read_recv.recv().await {
                    let res = async {
                        let len = read_len_prefix(&mut read).await?;
                        let mut buf = BytesMut::with_capacity(len);
                        while buf.len() < len {
                            buf.reserve(len - buf.len());
                            match read.read_buf(&mut buf).await {
                                Ok(0) => {
                                    return Err(io::Error::new(
                                        io::ErrorKind::UnexpectedEof,
                                        "channel closed",
                                    ))
                                }
                                Ok(_) => {}
                                Err(e) => return Err(io_err(e)),
                            }
                        }
                        Ok::<BytesMut, io::Error>(buf)
                    }
                    .await;
                    let _ = job.ret.send(res);
                }
            });
        }

        tokio::spawn(async move {
            while let Some(write_job) = write_recv.recv().await {
                let res = async {
                    let data = write_job.data;
                    write_len_prefix(&mut write, data.len()).await?;
                    let write_chunk = write_job.write_chunk.unwrap_or(default_write_chunk).max(1);
                    let bulk_mode = write_job.write_chunk.is_some();
                    let yield_every = if bulk_mode {
                        write_chunk.saturating_mul(4)
                    } else {
                        write_chunk
                    }
                    .max(write_chunk);
                    let mut off = 0;
                    let mut last_yield = 0usize;
                    while off < data.len() {
                        let end = (off + write_chunk).min(data.len());
                        write
                            .write_all(&data.slice(off..end))
                            .await
                            .map_err(io_err)?;
                        off = end;
                        if off.saturating_sub(last_yield) >= yield_every && off < data.len() {
                            tokio::task::yield_now().await;
                            last_yield = off;
                        }
                    }
                    Ok::<(), io::Error>(())
                }
                .await;

                match res {
                    Ok(()) => {
                        drop(write_job.write_permit);
                        let _ = write_job.ret.send(Ok(()));
                    }
                    Err(err) => {
                        drop(write_job.write_permit);
                        let _ = write_job.ret.send(Err(err));
                        break;
                    }
                }
            }
        });

        ChannelHandle {
            write_job_queue: write_send,
            read_job_queue: read_send,
            write_byte_budget: Some(write_byte_budget),
            write_byte_budget_max: quic_write_buf_bytes(),
            write_len_fn: Some(Bytes::len),
        }
    }

    async fn send_with_chunk(
        &self,
        data: Bytes,
        write_chunk: Option<usize>,
    ) -> oneshot::Receiver<Result<(), io::Error>> {
        let (ret, recv) = oneshot::channel();
        let write_permit = match self.acquire_write_permit_async(&data).await {
            Ok(permit) => permit,
            Err(err) => {
                let _ = ret.send(Err(err));
                return recv;
            }
        };
        let job = WriteJob {
            data,
            ret,
            write_permit,
            write_chunk,
        };
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

    fn blocking_send_with_chunk(
        &self,
        data: Bytes,
        write_chunk: Option<usize>,
    ) -> oneshot::Receiver<Result<(), io::Error>> {
        let (ret, recv) = oneshot::channel();
        let write_permit = match self.acquire_write_permit_blocking(&data) {
            Ok(permit) => permit,
            Err(err) => {
                let _ = ret.send(Err(err));
                return recv;
            }
        };
        let job = WriteJob {
            data,
            ret,
            write_permit,
            write_chunk,
        };
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

    pub async fn send_bulk(&self, data: Bytes) -> oneshot::Receiver<Result<(), io::Error>> {
        self.send_with_chunk(data, Some(quic_bulk_write_chunk_bytes()))
            .await
    }

    pub fn blocking_send_bulk(&self, data: Bytes) -> oneshot::Receiver<Result<(), io::Error>> {
        self.blocking_send_with_chunk(data, Some(quic_bulk_write_chunk_bytes()))
    }

    /// Optimized manager for Bytes over QUIC: chunked length-prefixed I/O without LengthDelimitedCodec buffering.
    /// This keeps message semantics the same for callers while reducing large-frame burstiness.
    pub fn manage_bytes_quic(
        chan: Channel<RecvStream, SendStream, LengthDelimitedCodec>,
    ) -> ChannelHandle<Bytes, BytesMut> {
        Self::manage_bytes_quic_impl(chan, quic_default_write_chunk_bytes(), true)
    }

    pub fn manage_bytes_quic_bulk(
        chan: Channel<RecvStream, SendStream, LengthDelimitedCodec>,
    ) -> ChannelHandle<Bytes, BytesMut> {
        Self::manage_bytes_quic_impl(chan, quic_bulk_write_chunk_bytes(), false)
    }

    /// Create a `ChannelHandle` that bridges a synchronous TLS coordinator
    /// client to the async channel interface used by the QUIC worker.
    ///
    /// Uses the `TlsCoordinatorClient`'s 4-byte length-prefixed send/recv,
    /// matching the `TcpTlsCoordinator`'s wire format.
    ///
    /// A single I/O thread alternates between processing writes and
    /// attempting reads with a short timeout, preventing deadlocks when
    /// the coordinator needs data from the worker before it can respond.
    #[cfg(feature = "tls")]
    pub fn manage_tls_coordinator(
        tls_client: crate::rep3::tls::coordinator::TlsCoordinatorClient,
    ) -> ChannelHandle<Bytes, BytesMut> {
        let (write_send, mut write_recv) = mpsc::channel::<WriteJob<Bytes>>(8192);
        let (read_send, mut read_recv) = mpsc::channel::<ReadJob<BytesMut>>(8192);

        std::thread::Builder::new()
            .name("tls-coord-io".into())
            .spawn(move || {
                let mut client = tls_client;

                // Drain and process all pending write jobs
                let drain_writes =
                    |client: &mut crate::rep3::tls::coordinator::TlsCoordinatorClient,
                     write_recv: &mut mpsc::Receiver<WriteJob<Bytes>>| {
                        while let Ok(wj) = write_recv.try_recv() {
                            let result = client.send(&wj.data);
                            let _ = wj.ret.send(result);
                        }
                    };

                loop {
                    // Process any pending writes first
                    drain_writes(&mut client, &mut write_recv);

                    // Check for a read job (non-blocking)
                    match read_recv.try_recv() {
                        Ok(rj) => {
                            // Wait for data, processing writes between attempts
                            loop {
                                match client.try_recv(Duration::from_millis(5)) {
                                    Ok(Some(data)) => {
                                        let _ = rj.ret.send(Ok(BytesMut::from(&data[..])));
                                        break;
                                    }
                                    Ok(None) => {
                                        // Timeout — process writes then retry
                                        drain_writes(&mut client, &mut write_recv);
                                    }
                                    Err(e) => {
                                        let _ = rj.ret.send(Err(e));
                                        break;
                                    }
                                }
                            }
                        }
                        Err(mpsc::error::TryRecvError::Empty) => {
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(mpsc::error::TryRecvError::Disconnected) => break,
                    }
                }
            })
            .expect("spawn tls-coord-io");

        ChannelHandle {
            write_job_queue: write_send,
            read_job_queue: read_send,
            write_byte_budget: None,
            write_byte_budget_max: 0,
            write_len_fn: None,
        }
    }
}

impl BulkBytesChannelHandle {
    pub fn manage_quic(chan: Channel<RecvStream, SendStream, LengthDelimitedCodec>) -> Self {
        let Channel {
            read_conn,
            write_conn,
        } = chan;
        Self {
            read: Arc::new(Mutex::new(read_conn.into_inner())),
            write: Arc::new(Mutex::new(write_conn.into_inner())),
            write_byte_budget: Arc::new(Semaphore::new(quic_write_buf_bytes())),
        }
    }

    async fn send_inner(&self, data: Bytes) -> io::Result<()> {
        const LEN_BYTES: usize = 5;

        #[inline]
        fn write_len_prefix_buf(len: usize) -> [u8; LEN_BYTES] {
            let v = len as u64;
            let mut buf = [0u8; LEN_BYTES];
            for i in (0..LEN_BYTES).rev() {
                buf[i] = (v >> (8 * (LEN_BYTES - 1 - i)) & 0xFF) as u8;
            }
            buf
        }

        let write_permit = if data.len() <= quic_write_buf_bytes() {
            Some(
                self.write_byte_budget
                    .clone()
                    .acquire_many_owned(data.len() as u32)
                    .await
                    .map_err(|_| {
                        io::Error::new(io::ErrorKind::BrokenPipe, "bulk write budget closed")
                    })?,
            )
        } else {
            None
        };

        let mut write = self.write.lock().await;
        write
            .write_all(&write_len_prefix_buf(data.len()))
            .await
            .map_err(io_err)?;
        let write_chunk = quic_bulk_write_chunk_bytes().max(1);
        let mut off = 0usize;
        let mut last_yield = 0usize;
        let yield_every = write_chunk.saturating_mul(4).max(write_chunk);
        while off < data.len() {
            let end = (off + write_chunk).min(data.len());
            write
                .write_all(&data.slice(off..end))
                .await
                .map_err(io_err)?;
            off = end;
            if off.saturating_sub(last_yield) >= yield_every && off < data.len() {
                tokio::task::yield_now().await;
                last_yield = off;
            }
        }
        drop(write);
        drop(write_permit);
        Ok(())
    }

    async fn recv_into_inner(&self, dst: &mut [u8]) -> io::Result<()> {
        const LEN_BYTES: usize = 5;

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

        let mut read = self.read.lock().await;
        let len = read_len_prefix(&mut read).await?;
        if len != dst.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "bulk frame size mismatch: expected {} bytes, got {}",
                    dst.len(),
                    len
                ),
            ));
        }
        read.read_exact(dst).await.map_err(io_err)
    }

    pub async fn send(&self, data: Bytes) -> io::Result<()> {
        self.send_inner(data).await
    }

    pub async fn recv_bytes(&self) -> io::Result<Vec<u8>> {
        const LEN_BYTES: usize = 5;

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

        let mut read = self.read.lock().await;
        let len = read_len_prefix(&mut read).await?;
        let mut buf = vec![0u8; len];
        read.read_exact(&mut buf).await.map_err(io_err)?;
        Ok(buf)
    }

    pub async fn recv_into(&self, dst: &mut [u8]) -> io::Result<()> {
        self.recv_into_inner(dst).await
    }
}

fn parse_quic_buf_mb(var: &str) -> Option<usize> {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .map(|v| v.saturating_mul(1024 * 1024))
}

fn quic_write_buf_bytes() -> usize {
    const DEFAULT_MB: usize = 64;
    parse_quic_buf_mb("MPC_QUIC_WRITE_BUF_MB")
        .or_else(|| parse_quic_buf_mb("MPC_WORKER_WRITE_BUF_MB"))
        .unwrap_or(DEFAULT_MB * 1024 * 1024)
}

fn quic_read_buf_bytes() -> usize {
    const DEFAULT_MB: usize = 64;
    parse_quic_buf_mb("MPC_QUIC_READ_BUF_MB").unwrap_or(DEFAULT_MB * 1024 * 1024)
}

fn parse_quic_chunk_kb(var: &str, default_kb: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_kb)
        .saturating_mul(1024)
}

fn quic_default_write_chunk_bytes() -> usize {
    parse_quic_chunk_kb("MPC_QUIC_WRITE_CHUNK_KB", 16)
}

fn quic_bulk_write_chunk_bytes() -> usize {
    parse_quic_chunk_kb("MPC_QUIC_BULK_WRITE_CHUNK_KB", 1024)
}

#[inline]
fn err(msg: &str) -> Result<(), io::Error> {
    Err(io::Error::new(io::ErrorKind::Other, msg))
}

#[inline]
fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(std::io::ErrorKind::Other, e.to_string())
}
