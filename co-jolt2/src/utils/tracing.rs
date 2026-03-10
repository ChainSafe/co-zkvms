use std::path::Path;

use tracing::info;
use tracing_chrome::ChromeLayerBuilder;
use tracing_forest::ForestLayer;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::{EnvFilter, Layer};

pub struct TracingGuard {
    _guard: Option<tracing_chrome::FlushGuard>,
    file: String,
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        if let Some(ref file) = Some(&self.file) {
            info!("tracing_chrome flushing to {file}");
        }
    }
}

/// Initialize tracing for benchmarks: always produces a chrome trace file,
/// with console output at the level specified by RUST_LOG (default INFO).
pub fn init_tracing_bench(file: &str, trace_dir: &Path) -> TracingGuard {
    std::fs::create_dir_all(trace_dir).unwrap();
    let trace_path = trace_dir.join(file);
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy()
        .add_directive("jolt_core=off".parse().unwrap())
        .add_directive("quinn=off".parse().unwrap())
        .add_directive("dory=off".parse().unwrap());

    // Only enable Tracy layer when TRACY env var is set, so only the target
    // process starts the Tracy client and binds port 8086.
    let tracy_layer = std::env::var("TRACY")
        .is_ok()
        .then(tracing_tracy::TracyLayer::default);
    let (chrome_layer, _guard) = ChromeLayerBuilder::new().file(trace_path).build();
    if tracing::subscriber::set_global_default(
        Registry::default()
            .with(env_filter)
            .with(chrome_layer)
            .with(tracy_layer)
            .with(ForestLayer::default().with_filter(LevelFilter::INFO)),
    )
    .is_err()
    {}
    info!("tracing_chrome writes to file: {}", file);
    TracingGuard {
        _guard: Some(_guard),
        file: file.to_string(),
    }
}

pub fn init_tracing(file: &str, trace_dir: &Path) -> Option<TracingGuard> {
    std::fs::create_dir_all(trace_dir).unwrap();
    let trace_path = trace_dir.join(file);
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy()
        .add_directive("jolt_core=off".parse().unwrap())
        .add_directive("co_jolt2=info".parse().unwrap())
        .add_directive("mpc_net=info".parse().unwrap())
        .add_directive("quinn=off".parse().unwrap());

    let current_level = env_filter.max_level_hint().unwrap_or(LevelFilter::INFO);
    let subscriber = Registry::default().with(env_filter);

    if current_level == LevelFilter::TRACE {
        let (chrome_layer, _guard) = ChromeLayerBuilder::new().file(trace_path).build();
        let _ = tracing::subscriber::set_global_default(
            subscriber
                .with(chrome_layer)
                .with(ForestLayer::default().with_filter(LevelFilter::TRACE)),
        );
        info!("tracing_chrome writes to file: {}", file);
        Some(TracingGuard {
            _guard: Some(_guard),
            file: file.to_string(),
        })
    } else {
        let _ = tracing::subscriber::set_global_default(subscriber.with(ForestLayer::default()));
        None
    }
}
