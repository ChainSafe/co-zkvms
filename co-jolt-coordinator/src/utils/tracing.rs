use std::path::Path;

#[cfg(feature = "test-utils")]
pub use jolt_core::utils::tracing::start_rss_monitor;
pub use jolt_core::utils::tracing::TracingGuard;
use tracing::info;
use tracing_chrome::ChromeLayerBuilder;
use tracing_forest::ForestLayer;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::{EnvFilter, Layer};

pub fn init_tracing_bench(file: &str, trace_dir: &Path) -> TracingGuard {
    std::fs::create_dir_all(trace_dir).unwrap();
    let trace_path = trace_dir.join(file);
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy()
        .add_directive("jolt_core=off".parse().unwrap())
        .add_directive("quinn=off".parse().unwrap())
        .add_directive("dory_pcs=off".parse().unwrap());

    let tracy_layer = std::env::var("TRACY").is_ok().then(tracing_tracy::TracyLayer::default);
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
    TracingGuard::new(Some(_guard), file)
}
