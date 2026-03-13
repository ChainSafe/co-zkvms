use std::path::Path;

pub use jolt_core::utils::tracing::{
    coordinator_trace_file, init_tracing_bench, sanitize_trace_label, worker_trace_file, TracingGuard,
};
#[cfg(feature = "test-utils")]
pub use jolt_core::utils::tracing::start_rss_monitor;
use tracing::info;
use tracing_chrome::ChromeLayerBuilder;
use tracing_forest::ForestLayer;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::{EnvFilter, Layer};

pub fn init_tracing(file: &str, trace_dir: &Path) -> Option<TracingGuard> {
    std::fs::create_dir_all(trace_dir).unwrap();
    let trace_path = trace_dir.join(file);
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy()
        .add_directive("jolt_core=off".parse().unwrap())
        .add_directive("co_jolt2=info".parse().unwrap())
        .add_directive("mpc_net=info".parse().unwrap())
        .add_directive("quinn=off".parse().unwrap())
        .add_directive("dory=off".parse().unwrap());

    let current_level = env_filter.max_level_hint().unwrap_or(LevelFilter::INFO);
    let subscriber = Registry::default().with(env_filter);

    if current_level == LevelFilter::TRACE {
        let (chrome_layer, _guard) = ChromeLayerBuilder::new().file(trace_path).build();
        let _ = tracing::subscriber::set_global_default(
            subscriber.with(chrome_layer).with(ForestLayer::default().with_filter(LevelFilter::TRACE)),
        );
        info!("tracing_chrome writes to file: {}", file);
        Some(TracingGuard::new(Some(_guard), file))
    } else {
        let _ = tracing::subscriber::set_global_default(subscriber.with(ForestLayer::default()));
        None
    }
}
