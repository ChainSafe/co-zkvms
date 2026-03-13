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

impl TracingGuard {
    pub fn new(guard: Option<tracing_chrome::FlushGuard>, file: impl Into<String>) -> Self {
        Self {
            _guard: guard,
            file: file.into(),
        }
    }
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        if let Some(ref file) = Some(&self.file) {
            info!("tracing_chrome flushing to {file}");
        }
    }
}

pub fn init_tracing_bench(file: &str, trace_dir: &Path) -> TracingGuard {
    std::fs::create_dir_all(trace_dir).unwrap();
    let trace_path = trace_dir.join(file);
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy()
        .add_directive("jolt_core=off".parse().unwrap())
        .add_directive("quinn=off".parse().unwrap())
        .add_directive("dory=off".parse().unwrap());

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

pub fn sanitize_trace_label(label: &str) -> String {
    let mut sanitized = String::with_capacity(label.len());
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            sanitized.push(ch);
        } else {
            sanitized.push('-');
        }
    }
    sanitized.trim_matches('-').to_string()
}

pub fn worker_trace_file(my_id: usize, program_id: &str) -> String {
    let program_id = sanitize_trace_label(program_id);
    format!("trace_party-{my_id}_{program_id}_{}CPU.json", num_cpus::get())
}

pub fn coordinator_trace_file(program_id: &str) -> String {
    let program_id = sanitize_trace_label(program_id);
    format!("trace_coordinator_{program_id}_{}CPU.json", num_cpus::get())
}

#[cfg(feature = "test-utils")]
pub fn start_rss_monitor(interval: std::time::Duration) {
    use tracy_client::{plot_name, Client, PlotConfiguration, PlotFormat, PlotLineStyle};

    static RSS_PLOT: tracy_client::PlotName = plot_name!("RSS");
    let client = Client::running().expect("Tracy client must be running");
    client.plot_config(
        RSS_PLOT,
        PlotConfiguration::default()
            .format(PlotFormat::Memory)
            .line_style(PlotLineStyle::Smooth)
            .fill(true)
            .color(Some(0xFF6600)),
    );

    std::thread::Builder::new()
        .name("rss-monitor".into())
        .spawn(move || loop {
            let rss = get_rss_bytes();
            if let Some(c) = Client::running() {
                c.plot(RSS_PLOT, rss as f64);
            }
            std::thread::sleep(interval);
        })
        .expect("spawn rss-monitor thread");
}

#[cfg(all(feature = "test-utils", target_os = "macos"))]
fn get_rss_bytes() -> u64 {
    unsafe {
        let mut info: libc::mach_task_basic_info_data_t = std::mem::zeroed();
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
        let ret = libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut i32,
            &mut count,
        );
        if ret == 0 {
            info.resident_size
        } else {
            0
        }
    }
}

#[cfg(all(feature = "test-utils", target_os = "linux"))]
fn get_rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|contents| contents.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map(|pages| pages * 4096)
        .unwrap_or(0)
}

#[cfg(all(feature = "test-utils", not(any(target_os = "linux", target_os = "macos"))))]
fn get_rss_bytes() -> u64 {
    0
}
