
use tracing::info;

pub struct TracingGuard {
    _guard: Option<tracing_chrome::FlushGuard>,
    file: String,
}

impl TracingGuard {
    pub fn new(guard: Option<tracing_chrome::FlushGuard>, file: impl Into<String>) -> Self {
        Self { _guard: guard, file: file.into() }
    }
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        if let Some(ref file) = Some(&self.file) {
            info!("tracing_chrome flushing to {file}");
        }
    }
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
