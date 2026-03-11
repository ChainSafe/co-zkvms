use std::time::Duration;

#[cfg(feature = "jemalloc-stats")]
mod jemalloc {
    use super::Duration;

    use tikv_jemalloc_ctl::{arenas, epoch, raw, stats};
    use tracy_client::{plot_name, Client, PlotConfiguration, PlotFormat, PlotLineStyle};

    static P_ALLOCATED: tracy_client::PlotName = plot_name!("jemalloc.allocated");
    static P_ACTIVE: tracy_client::PlotName = plot_name!("jemalloc.active");
    static P_RESIDENT: tracy_client::PlotName = plot_name!("jemalloc.resident");
    static P_RETAINED: tracy_client::PlotName = plot_name!("jemalloc.retained");
    static P_MAPPED: tracy_client::PlotName = plot_name!("jemalloc.mapped");
    static P_METADATA: tracy_client::PlotName = plot_name!("jemalloc.metadata");

    fn config_plot() -> PlotConfiguration {
        PlotConfiguration::default().format(PlotFormat::Memory).line_style(PlotLineStyle::Smooth).fill(false)
    }

    pub(super) fn start_monitor(interval: Duration) {
        let Some(client) = Client::running() else {
            return;
        };

        client.plot_config(P_ALLOCATED, config_plot());
        client.plot_config(P_ACTIVE, config_plot());
        client.plot_config(P_RESIDENT, config_plot());
        client.plot_config(P_RETAINED, config_plot());
        client.plot_config(P_MAPPED, config_plot());
        client.plot_config(P_METADATA, config_plot());

        std::thread::Builder::new()
            .name("jemalloc-monitor".into())
            .spawn(move || loop {
                let _ = epoch::advance();
                let allocated = stats::allocated::read().unwrap_or(0);
                let active = stats::active::read().unwrap_or(0);
                let resident = stats::resident::read().unwrap_or(0);
                let retained = stats::retained::read().unwrap_or(0);
                let mapped = stats::mapped::read().unwrap_or(0);
                let metadata = stats::metadata::read().unwrap_or(0);

                if let Some(c) = Client::running() {
                    c.plot(P_ALLOCATED, allocated as f64);
                    c.plot(P_ACTIVE, active as f64);
                    c.plot(P_RESIDENT, resident as f64);
                    c.plot(P_RETAINED, retained as f64);
                    c.plot(P_MAPPED, mapped as f64);
                    c.plot(P_METADATA, metadata as f64);
                }
                std::thread::sleep(interval);
            })
            .expect("spawn jemalloc-monitor thread");
    }

    pub(super) fn purge_all_arenas() {
        let Ok(narenas) = arenas::narenas::read() else {
            return;
        };
        for arena in 0..narenas {
            let name = format!("arena.{arena}.purge");
            unsafe {
                let _ = raw::write(name.as_bytes(), ());
            }
        }
    }
}

/// Spawn a background thread that samples jemalloc heap stats and emits Tracy plots:
/// `jemalloc.allocated`, `jemalloc.active`, `jemalloc.resident`, `jemalloc.retained`.
///
/// No-op unless the `jemalloc-stats` feature is enabled and the Tracy client is running.
pub fn start_jemalloc_monitor(interval: Duration) {
    #[cfg(feature = "jemalloc-stats")]
    jemalloc::start_monitor(interval);
    #[cfg(not(feature = "jemalloc-stats"))]
    let _ = interval;
}

/// If `JEMALLOC_PURGE=1` is set, purge all jemalloc arenas.
///
/// This is intended as a diagnostic to distinguish:
/// - live heap growth (allocated/active stay high) vs
/// - allocator retention (allocated/active drop but RSS stays high).
pub fn maybe_purge_jemalloc() {
    let enabled = matches!(std::env::var("JEMALLOC_PURGE").as_deref(), Ok("1"));
    if !enabled {
        return;
    }

    #[cfg(feature = "jemalloc-stats")]
    jemalloc::purge_all_arenas();
}
