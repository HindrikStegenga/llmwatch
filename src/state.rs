use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::prom::HistStats;

#[derive(Debug, Clone, Copy)]
pub struct Rates {
    pub prompt_tps: f64,
    pub gen_tps: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SpecMetrics {
    pub draft: Option<f64>,
    pub accepted: Option<f64>,
    pub emitted: Option<f64>,
    pub drafts: Option<f64>,
    pub counters: bool,
    pub accept_rate: Option<f64>,
    pub efficiency: Option<f64>,
    pub accept_pos: Option<HistStats>,
    pub per_pos: Vec<(u32, f64)>,
}

impl SpecMetrics {
    pub fn present(&self) -> bool {
        self.draft.is_some()
            || self.accepted.is_some()
            || self.emitted.is_some()
            || self.accept_rate.is_some()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpecRates {
    pub draft_tps: f64,
    pub accepted_tps: f64,
    pub emitted_tps: f64,
}

#[derive(Debug, Clone)]
pub struct VllmMetrics {
    pub prompt_tokens_total: f64,
    pub generation_tokens_total: f64,
    pub process_start_time: Option<f64>,
    pub prompt_compute_total: Option<f64>,
    pub running: f64,
    pub waiting: f64,
    pub wait_capacity: Option<f64>,
    pub wait_deferred: Option<f64>,
    pub swapped: Option<f64>,
    pub gpu_cache_usage: Option<f64>,
    pub cpu_cache_usage: Option<f64>,
    pub prefix_cache_hit: Option<f64>,
    pub preemptions: Option<f64>,
    pub ttft: Option<HistStats>,
    pub tpot: Option<HistStats>,
    pub e2e: Option<HistStats>,
    pub prompt_len: Option<HistStats>,
    pub gen_len: Option<HistStats>,
    pub batch: Option<HistStats>,
    pub prefill_time: Option<HistStats>,
    pub decode_time: Option<HistStats>,
    pub max_gen_tokens: Option<HistStats>,
    pub tool_calls_total: Option<f64>,
    pub tool_no_call_total: Option<f64>,
    pub http_latency: Option<HistStats>,
    pub http_requests_total: Option<f64>,
    pub http_errors_total: Option<f64>,
    pub process_cpu_total: Option<f64>,
    pub process_rss_bytes: Option<f64>,
    pub kv_pool_tokens: Option<f64>,
    pub gpu_mem_util: Option<f64>,
    pub spec: SpecMetrics,
}

#[derive(Default)]
pub struct VllmState {
    pub last_ok: Option<Instant>,
    pub last_err: Option<String>,
    pub metrics: Option<VllmMetrics>,
    pub rates: Option<Rates>,
    pub prompt_hist: Vec<u64>,
    pub gen_hist: Vec<u64>,
    pub all_names: Vec<String>,
    pub hist_t: Vec<f64>,
    pub hist_prompt: Vec<f64>,
    pub hist_gen: Vec<f64>,
    pub hist_compute: Vec<f64>,
    pub(crate) hist_start: Option<Instant>,
    pub(crate) prev: Option<(Instant, f64, f64)>,
    pub spec_rates: Option<SpecRates>,
    pub(crate) spec_prev: Option<(Instant, f64, f64, f64)>,
    pub req_rate: Option<f64>,
    pub(crate) req_prev: Option<(Instant, f64)>,
    pub tool_rate: Option<f64>,
    pub(crate) tool_prev: Option<(Instant, f64)>,
    pub cpu_pct: Option<f64>,
    pub(crate) cpu_prev: Option<(Instant, f64)>,
    pub compute_tps: Option<f64>,
    pub(crate) compute_prev: Option<(Instant, f64)>,
    pub(crate) engine_start: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub temp: Option<f64>,
    pub mem_temp: Option<f64>,
    pub mem_used: Option<f64>,
    pub mem_total: Option<f64>,
    pub util: Option<f64>,
    pub power: Option<f64>,
    pub power_limit: Option<f64>,
    pub pcie_gen: Option<f64>,
    pub pcie_width: Option<f64>,
    pub pcie_rx: Option<f64>,
    pub pcie_tx: Option<f64>,
    pub p2p: Option<String>,
    pub nvlink: Option<String>,
}

#[derive(Default)]
pub struct GpuState {
    pub last_ok: Option<Instant>,
    pub last_err: Option<String>,
    pub gpus: Vec<GpuInfo>,
}

pub struct AppState {
    pub vllm: Mutex<VllmState>,
    pub gpu: Mutex<GpuState>,
    pub vllm_url: String,
    pub ssh_host: Option<String>,
    pub paused: AtomicBool,
    pub show_metrics: AtomicBool,
    pub metrics_scroll: AtomicUsize,
    interval: Mutex<f64>,
    pub refresh: tokio::sync::Notify,
}

impl AppState {
    pub fn new(interval: f64) -> Self {
        Self {
            vllm: Mutex::new(VllmState::default()),
            gpu: Mutex::new(GpuState::default()),
            vllm_url: String::new(),
            ssh_host: None,
            paused: AtomicBool::new(false),
            show_metrics: AtomicBool::new(false),
            metrics_scroll: AtomicUsize::new(0),
            interval: Mutex::new(interval.clamp(0.5, 30.0)),
            refresh: tokio::sync::Notify::new(),
        }
    }

    pub fn interval_secs(&self) -> f64 {
        *self.interval.lock().unwrap()
    }

    pub fn bump_interval(&self, delta: f64) {
        let mut i = self.interval.lock().unwrap();
        *i = (*i + delta).clamp(0.5, 30.0);
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    pub fn toggle_paused(&self) {
        self.paused.store(!self.is_paused(), Ordering::SeqCst);
    }

    pub fn toggle_metrics_popup(&self) {
        let show = !self.metrics_popup_visible();
        if show {
            self.metrics_scroll.store(0, Ordering::SeqCst);
        }
        self.show_metrics.store(show, Ordering::SeqCst);
    }

    pub fn metrics_popup_visible(&self) -> bool {
        self.show_metrics.load(Ordering::SeqCst)
    }

    pub fn scroll_metrics(&self, delta: isize) {
        if !self.metrics_popup_visible() {
            return;
        }
        let cur = self.metrics_scroll.load(Ordering::SeqCst) as isize;
        let next = (cur + delta).max(0) as usize;
        self.metrics_scroll.store(next, Ordering::SeqCst);
    }
}
