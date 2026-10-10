use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::prom::{self, HistStats};
use crate::state::{AppState, Rates, SpecMetrics, SpecRates, VllmMetrics};

/// Generation throughput below which token flow is considered a starved
/// trickle — a concurrent decode squeezed in between chunks of a big prefill —
/// rather than real progress. A trickle must not keep the prefill badge quiet.
const GEN_PROGRESS_FLOOR: f64 = 20.0;

fn is_progress(prompt_tps: f64, gen_tps: f64) -> bool {
    prompt_tps > 0.5 || gen_tps > GEN_PROGRESS_FLOOR
}

pub async fn fetch_loop(state: Arc<AppState>, client: reqwest::Client, url: String) {
    loop {
        if !state.is_paused() {
            match fetch(&client, &url).await {
                Ok((metrics, names)) => update(&state, metrics, names),
                Err(err) => {
                    let mut vs = state.vllm.lock().unwrap();
                    vs.last_err = Some(err.to_string());
                }
            }
        }
        let nap = Duration::from_secs_f64(state.interval_secs());
        tokio::select! {
            _ = tokio::time::sleep(nap) => {}
            _ = state.refresh.notified() => {}
        }
    }
}

fn update(state: &AppState, metrics: VllmMetrics, names: Vec<String>) {
    let mut vs = state.vllm.lock().unwrap();
    vs.all_names = names;
    let now = Instant::now();
    let window = state.window_secs();
    if let Some(start) = metrics.process_start_time
        && vs.engine_start != Some(start)
    {
        vs.engine_start = Some(start);
        vs.win.clear();
        vs.win_compute.clear();
        vs.last_progress = None;
        vs.spec_prev = None;
        vs.req_prev = None;
        vs.tool_prev = None;
        vs.cpu_prev = None;
    }
    if vs.last_progress.is_none() {
        vs.last_progress = Some(now);
    }
    // A request starting after an idle stretch would otherwise inherit the
    // idle time as "prefill" — restart the clock when the engine goes busy.
    if metrics.running > 0.0 && vs.metrics.as_ref().is_none_or(|m| m.running <= 0.0) {
        vs.last_progress = Some(now);
    }
    // The engine's token counters are lump-sum (updated once per completed
    // request, not per chunk), so per-poll deltas spike to the full prompt
    // size. Rate everything over a sliding window instead.
    let mut compute_rate: Option<f64> = None;
    if let Some(pc) = metrics.prompt_compute_total {
        vs.win_compute.push_back((now, pc));
        prune_window(&mut vs.win_compute, window, now, |s| s.0);
        if let (Some(&(t0, c0)), Some(&(t1, c1))) =
            (vs.win_compute.front(), vs.win_compute.back())
            && let Some(r) = delta_rate(t0, c0, t1, c1)
        {
            compute_rate = Some(r);
            vs.compute_tps = compute_rate;
        }
    }
    let compute_push = if metrics.prompt_compute_total.is_some() {
        Some(compute_rate.unwrap_or(0.0))
    } else {
        None
    };
    vs.win.push_back((
        now,
        metrics.prompt_tokens_total,
        metrics.generation_tokens_total,
    ));
    prune_window(&mut vs.win, window, now, |s| s.0);
    if let (Some(&(t0, p0, g0)), Some(&(t1, p1, g1))) = (vs.win.front(), vs.win.back())
        && let (Some(pr), Some(gr)) = (delta_rate(t0, p0, t1, p1), delta_rate(t0, g0, t1, g1))
    {
        let rates = Rates {
            prompt_tps: pr,
            gen_tps: gr,
        };
        vs.rates = Some(rates);
        push(&mut vs.prompt_hist, rates.prompt_tps);
        push(&mut vs.gen_hist, rates.gen_tps);
        push_history(&mut vs, now, rates, compute_push);
            if is_progress(rates.prompt_tps, rates.gen_tps) {
                vs.last_progress = Some(now);
            }
    }
    if metrics.spec.counters
        && let (Some(d), Some(a)) = (metrics.spec.draft, metrics.spec.accepted)
    {
        let e = metrics.spec.emitted;
        if let Some((t0, d0, a0, e0)) = vs.spec_prev {
            let dt = now.duration_since(t0).as_secs_f64();
            if dt > 0.05 {
                vs.spec_rates = Some(SpecRates {
                    draft_tps: (d - d0).max(0.0) / dt,
                    accepted_tps: (a - a0).max(0.0) / dt,
                    emitted_tps: e.map_or(0.0, |e| (e - e0).max(0.0) / dt),
                });
            }
        }
        vs.spec_prev = Some((now, d, a, e.unwrap_or(0.0)));
    }
    if let Some(total) = metrics.http_requests_total {
        if let Some((t0, r0)) = vs.req_prev {
            let dt = now.duration_since(t0).as_secs_f64();
            if dt > 0.05 {
                vs.req_rate = Some((total - r0).max(0.0) / dt);
            }
        }
        vs.req_prev = Some((now, total));
    }
    if let Some(tc) = metrics.tool_calls_total {
        if let Some((t0, c0)) = vs.tool_prev {
            let dt = now.duration_since(t0).as_secs_f64();
            if dt > 0.05 {
                vs.tool_rate = Some((tc - c0).max(0.0) / dt);
            }
        }
        vs.tool_prev = Some((now, tc));
    }
    if let Some(cpu) = metrics.process_cpu_total {
        if let Some((t0, c0)) = vs.cpu_prev {
            let dt = now.duration_since(t0).as_secs_f64();
            if dt > 0.05 {
                vs.cpu_pct = Some(((cpu - c0).max(0.0) / dt * 100.0).min(100.0));
            }
        }
        vs.cpu_prev = Some((now, cpu));
    }
    vs.metrics = Some(metrics);
    vs.last_ok = Some(now);
    vs.last_err = None;
}

fn push_history(
    vs: &mut crate::state::VllmState,
    now: Instant,
    rates: Rates,
    compute: Option<f64>,
) {
    let t0 = *vs.hist_start.get_or_insert(now);
    let t = now.duration_since(t0).as_secs_f64();
    vs.hist_t.push(t);
    vs.hist_prompt.push(rates.prompt_tps);
    vs.hist_gen.push(rates.gen_tps);
    if let Some(c) = compute {
        vs.hist_compute.push(c);
    }
    if vs.hist_t.len() > 1300 {
        vs.hist_t.remove(0);
        vs.hist_prompt.remove(0);
        vs.hist_gen.remove(0);
        if vs.hist_compute.len() > vs.hist_t.len() {
            vs.hist_compute.remove(0);
        }
    }
}

fn push(hist: &mut Vec<u64>, value: f64) {
    hist.push(value.round().max(0.0) as u64);
    if hist.len() > 240 {
        hist.remove(0);
    }
}

/// Drop samples older than `window` secs, always keeping the newest two so a
/// rate can still be formed when the poll interval exceeds the window.
fn prune_window<T>(
    win: &mut VecDeque<T>,
    window: f64,
    now: Instant,
    ts: impl Fn(&T) -> Instant,
) {
    while win.len() > 2
        && now
            .duration_since(ts(win.front().expect("non-empty")))
            .as_secs_f64()
            > window
    {
        win.pop_front();
    }
}

fn delta_rate(t0: Instant, v0: f64, t1: Instant, v1: f64) -> Option<f64> {
    let dt = t1.duration_since(t0).as_secs_f64();
    (dt > 0.05).then_some((v1 - v0).max(0.0) / dt)
}

async fn fetch(
    client: &reqwest::Client,
    url: &str,
) -> anyhow::Result<(VllmMetrics, Vec<String>)> {
    let body = client
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let samples = prom::parse(&body);
    let mut names: Vec<String> = samples.iter().map(|s| s.name.clone()).collect();
    names.sort_unstable();
    names.dedup();
    Ok((extract_from(&samples), names))
}

#[cfg(test)]
pub fn extract(text: &str) -> VllmMetrics {
    extract_from(&prom::parse(text))
}

fn extract_from(samples: &[prom::Sample]) -> VllmMetrics {
    let mut http_total: Option<f64> = None;
    let mut http_errors: Option<f64> = None;
    for s in samples.iter().filter(|s| s.name == "http_requests_total") {
        *http_total.get_or_insert(0.0) += s.value;
        if !s.label("status").unwrap_or("").starts_with('2') {
            *http_errors.get_or_insert(0.0) += s.value;
        }
    }
    let mut tool_calls: Option<f64> = None;
    let mut tool_no_call: Option<f64> = None;
    for s in samples
        .iter()
        .filter(|s| s.name == "vllm_tool_call_parser_invocations_total")
    {
        match s.label("outcome") {
            Some("tool_call") => *tool_calls.get_or_insert(0.0) += s.value,
            Some("no_tool_call") => *tool_no_call.get_or_insert(0.0) += s.value,
            _ => {}
        }
    }
    let mut wait_capacity: Option<f64> = None;
    let mut wait_deferred: Option<f64> = None;
    for s in samples
        .iter()
        .filter(|s| s.name == "vllm_num_requests_waiting_by_reason")
    {
        match s.label("reason") {
            Some("capacity") => *wait_capacity.get_or_insert(0.0) += s.value,
            Some("deferred") => *wait_deferred.get_or_insert(0.0) += s.value,
            _ => {}
        }
    }
    let mut kv_pool_tokens: Option<f64> = None;
    let mut gpu_mem_util: Option<f64> = None;
    for s in samples
        .iter()
        .filter(|s| s.name == "vllm_cache_config_info")
    {
        kv_pool_tokens = s
            .label("kv_cache_size_tokens")
            .and_then(|v| v.parse::<f64>().ok());
        gpu_mem_util = s
            .label("gpu_memory_utilization")
            .and_then(|v| v.parse::<f64>().ok());
    }
    VllmMetrics {
        prompt_tokens_total: scalar(samples, "vllm_prompt_tokens_total").unwrap_or(0.0),
        generation_tokens_total: scalar(samples, "vllm_generation_tokens_total").unwrap_or(0.0),
        process_start_time: scalar(samples, "process_start_time_seconds"),
        prompt_compute_total: samples
            .iter()
            .find(|s| {
                s.name == "vllm_prompt_tokens_by_source_total"
                    && s.label("source") == Some("local_compute")
            })
            .map(|s| s.value),
        running: scalar(samples, "vllm_num_requests_running").unwrap_or(0.0),
        waiting: scalar(samples, "vllm_num_requests_waiting").unwrap_or(0.0),
        wait_capacity,
        wait_deferred,
        swapped: scalar(samples, "vllm_num_requests_swapped"),
        gpu_cache_usage: scalar(samples, "vllm_gpu_cache_usage_perc")
            .or_else(|| scalar(samples, "vllm_kv_cache_usage_perc")),
        cpu_cache_usage: scalar(samples, "vllm_cpu_cache_usage_perc"),
        prefix_cache_hit: prefix_cache_hit(samples),
        preemptions: scalar(samples, "vllm_num_preemptions_total"),
        ttft: histogram(samples, "vllm_time_to_first_token_seconds"),
        tpot: histogram(samples, "vllm_time_per_output_token_seconds")
            .or_else(|| histogram(samples, "vllm_inter_token_latency_seconds")),
        e2e: histogram(samples, "vllm_e2e_request_latency_seconds"),
        prompt_len: histogram(samples, "vllm_request_prompt_tokens"),
        gen_len: histogram(samples, "vllm_request_generation_tokens"),
        batch: histogram(samples, "vllm_iteration_tokens_total"),
        prefill_time: histogram(samples, "vllm_request_prefill_time_seconds"),
        decode_time: histogram(samples, "vllm_request_decode_time_seconds"),
        max_gen_tokens: histogram(samples, "vllm_request_max_num_generation_tokens"),
        tool_calls_total: tool_calls,
        tool_no_call_total: tool_no_call,
        http_latency: histogram(samples, "http_request_duration_highr_seconds"),
        http_requests_total: http_total,
        http_errors_total: http_errors,
        process_cpu_total: scalar(samples, "process_cpu_seconds_total"),
        process_rss_bytes: scalar(samples, "process_resident_memory_bytes"),
        kv_pool_tokens,
        gpu_mem_util,
        spec: extract_spec(samples),
    }
}

fn extract_spec(samples: &[prom::Sample]) -> SpecMetrics {
    let find = |suffix: &str| {
        [
            format!("vllm_spec_decode_{suffix}_total"),
            format!("vllm_spec_decode_{suffix}"),
            format!("spec_decode_{suffix}_total"),
            format!("spec_decode_{suffix}"),
        ]
        .into_iter()
        .find_map(|n| scalar(samples, &n))
    };
    let is_counter = |suffix: &str| {
        scalar(samples, &format!("vllm_spec_decode_{suffix}_total")).is_some()
            || scalar(samples, &format!("spec_decode_{suffix}_total")).is_some()
    };
    let mut per_pos: Vec<(u32, f64)> = samples
        .iter()
        .filter(|s| {
            s.name == "vllm_spec_decode_num_accepted_tokens_per_pos_total"
                || s.name == "spec_decode_num_accepted_tokens_per_pos_total"
        })
        .filter_map(|s| {
            let pos = s.label("position").and_then(|p| p.parse::<u32>().ok())?;
            Some((pos, s.value))
        })
        .collect();
    per_pos.sort_by_key(|(p, _)| *p);
    SpecMetrics {
        draft: find("num_draft_tokens"),
        accepted: find("num_accepted_tokens"),
        emitted: find("num_emitted_tokens"),
        drafts: find("num_drafts"),
        counters: is_counter("num_draft_tokens") && is_counter("num_accepted_tokens"),
        accept_rate: find("draft_acceptance_rate"),
        efficiency: find("efficiency"),
        accept_pos: histogram(samples, "vllm_spec_decode_num_accepted_tokens_per_pos")
            .or_else(|| histogram(samples, "spec_decode_num_accepted_tokens_per_pos")),
        per_pos,
    }
}

fn prefix_cache_hit(samples: &[prom::Sample]) -> Option<f64> {
    if let Some(v) = scalar(samples, "vllm_prefix_cache_hit_rate") {
        return Some(v);
    }
    let hits = scalar(samples, "vllm_prefix_cache_hits_total")
        .or_else(|| scalar(samples, "vllm_prefix_cache_hits"));
    let queries = scalar(samples, "vllm_prefix_cache_queries_total")
        .or_else(|| scalar(samples, "vllm_prefix_cache_queries"));
    match (hits, queries) {
        (Some(h), Some(q)) if q > 0.0 => Some(h / q),
        _ => None,
    }
}

fn scalar(samples: &[prom::Sample], name: &str) -> Option<f64> {
    let mut acc: Option<f64> = None;
    for s in samples {
        if s.name == name {
            *acc.get_or_insert(0.0) += s.value;
        }
    }
    acc
}

fn histogram(samples: &[prom::Sample], base: &str) -> Option<HistStats> {
    let count = scalar(samples, &format!("{base}_count"))?;
    let sum = scalar(samples, &format!("{base}_sum")).unwrap_or(0.0);
    let bucket_name = format!("{base}_bucket");
    let mut buckets: Vec<(f64, f64)> = Vec::new();
    for s in samples {
        if s.name == bucket_name
            && let Some(le) = s.label("le").and_then(|v| v.parse::<f64>().ok())
        {
            match buckets.iter_mut().find(|(l, _)| *l == le) {
                Some((_, cnt)) => *cnt += s.value,
                None => buckets.push((le, s.value)),
            }
        }
    }
    buckets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Some(HistStats {
        count,
        sum,
        p50: prom::histogram_quantile(0.5, &buckets),
        p90: prom::histogram_quantile(0.9, &buckets),
        p99: prom::histogram_quantile(0.99, &buckets),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn window_rate_averages_lump_over_span() {
        let now = Instant::now();
        let mut win: VecDeque<(Instant, f64, f64)> = VecDeque::new();
        win.push_back((now - Duration::from_secs(8), 0.0, 0.0));
        win.push_back((now - Duration::from_secs(4), 0.0, 0.0));
        // A whole request's tokens land in one lump at completion:
        win.push_back((now, 44053.0, 100.0));
        prune_window(&mut win, 10.0, now, |s| s.0);
        assert_eq!(win.len(), 3);
        let (t0, p0, g0) = *win.front().unwrap();
        let (t1, p1, g1) = *win.back().unwrap();
        let pr = delta_rate(t0, p0, t1, p1).unwrap();
        let gr = delta_rate(t0, g0, t1, g1).unwrap();
        // ~5.5k tok/s over the 8s span, not 44k in one poll:
        assert!((pr - 44053.0 / 8.0).abs() < 10.0, "prompt rate {pr}");
        assert!((gr - 100.0 / 8.0).abs() < 1.0, "gen rate {gr}");
    }

    #[test]
    fn prune_keeps_two_samples_beyond_window() {
        let now = Instant::now();
        let mut win: VecDeque<(Instant, f64)> = VecDeque::new();
        win.push_back((now - Duration::from_secs(60), 0.0));
        win.push_back((now - Duration::from_secs(40), 10.0));
        win.push_back((now - Duration::from_secs(20), 20.0));
        win.push_back((now, 30.0));
        prune_window(&mut win, 10.0, now, |s| s.0);
        // Oldest two dropped, but never below two samples:
        assert_eq!(win.len(), 2);
        assert_eq!(win.front().unwrap().1, 20.0);
    }

    #[test]
    fn delta_rate_rejects_tiny_spans_and_countershrink() {
        let now = Instant::now();
        assert!(delta_rate(now, 0.0, now, 10.0).is_none());
        let later = now + Duration::from_millis(200);
        assert_eq!(delta_rate(now, 10.0, later, 4.0).unwrap(), 0.0);
    }

    #[test]
    fn trickle_does_not_count_as_progress() {
        // A big prefill starving a concurrent decode to a few tok/s:
        assert!(!is_progress(0.0, 6.0));
        // Real decode, and a prompt lump landing:
        assert!(is_progress(0.0, 80.0));
        assert!(is_progress(500.0, 0.0));
        // Dead silence:
        assert!(!is_progress(0.0, 0.0));
    }

    #[test]
    fn extracts_dflash_style_spec_metrics() {
        let text = "vllm:spec_decode_num_drafts_total{engine=\"0\",model_name=\"m\"} 69322.0\nvllm:spec_decode_num_draft_tokens_total{engine=\"0\",model_name=\"m\"} 327132.0\nvllm:spec_decode_num_accepted_tokens_total{engine=\"0\",model_name=\"m\"} 168006.0\nvllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",position=\"0\"} 54288.0\nvllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",position=\"1\"} 41273.0\nvllm:inter_token_latency_seconds_count{engine=\"0\"} 69322.0\nvllm:inter_token_latency_seconds_sum{engine=\"0\"} 2398.7\n";
        let m = extract(text);
        assert!(m.spec.counters);
        assert_eq!(m.spec.draft, Some(327132.0));
        assert_eq!(m.spec.accepted, Some(168006.0));
        assert_eq!(m.spec.drafts, Some(69322.0));
        assert!(m.spec.emitted.is_none());
        assert_eq!(m.spec.per_pos, vec![(0, 54288.0), (1, 41273.0)]);
        assert!(m.tpot.is_some());
        assert_eq!(m.tpot.unwrap().avg(), Some(2398.7 / 69322.0));
    }

    #[test]
    fn extracts_unprefixed_spec_metrics() {
        let text = "spec_decode_num_draft_tokens_total 10.0\nspec_decode_num_accepted_tokens_total 7.0\nspec_decode_num_emitted_tokens_total 8.0\nspec_decode_efficiency 1.6\n";
        let m = extract(text);
        assert!(m.spec.counters);
        assert_eq!(m.spec.draft, Some(10.0));
        assert_eq!(m.spec.accepted, Some(7.0));
        assert_eq!(m.spec.emitted, Some(8.0));
        assert_eq!(m.spec.efficiency, Some(1.6));
    }

    #[test]
    fn push_history_tracks_compute() {
        let mut vs = crate::state::VllmState::default();
        let now = Instant::now();
        push_history(
            &mut vs,
            now,
            Rates {
                prompt_tps: 1.0,
                gen_tps: 1.0,
            },
            Some(0.5),
        );
        push_history(
            &mut vs,
            now,
            Rates {
                prompt_tps: 2.0,
                gen_tps: 1.0,
            },
            Some(0.0),
        );
        assert_eq!(vs.hist_t.len(), 2);
        assert_eq!(vs.hist_compute.len(), 2);
        assert_eq!(vs.hist_compute[0], 0.5);
        assert_eq!(vs.hist_compute[1], 0.0);
    }

    #[test]
    fn extracts_prompt_compute_source() {
        let text = "vllm:prompt_tokens_by_source_total{source=\"local_compute\"} 754781.0\nvllm:prompt_tokens_by_source_total{source=\"local_cache_hit\"} 29494656.0\n";
        let m = extract(text);
        assert_eq!(m.prompt_compute_total, Some(754781.0));
    }

    #[test]
    fn extracts_waiting_process_and_config() {
        let text = "vllm:num_requests_waiting_by_reason{reason=\"capacity\"} 2.0\nvllm:num_requests_waiting_by_reason{reason=\"deferred\"} 1.0\nprocess_cpu_seconds_total 21.0\nprocess_resident_memory_bytes 1302167552.0\nvllm:cache_config_info{kv_cache_size_tokens=\"3419710\",gpu_memory_utilization=\"0.95\"} 1.0\n";
        let m = extract(text);
        assert_eq!(m.wait_capacity, Some(2.0));
        assert_eq!(m.wait_deferred, Some(1.0));
        assert_eq!(m.process_cpu_total, Some(21.0));
        assert_eq!(m.process_rss_bytes, Some(1302167552.0));
        assert_eq!(m.kv_pool_tokens, Some(3419710.0));
        assert_eq!(m.gpu_mem_util, Some(0.95));
    }

    #[test]
    fn extracts_batch_phase_and_tool_metrics() {
        let text = "vllm:iteration_tokens_total_count 100.0\nvllm:iteration_tokens_total_sum 1400.0\nvllm:request_prefill_time_seconds_count 10.0\nvllm:request_prefill_time_seconds_sum 12.0\nvllm:request_decode_time_seconds_count 10.0\nvllm:request_decode_time_seconds_sum 90.0\nvllm:tool_call_parser_invocations_total{mode=\"streaming\",outcome=\"tool_call\"} 30.0\nvllm:tool_call_parser_invocations_total{mode=\"streaming\",outcome=\"no_tool_call\"} 10.0\nvllm:request_max_num_generation_tokens_count 5.0\nvllm:request_max_num_generation_tokens_sum 2500.0\n";
        let m = extract(text);
        assert_eq!(m.batch.unwrap().avg(), Some(14.0));
        assert_eq!(m.prefill_time.unwrap().avg(), Some(1.2));
        assert_eq!(m.decode_time.unwrap().avg(), Some(9.0));
        assert_eq!(m.tool_calls_total, Some(30.0));
        assert_eq!(m.tool_no_call_total, Some(10.0));
        assert_eq!(m.max_gen_tokens.unwrap().avg(), Some(500.0));
    }

    #[test]
    fn extracts_http_metrics() {
        let text = "http_requests_total{handler=\"/v1/chat/completions\",method=\"POST\",status=\"2xx\"} 214.0\nhttp_requests_total{handler=\"/x\",method=\"POST\",status=\"5xx\"} 3.0\nhttp_request_duration_highr_seconds_bucket{le=\"1.0\"} 8.0\nhttp_request_duration_highr_seconds_bucket{le=\"2.0\"} 59.0\nhttp_request_duration_highr_seconds_bucket{le=\"+Inf\"} 215.0\nhttp_request_duration_highr_seconds_count 215.0\nhttp_request_duration_highr_seconds_sum 2720.7\n";
        let m = extract(text);
        assert_eq!(m.http_requests_total, Some(217.0));
        assert_eq!(m.http_errors_total, Some(3.0));
        let h = m.http_latency.unwrap();
        assert_eq!(h.count, 215.0);
        assert_eq!(h.avg(), Some(2720.7 / 215.0));
        let p50 = h.p50.unwrap();
        assert!((p50 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn extracts_cache_aliases() {
        let text = "vllm:kv_cache_usage_perc 0.5\nvllm:prefix_cache_queries 100.0\nvllm:prefix_cache_hits 25.0\n";
        let m = extract(text);
        assert_eq!(m.gpu_cache_usage, Some(0.5));
        assert_eq!(m.prefix_cache_hit, Some(0.25));
    }

    #[test]
    fn extracts_vllm_metrics() {
        let text = r#"
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 8610.0
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 4022.0
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 2.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 1.0
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.4218
# TYPE vllm:prefix_cache_hit_rate gauge
vllm:prefix_cache_hit_rate 0.1131
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.005"} 1.0
vllm:time_to_first_token_seconds_bucket{le="0.01"} 2.0
vllm:time_to_first_token_seconds_bucket{le="0.02"} 3.0
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 4.0
vllm:time_to_first_token_seconds_sum 0.05
vllm:time_to_first_token_seconds_count 4.0
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.005"} 0.0
vllm:time_per_output_token_seconds_bucket{le="0.01"} 1.0
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 2.0
vllm:time_per_output_token_seconds_sum 0.02
vllm:time_per_output_token_seconds_count 2.0
# TYPE vllm:request_success_total counter
vllm:request_success_total 12.0
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total 0.0
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total 100.0
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total 70.0
# TYPE vllm:spec_decode_num_emitted_tokens_total counter
vllm:spec_decode_num_emitted_tokens_total 85.0
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos histogram
vllm:spec_decode_num_accepted_tokens_per_pos_bucket{le="1"} 30.0
vllm:spec_decode_num_accepted_tokens_per_pos_bucket{le="2"} 70.0
vllm:spec_decode_num_accepted_tokens_per_pos_bucket{le="+Inf"} 100.0
vllm:spec_decode_num_accepted_tokens_per_pos_sum 140.0
vllm:spec_decode_num_accepted_tokens_per_pos_count 100.0
"#;
        let m = extract(text);
        assert_eq!(m.prompt_tokens_total, 8610.0);
        assert_eq!(m.generation_tokens_total, 4022.0);
        assert_eq!(m.running, 2.0);
        assert_eq!(m.waiting, 1.0);
        assert_eq!(m.gpu_cache_usage, Some(0.4218));
        assert_eq!(m.prefix_cache_hit, Some(0.1131));
        let ttft = m.ttft.unwrap();
        assert_eq!(ttft.count, 4.0);
        assert_eq!(ttft.avg(), Some(0.0125));
        let p50 = ttft.p50.unwrap();
        assert!((p50 - 0.01).abs() < 1e-9);
        assert!(m.tpot.is_some());
        assert_eq!(m.preemptions, Some(0.0));
        assert!(m.e2e.is_none());
        assert!(m.spec.counters);
        assert_eq!(m.spec.draft, Some(100.0));
        assert_eq!(m.spec.accepted, Some(70.0));
        assert_eq!(m.spec.emitted, Some(85.0));
        let pos = m.spec.accept_pos.unwrap();
        assert_eq!(pos.avg(), Some(1.4));
    }
}
