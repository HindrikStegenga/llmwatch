use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Axis, Chart, Dataset, GraphType};
use ratatui::widgets::{Block, Cell, Clear, Paragraph, Row, Table};
use ratatui::Frame;

use crate::state::{AppState, GpuInfo};

const PROMPT: Color = Color::Rgb(0, 229, 255);
const GEN: Color = Color::Rgb(255, 145, 0);
const OK: Color = Color::Rgb(80, 250, 123);
const WARN: Color = Color::Rgb(255, 214, 0);
const CRIT: Color = Color::Rgb(255, 64, 64);
const ACCENT: Color = Color::Rgb(189, 147, 249);

pub fn draw(f: &mut Frame, state: &AppState) {
    let area = f.area();
    let header_h = if state.vllm.lock().unwrap().last_err.is_some() {
        4
    } else {
        3
    };
    let outer = Layout::vertical([
        Constraint::Length(header_h),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .split(area);
    let body = Layout::vertical([
        Constraint::Min(13),
        Constraint::Length(11),
        Constraint::Length(10),
    ])
    .split(outer[1]);
    let charts = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(body[0]);
    let mid = Layout::horizontal([
        Constraint::Percentage(34),
        Constraint::Percentage(33),
        Constraint::Percentage(33),
    ])
    .split(body[1]);
    let bottom = Layout::horizontal([
        Constraint::Percentage(30),
        Constraint::Percentage(70),
    ])
    .split(body[2]);

    draw_header(f, state, outer[0]);
    draw_throughput(f, state, charts[0], charts[1]);
    draw_requests(f, state, mid[0]);
    draw_latency(f, state, mid[1]);
    draw_cache(f, state, mid[2]);
    draw_spec(f, state, bottom[0]);
    draw_gpus(f, state, bottom[1]);
    draw_footer(f, state, outer[2]);
    if state.metrics_popup_visible() {
        draw_metrics_popup(f, state, area);
    }
}

fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

fn bold(color: Color) -> Style {
    Style::default().fg(color).add_modifier(Modifier::BOLD)
}

fn draw_header(f: &mut Frame, state: &AppState, area: Rect) {
    let vs = state.vllm.lock().unwrap();
    let (status, color) = if state.is_paused() {
        ("PAUSED", WARN)
    } else if vs.last_err.is_some() {
        ("ERROR", CRIT)
    } else if vs.last_ok.is_some() {
        ("LIVE", OK)
    } else {
        ("CONNECTING", WARN)
    };
    let l1_left = vec![
        Span::styled(" llmwatch ", bold(PROMPT)),
        Span::raw(state.vllm_url.clone()),
        Span::raw("  ·  gpu: "),
        Span::raw(
            state
                .ssh_host
                .clone()
                .unwrap_or_else(|| "off (--ssh host)".into()),
        ),
    ];
    let right = vec![Span::raw(match vs.last_ok.map(|t| t.elapsed().as_secs_f64()) {
        Some(age) => format!(
            " updated {age:.1}s ago · interval {:.1}s ",
            state.interval_secs()
        ),
        None => format!(" never updated · interval {:.1}s ", state.interval_secs()),
    })];
    let left_len: usize = l1_left.iter().map(|s| s.content.chars().count()).sum();
    let right_len: usize =
        right.iter().map(|s| s.content.chars().count()).sum::<usize>() + status.chars().count() + 2;
    let inner_w = area.width.saturating_sub(2) as usize;
    let pad = inner_w.saturating_sub(left_len + right_len).max(1);
    let mut l1 = l1_left;
    l1.push(Span::raw(" ".repeat(pad)));
    l1.extend(right);
    l1.push(Span::styled(format!(" {status} "), bold(color)));

    let mut lines = vec![Line::from(l1)];
    if let Some(err) = &vs.last_err {
        lines.push(Line::from(vec![Span::styled(
            format!(" {err}"),
            Style::default().fg(CRIT),
        )]));
    }
    let p = Paragraph::new(lines).block(
        Block::bordered().title(Span::styled(" vLLM monitor ", bold(Color::White))),
    );
    f.render_widget(p, area);
}

fn draw_throughput(f: &mut Frame, state: &AppState, prompt_area: Rect, gen_area: Rect) {
    let vs = state.vllm.lock().unwrap();
    let (pr, gr) = match vs.rates {
        Some(r) => (r.prompt_tps, r.gen_tps),
        None => (0.0, 0.0),
    };
    let compute_vals = if !vs.hist_compute.is_empty() && vs.hist_compute.len() == vs.hist_t.len()
    {
        Some(&vs.hist_compute[..])
    } else {
        None
    };
    rate_chart(
        f,
        prompt_area,
        "Prompt tok/s (incl. cached)",
        PROMPT,
        &vs.hist_t,
        &vs.hist_prompt,
        pr,
        compute_vals,
        vs.compute_tps,
        true,
    );
    rate_chart(
        f,
        gen_area,
        "Generation tok/s",
        GEN,
        &vs.hist_t,
        &vs.hist_gen,
        gr,
        None,
        None,
        false,
    );
}

#[allow(clippy::too_many_arguments)]
fn rate_chart(
    f: &mut Frame,
    area: Rect,
    title: &str,
    color: Color,
    ts: &[f64],
    vals: &[f64],
    current: f64,
    compute_vals: Option<&[f64]>,
    compute_current: Option<f64>,
    log_scale: bool,
) {
    let mut title_spans = vec![
        Span::styled(format!(" {title} "), bold(Color::White)),
        Span::styled(format!("{current:.1} "), bold(color)),
    ];
    if let Some(c) = compute_current {
        title_spans.push(Span::styled("· compute ", Style::default().fg(WARN)));
        title_spans.push(Span::styled(
            format!("{c:.0} "),
            Style::default().fg(Color::White),
        ));
    }
    let block = Block::bordered().title(Line::from(title_spans));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if ts.len() < 2 {
        f.render_widget(Paragraph::new(" collecting data… ").style(dim()), inner);
        return;
    }
    let x_max = *ts.last().unwrap();
    let data_min = ts.first().copied().unwrap_or(0.0);
    let window = (x_max - data_min).clamp(1.0, 1200.0);
    let x_min = x_max - window;
    let mut pts = Vec::new();
    let mut pts_c: Vec<(f64, f64)> = Vec::new();
    let mut raw: Vec<f64> = Vec::new();
    let mut raw_c: Vec<f64> = Vec::new();
    let compute_vals = compute_vals.filter(|c| c.len() == ts.len());
    let mut y_max = 1.0f64;
    let to_y = |v: f64| -> f64 {
        if log_scale {
            (v + 1.0).log10()
        } else {
            v
        }
    };
    for (i, &t) in ts.iter().enumerate() {
        if t < x_min {
            continue;
        }
        raw.push(vals[i]);
        pts.push((t, to_y(vals[i])));
        if let Some(cv) = compute_vals {
            raw_c.push(cv[i]);
            pts_c.push((t, to_y(cv[i])));
        }
        y_max = y_max.max(vals[i]);
    }
    y_max *= 1.1;
    let y_top = if log_scale { (y_max + 1.0).log10() } else { y_max };
    let y_labels: Vec<Span<'static>> = (0..5)
        .map(|k| {
            let frac = k as f64 / 4.0;
            let v = if log_scale {
                10f64.powf(frac * y_top) - 1.0
            } else {
                frac * y_top
            };
            Span::styled(fmt_count(v.max(0.0)), dim())
        })
        .collect();
    let x_left = if window < 180.0 {
        format!("-{window:.0}s")
    } else {
        format!("-{:.0}m", window / 60.0)
    };
    let mut datasets = vec![
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(color))
            .data(&pts),
    ];
    if !pts_c.is_empty() {
        datasets.push(
            Dataset::default()
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(WARN))
                .data(&pts_c),
        );
    }
    let chart = Chart::new(datasets)
        .x_axis(
            Axis::default()
                .style(dim())
                .bounds([x_min, x_max])
                .labels(vec![Span::styled(x_left, dim()), Span::styled("now", dim())]),
        )
        .y_axis(
            Axis::default()
                .title("tok/s")
                .style(dim())
                .bounds([0.0, y_top])
                .labels(y_labels),
        );
    let rows = if pts_c.is_empty() {
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).split(inner)
    } else {
        Layout::vertical([
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner)
    };
    f.render_widget(chart, rows[0]);

    let mut samples: Vec<f64> = raw.iter().filter(|v| **v > 0.0).copied().collect();
    if !samples.is_empty() {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        f.render_widget(
            Paragraph::new(percentile_line(
                &samples,
                color,
                if pts_c.is_empty() { None } else { Some("compute+cached") },
            )),
            rows[1],
        );
    }
    if !pts_c.is_empty() {
        let mut csamples: Vec<f64> = raw_c.iter().filter(|v| **v > 0.0).copied().collect();
        if !csamples.is_empty() {
            csamples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            f.render_widget(
                Paragraph::new(percentile_line(&csamples, WARN, Some("computed"))),
                rows[2],
            );
        }
    }
}

fn percentile_line(samples: &[f64], color: Color, tag: Option<&str>) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    if let Some(tag) = tag {
        spans.push(Span::styled(format!(" {tag} "), bold(color)));
    }
    let max = *samples.last().unwrap();
    let stats = [
        ("min", samples[0]),
        ("p25", percentile(samples, 0.25)),
        ("p50", percentile(samples, 0.5)),
        ("p75", percentile(samples, 0.75)),
        ("p90", percentile(samples, 0.9)),
        ("p95", percentile(samples, 0.95)),
        ("p99", percentile(samples, 0.99)),
        ("max", max),
    ];
    for (label, value) in stats {
        spans.push(Span::styled(format!(" {label} "), dim()));
        spans.push(Span::styled(
            format!("{value:.0}"),
            Style::default().fg(color),
        ));
    }
    Line::from(spans)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = q * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn kv_line(label: &str, value: String, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<15}"), dim()),
        Span::styled(value, bold(color)),
    ])
}

fn draw_requests(f: &mut Frame, state: &AppState, area: Rect) {
    let vs = state.vllm.lock().unwrap();
    let m = vs.metrics.as_ref();
    let mut lines = Vec::new();
    let fmt_opt = |v: Option<f64>| v.map(fmt_count).unwrap_or_else(|| "—".into());

    let running = m.map(|m| m.running).unwrap_or(0.0);
    lines.push(kv_line("Running", fmt_count(running), OK));
    let waiting = m.map(|m| m.waiting).unwrap_or(0.0);
    lines.push(kv_line(
        "Waiting",
        fmt_count(waiting),
        if waiting > 0.0 { WARN } else { OK },
    ));
    if m.and_then(|m| m.wait_capacity).is_some() || m.and_then(|m| m.wait_deferred).is_some() {
        lines.push(kv_line(
            "Wait cap/def",
            format!(
                "{} / {}",
                fmt_opt(m.and_then(|m| m.wait_capacity)),
                fmt_opt(m.and_then(|m| m.wait_deferred))
            ),
            if waiting > 0.0 { WARN } else { Color::DarkGray },
        ));
    }
    if let Some(sw) = m.and_then(|m| m.swapped) {
        lines.push(kv_line(
            "Swapped",
            fmt_count(sw),
            if sw > 0.0 { CRIT } else { OK },
        ));
    }
    let pre = m.and_then(|m| m.preemptions);
    lines.push(kv_line(
        "Preemptions",
        fmt_opt(pre),
        match pre {
            Some(p) if p > 0.0 => CRIT,
            Some(_) => OK,
            None => Color::DarkGray,
        },
    ));
    lines.push(kv_line(
        "Req/s",
        vs.req_rate
            .map(|r| format!("{r:.1}"))
            .unwrap_or_else(|| "—".into()),
        PROMPT,
    ));
    lines.push(kv_line(
        "Tool/s",
        match (
            vs.tool_rate,
            m.and_then(|m| m.tool_calls_total),
            m.and_then(|m| m.tool_no_call_total),
        ) {
            (Some(r), Some(tc), Some(tn)) => {
                let total = tc + tn;
                let pct = if total > 0.0 { tc / total * 100.0 } else { 0.0 };
                format!("{r:.1} · {pct:.0}%")
            }
            (Some(r), _, _) => format!("{r:.1}"),
            _ => "—".to_string(),
        },
        PROMPT,
    ));
    if let Some(h) = m.and_then(|m| m.batch).filter(|h| h.count > 0.0) {
        lines.push(kv_line(
            "Batch avg/p99",
            format!("{:.1} / {}", h.sum / h.count, fmt_count(h.p99.unwrap_or(0.0))),
            Color::White,
        ));
    }
    let http = m.and_then(|m| m.http_latency);
    lines.push(kv_line(
        "HTTP p50/p99",
        format!(
            "{} / {}",
            fmt_secs(http.and_then(|h| h.p50)),
            fmt_secs(http.and_then(|h| h.p99))
        ),
        ACCENT,
    ));
    match (m.and_then(|m| m.http_errors_total), m.and_then(|m| m.http_requests_total)) {
        (Some(e), Some(t)) if t > 0.0 => {
            let pct = e / t * 100.0;
            lines.push(kv_line(
                "Errors",
                format!("{e:.0} ({pct:.1}%)"),
                if e > 0.0 { CRIT } else { OK },
            ));
        }
        (Some(e), _) => lines.push(kv_line(
            "Errors",
            format!("{e:.0}"),
            if e > 0.0 { CRIT } else { OK },
        )),
        _ => {}
    }
    f.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(Span::styled(" Requests ", bold(Color::White)))),
        area,
    );
}

fn draw_latency(f: &mut Frame, state: &AppState, area: Rect) {
    let vs = state.vllm.lock().unwrap();
    let m = vs.metrics.as_ref();
    let mut lines = Vec::new();
    let ttft = m.and_then(|m| m.ttft);
    let tpot = m.and_then(|m| m.tpot);
    let e2e = m.and_then(|m| m.e2e);

    lines.push(kv_line(
        "TTFT avg",
        fmt_secs(ttft.and_then(|h| h.avg())),
        PROMPT,
    ));
    lines.push(kv_line("TTFT p50", fmt_secs(ttft.and_then(|h| h.p50)), PROMPT));
    lines.push(kv_line("TTFT p90", fmt_secs(ttft.and_then(|h| h.p90)), PROMPT));
    lines.push(kv_line("TTFT p99", fmt_secs(ttft.and_then(|h| h.p99)), PROMPT));
    lines.push(kv_line(
        "TPOT avg",
        fmt_secs(tpot.and_then(|h| h.avg())),
        GEN,
    ));
    if let Some(t) = tpot.and_then(|h| h.avg()).filter(|t| *t > 0.0) {
        lines.push(kv_line(
            "Per-stream",
            format!("{:.1} tok/s", 1.0 / t),
            GEN,
        ));
    }
    lines.push(kv_line("E2E p99", fmt_secs(e2e.and_then(|h| h.p99)), Color::White));
    lines.push(kv_line(
        "Prefill avg",
        fmt_secs(m.and_then(|m| m.prefill_time).and_then(|h| h.avg())),
        PROMPT,
    ));
    lines.push(kv_line(
        "Decode avg",
        fmt_secs(m.and_then(|m| m.decode_time).and_then(|h| h.avg())),
        GEN,
    ));

    f.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(Span::styled(" Latency ", bold(Color::White)))),
        area,
    );
}

fn draw_cache(f: &mut Frame, state: &AppState, area: Rect) {
    let vs = state.vllm.lock().unwrap();
    let block = Block::bordered().title(Span::styled(
        " Context / Cache ",
        bold(Color::White),
    ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let m = vs.metrics.as_ref();
    let kv = m.and_then(|m| m.gpu_cache_usage);
    let hit = m.and_then(|m| m.prefix_cache_hit);
    let cpu = m.and_then(|m| m.cpu_cache_usage);
    if kv.is_none() && hit.is_none() && cpu.is_none() {
        f.render_widget(
            Paragraph::new(" no data — press m to list server metrics ").style(dim()),
            inner,
        );
        return;
    }
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(inner);
    let bar_width = inner.width.saturating_sub(2) as usize;
    let mut kv_label = vec![Span::raw("KV cache (context) usage")];
    if let Some(v) = kv {
        kv_label.push(Span::styled(
            format!("  {:.0}%", v * 100.0),
            bold(kv_color(v)),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(kv_label)), rows[0]);
    if let Some(v) = kv {
        let pct = v.clamp(0.0, 1.0);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                bar(pct, bar_width),
                Style::default().fg(kv_color(pct)),
            ))),
            rows[1],
        );
    }
    let mut hit_label = vec![Span::raw("Prefix cache hit")];
    if let Some(v) = hit {
        hit_label.push(Span::styled(format!("  {:.0}%", v * 100.0), bold(PROMPT)));
    }
    f.render_widget(Paragraph::new(Line::from(hit_label)), rows[2]);
    if let Some(v) = hit {
        let pct = v.clamp(0.0, 1.0);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                bar(pct, bar_width),
                Style::default().fg(PROMPT),
            ))),
            rows[3],
        );
    }

    let mut stats: Vec<Span<'static>> = Vec::new();
    if let Some(p) = m.and_then(|m| m.prompt_len).and_then(|h| h.p99) {
        stats.push(Span::raw(format!("p99 prompt {}", fmt_count(p))));
    }
    if let Some(g) = m.and_then(|m| m.gen_len).and_then(|h| h.p99) {
        if !stats.is_empty() {
            stats.push(Span::raw("  ".to_string()));
        }
        stats.push(Span::raw(format!("p99 gen {}", fmt_count(g))));
    }
    if !stats.is_empty() {
        f.render_widget(Paragraph::new(Line::from(stats)), rows[4]);
    }

    let mut extra: Vec<Span<'static>> = Vec::new();
    if let Some(h) = m.and_then(|m| m.max_gen_tokens).filter(|h| h.count > 0.0) {
        extra.push(Span::raw(format!(
            "max_tokens avg {}",
            fmt_count(h.sum / h.count)
        )));
    }
    if let Some(c) = cpu {
        if !extra.is_empty() {
            extra.push(Span::raw("  ".to_string()));
        }
        extra.push(Span::styled(
            format!("cpu cache {:.0}%", c * 100.0),
            Style::default().fg(kv_color(c)),
        ));
    }
    if !extra.is_empty() {
        f.render_widget(Paragraph::new(Line::from(extra)), rows[5]);
    }

    let mut health: Vec<Span<'static>> = Vec::new();
    if let Some(pct) = vs.cpu_pct {
        health.push(Span::raw(format!("engine cpu {pct:.0}%")));
    }
    if let Some(rss) = m.and_then(|m| m.process_rss_bytes) {
        if !health.is_empty() {
            health.push(Span::raw(" · "));
        }
        health.push(Span::raw(format!("rss {:.1}G", rss / 1073741824.0)));
    }
    if !health.is_empty() {
        f.render_widget(Paragraph::new(Line::from(health)), rows[6]);
    }

    let mut cfg: Vec<Span<'static>> = Vec::new();
    if let Some(t) = m.and_then(|m| m.kv_pool_tokens) {
        cfg.push(Span::raw(format!("KV pool {} tok", fmt_count(t))));
    }
    if let Some(u) = m.and_then(|m| m.gpu_mem_util) {
        if !cfg.is_empty() {
            cfg.push(Span::raw(" · "));
        }
        cfg.push(Span::raw(format!("mem util {u:.2}")));
    }
    if !cfg.is_empty() {
        f.render_widget(Paragraph::new(Line::from(cfg)), rows[7]);
    }
}

fn kv_color(v: f64) -> Color {
    if v >= 0.95 {
        CRIT
    } else if v >= 0.8 {
        WARN
    } else {
        OK
    }
}

fn draw_spec(f: &mut Frame, state: &AppState, area: Rect) {
    let vs = state.vllm.lock().unwrap();
    let block = Block::bordered().title(Span::styled(" Spec decode ", bold(Color::White)));
    let spec = vs.metrics.as_ref().map(|m| &m.spec).filter(|s| s.present());
    let Some(spec) = spec else {
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new(" no spec decode metrics — press m to list server metrics ")
                .style(dim()),
            inner,
        );
        return;
    };

    let mut lines = Vec::new();
    if let Some(r) = vs.spec_rates {
        lines.push(kv_line("Draft", format!("{:.1} tok/s", r.draft_tps), Color::White));
        lines.push(kv_line(
            "Accepted",
            format!("{:.1} tok/s", r.accepted_tps),
            Color::White,
        ));
        if spec.emitted.is_some() {
            lines.push(kv_line(
                "Emitted",
                format!("{:.1} tok/s", r.emitted_tps),
                Color::White,
            ));
        }
    }
    let accept_rate = if spec.counters
        && let (Some(a), Some(d)) = (spec.accepted, spec.draft)
        && d > 0.0
    {
        Some(a / d)
    } else {
        spec.accept_rate
    };
    if let Some(r) = accept_rate {
        lines.push(kv_line("Accept rate", format!("{:.1}%", r * 100.0), accept_color(r)));
    }
    if let (Some(a), Some(dr)) = (spec.accepted, spec.drafts)
        && dr > 0.0
    {
        lines.push(kv_line("Accept/step", format!("{:.2} tok", a / dr), PROMPT));
    }
    if !spec.per_pos.is_empty() {
        let drafts = spec.drafts.unwrap_or(0.0);
        let mut chunk: Vec<Span<'static>> = Vec::new();
        for &(pos, cnt) in spec.per_pos.iter() {
            let denom = if pos == 0 {
                drafts
            } else {
                spec.per_pos
                    .iter()
                    .find(|(p, _)| *p == pos - 1)
                    .map(|(_, c)| *c)
                    .unwrap_or(0.0)
            };
            if denom <= 0.0 {
                continue;
            }
            let rate = (cnt / denom).clamp(0.0, 1.0);
            chunk.push(Span::styled(
                format!("{pos}:{:.0}%", rate * 100.0),
                Style::default().fg(accept_color(rate)),
            ));
            chunk.push(Span::raw(" "));
            if chunk.len() >= 8 {
                lines.push(Line::from(std::mem::take(&mut chunk)));
            }
        }
        if !chunk.is_empty() {
            lines.push(Line::from(chunk));
        }
    }
    let efficiency = if spec.counters
        && let (Some(e), Some(d)) = (spec.emitted, spec.draft)
        && d > 0.0
    {
        Some(e / d)
    } else {
        spec.efficiency
    };
    if let Some(e) = efficiency {
        lines.push(kv_line("Efficiency", format!("{e:.2}x"), PROMPT));
    }
    if let Some(p) = spec.accept_pos.and_then(|h| h.avg()) {
        lines.push(kv_line("Avg accept pos", format!("{p:.2}"), PROMPT));
    }

    f.render_widget(
        Paragraph::new(lines).block(block),
        area,
    );
}

fn accept_color(r: f64) -> Color {
    if r >= 0.7 {
        OK
    } else if r >= 0.4 {
        WARN
    } else {
        CRIT
    }
}

fn draw_gpus(f: &mut Frame, state: &AppState, area: Rect) {
    let gs = state.gpu.lock().unwrap();
    let block = Block::bordered().title(Span::styled(" GPUs ", bold(Color::White)));
    if gs.gpus.is_empty() {
        let (msg, style) = match &gs.last_err {
            Some(e) => (format!(" {e} "), Style::default().fg(CRIT)),
            None => (" no --ssh host configured — GPU stats disabled ".to_string(), dim()),
        };
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(Paragraph::new(msg).style(style), inner);
        return;
    }

    let bold_hdr = Style::default().add_modifier(Modifier::BOLD);
    let header = Row::new(vec![
        Cell::from(Text::from(Span::styled("GPU", bold_hdr))),
        Cell::from(Text::from(Span::styled("Name", bold_hdr))),
        Cell::from(Text::from(Span::styled(format!("{:>5}", "Temp"), bold_hdr))),
        Cell::from(Text::from(Span::styled(format!("{:>5}", "MemT"), bold_hdr))),
        Cell::from(Text::from(Span::styled("Memory", bold_hdr))),
        Cell::from(Text::from(Span::styled(format!("{:>5}", "Util"), bold_hdr))),
        Cell::from(Text::from(Span::styled(format!("{:>13}", "Power"), bold_hdr))),
        Cell::from(Text::from(Span::styled(format!("{:>6}", "PCIe"), bold_hdr))),
        Cell::from(Text::from(Span::styled(format!("{:>13}", "RX/TX"), bold_hdr))),
    ]);
    let widths = [
        Constraint::Length(4),
        Constraint::Min(8),
        Constraint::Length(5),
        Constraint::Length(5),
        Constraint::Length(17),
        Constraint::Length(5),
        Constraint::Length(13),
        Constraint::Length(6),
        Constraint::Length(13),
    ];
    let rows = gs.gpus.iter().map(gpu_row);
    let table = Table::new(rows, widths)
        .block(block)
        .header(header)
        .column_spacing(1);
    f.render_widget(table, area);
}

fn gpu_row(g: &GpuInfo) -> Row<'static> {
    let temp = match g.temp {
        Some(t) => Span::styled(format!("{t:>5.0}"), temp_style(t)),
        None => Span::styled(format!("{:>5}", "—"), dim()),
    };
    let memt = match g.mem_temp {
        Some(t) => Span::styled(format!("{t:>5.0}"), temp_style(t)),
        None => Span::styled(format!("{:>5}", "—"), dim()),
    };
    let mem = match (g.mem_used, g.mem_total) {
        (Some(u), Some(t)) if t > 0.0 => {
            let pct = (u / t).clamp(0.0, 1.0);
            Line::from(vec![
                Span::styled(bar(pct, 6), Style::default().fg(mem_color(pct))),
                Span::raw(format!(" {}/{}", fmt_mib(u), fmt_mib(t))),
            ])
        }
        _ => Line::from(Span::styled("—".to_string(), dim())),
    };
    let util = match g.util {
        Some(u) => Span::styled(
            format!("{:>5}", format!("{u:.0}%")),
            Style::default().fg(mem_color((u / 100.0).clamp(0.0, 1.0))),
        ),
        None => Span::styled(format!("{:>5}", "—"), dim()),
    };
    let power = match (g.power, g.power_limit) {
        (Some(p), Some(l)) if l > 0.0 => {
            Span::raw(format!("{:>13}", format!("{p:.0} / {l:.0} W")))
        }
        (Some(p), _) => Span::raw(format!("{:>13}", format!("{p:.0} W"))),
        _ => Span::styled(format!("{:>13}", "—"), dim()),
    };
    let pcie = match (g.pcie_gen, g.pcie_width) {
        (Some(pgen), Some(w)) => {
            Span::raw(format!("{:>6}", format!("{}x{}", pgen as u32, w as u32)))
        }
        _ => Span::styled(format!("{:>6}", "—"), dim()),
    };
    let link = match (g.pcie_rx, g.pcie_tx) {
        (Some(rx), Some(tx)) => Line::from(vec![
            Span::styled(format!("{:>5}", fmt_rate(rx)), Style::default().fg(OK)),
            Span::raw(" / "),
            Span::styled(format!("{:>5}", fmt_rate(tx)), Style::default().fg(ACCENT)),
        ]),
        _ => Line::from(Span::styled(format!("{:>13}", "—"), dim())),
    };
    Row::new(vec![
        Cell::from(Text::from(Span::raw(format!("{}", g.index)))),
        Cell::from(Text::from(Span::raw(g.name.clone()))),
        Cell::from(Text::from(temp)),
        Cell::from(Text::from(memt)),
        Cell::from(Text::from(mem)),
        Cell::from(Text::from(util)),
        Cell::from(Text::from(power)),
        Cell::from(Text::from(pcie)),
        Cell::from(Text::from(link)),
    ])
}

fn temp_style(t: f64) -> Style {
    if t >= 80.0 {
        bold(CRIT)
    } else if t >= 65.0 {
        Style::default().fg(WARN)
    } else {
        Style::default().fg(OK)
    }
}

fn mem_color(pct: f64) -> Color {
    if pct >= 0.9 {
        CRIT
    } else if pct >= 0.75 {
        WARN
    } else {
        OK
    }
}

fn bar(pct: f64, width: usize) -> String {
    let filled = (pct.clamp(0.0, 1.0) * width as f64).round() as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

fn fmt_mib(mib: f64) -> String {
    if mib >= 1024.0 {
        format!("{:.1}G", mib / 1024.0)
    } else {
        format!("{:.0}M", mib)
    }
}

fn fmt_rate(mbs: f64) -> String {
    if mbs >= 1000.0 {
        format!("{:.1}G", mbs / 1000.0)
    } else if mbs >= 1.0 {
        format!("{:.0}M", mbs)
    } else {
        format!("{:.0}K", mbs * 1000.0)
    }
}

fn fmt_count(v: f64) -> String {
    if v >= 1_000_000.0 {
        format!("{:.1}M", v / 1e6)
    } else if v >= 1_000.0 {
        format!("{:.1}k", v / 1e3)
    } else {
        format!("{}", v as u64)
    }
}

fn fmt_secs(v: Option<f64>) -> String {
    match v {
        None => "—".to_string(),
        Some(s) if s < 1.0 => format!("{:.1} ms", s * 1000.0),
        Some(s) => format!("{:.2} s", s),
    }
}

fn draw_metrics_popup(f: &mut Frame, state: &AppState, area: Rect) {
    let vs = state.vllm.lock().unwrap();
    let names = &vs.all_names;
    let width = area.width.min(70);
    let height = (names.len() as u16 + 2)
        .min(area.height.saturating_sub(4))
        .max(3);
    let popup = Rect {
        x: (area.width - width) / 2,
        y: (area.height - height) / 2,
        width,
        height,
    };
    f.render_widget(Clear, popup);
    let scroll = state
        .metrics_scroll
        .load(std::sync::atomic::Ordering::SeqCst)
        .min(names.len().saturating_sub(1));
    let lines: Vec<Line<'_>> = names
        .iter()
        .skip(scroll)
        .map(|n| Line::from(n.clone()))
        .collect();
    let title = format!(" {} metrics (m close · j/k scroll) ", names.len());
    f.render_widget(
        Paragraph::new(lines).block(
            Block::bordered()
                .title(Span::styled(title, bold(Color::White)))
                .style(Style::default().bg(Color::Black)),
        ),
        popup,
    );
}

fn draw_footer(f: &mut Frame, state: &AppState, area: Rect) {
    let key = Style::default().fg(ACCENT);
    let line = Line::from(vec![
        Span::styled(" q", key),
        Span::raw(" quit  "),
        Span::styled("r", key),
        Span::raw(" refresh  "),
        Span::styled("p", key),
        Span::raw(if state.is_paused() { " resume  " } else { " pause  " }),
        Span::styled("+/-", key),
        Span::raw(format!(" interval ({:.1}s)", state.interval_secs())),
        Span::styled("  m", key),
        Span::raw(" metrics"),
    ]);
    f.render_widget(Paragraph::new(line).style(dim()), area);
}
