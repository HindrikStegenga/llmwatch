use std::sync::Arc;
use std::time::Duration;

use tokio::process::Command;

use crate::state::{AppState, GpuInfo};

const QUERY_BASE: &str = "index,name,temperature.gpu,temperature.memory,memory.used,memory.total,utilization.gpu,power.draw,power.limit,fan.speed,pcie.link.gen.current,pcie.link.width.current";
const QUERY_PLAIN: &str = "index,name,temperature.gpu,temperature.memory,memory.used,memory.total,utilization.gpu,power.draw,power.limit,fan.speed";
const Q_MARKER: &str = "---LLMWATCH-Q---";
const T_MARKER: &str = "---LLMWATCH-T---";

pub async fn fetch_loop(state: Arc<AppState>, host: String, timeout_secs: f64) {
    let mut topo_cache: Option<String> = None;
    loop {
        if !state.is_paused() {
            match fetch(&host, timeout_secs, topo_cache.as_deref()).await {
                Ok((gpus, fresh_topo)) => {
                    if fresh_topo.is_some() {
                        topo_cache = fresh_topo;
                    }
                    let mut gs = state.gpu.lock().unwrap();
                    gs.gpus = gpus;
                    gs.last_ok = Some(std::time::Instant::now());
                    gs.last_err = None;
                }
                Err(err) => {
                    let mut gs = state.gpu.lock().unwrap();
                    gs.last_err = Some(err.to_string());
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

async fn fetch(
    host: &str,
    timeout_secs: f64,
    topo_cache: Option<&str>,
) -> anyhow::Result<(Vec<GpuInfo>, Option<String>)> {
    let script = |query: &str| {
        let mut s = format!(
            "nvidia-smi --query-gpu={query} --format=csv,noheader,nounits && echo '{Q_MARKER}' && nvidia-smi -q"
        );
        if topo_cache.is_none() {
            s.push_str(&format!(
                " && echo '{T_MARKER}' && (nvidia-smi topo -m || true)"
            ));
        }
        s
    };
    let text = match run_ssh(host, &script(QUERY_BASE), timeout_secs).await {
        Ok(text) => text,
        Err(first) => {
            let text = run_ssh(host, &script(QUERY_PLAIN), timeout_secs).await?;
            let _ = first;
            text
        }
    };
    let (csv, q, topo_sec) = split_sections(&text);
    let fresh = topo_sec
        .as_deref()
        .filter(|t| !parse_topo(t).is_empty());
    let gpus = build_gpus(&csv, &q, fresh.or(topo_cache))?;
    Ok((gpus, fresh.map(str::to_string)))
}

async fn run_ssh(host: &str, script: &str, timeout_secs: f64) -> anyhow::Result<String> {
    let output = Command::new("ssh")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=5")
        .arg("-o")
        .arg("StrictHostKeyChecking=accept-new")
        .arg(host)
        .arg(script)
        .output();
    let output = tokio::time::timeout(Duration::from_secs_f64(timeout_secs.max(8.0)), output).await??;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let msg = if stderr.trim().is_empty() {
            stdout.trim()
        } else {
            stderr.trim()
        };
        anyhow::bail!("nvidia-smi via ssh failed: {msg}");
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn split_sections(text: &str) -> (String, String, Option<String>) {
    let (csv, rest) = match text.split_once(Q_MARKER) {
        Some((c, r)) => (c.to_string(), r.to_string()),
        None => (text.to_string(), String::new()),
    };
    match rest.split_once(T_MARKER) {
        Some((q, t)) => (csv, q.to_string(), Some(t.to_string())),
        None => (csv, rest, None),
    }
}

fn build_gpus(csv: &str, q: &str, topo_text: Option<&str>) -> anyhow::Result<Vec<GpuInfo>> {
    let pcie = parse_q_pcie(q);
    let topo = topo_text.map(parse_topo).unwrap_or_default();
    let mut gpus = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(", ").collect();
        if fields.len() < 10 {
            anyhow::bail!("unexpected nvidia-smi output: {line}");
        }
        let link = pcie.get(i).copied().unwrap_or(PcieInfo::default());
        let tp = topo.get(i).cloned().unwrap_or(TopoInfo::default());
        let (csv_gen, csv_width) = if fields.len() >= 12 {
            (num(fields[10]), num(fields[11]))
        } else {
            (None, None)
        };
        gpus.push(GpuInfo {
            index: fields[0].trim().parse()?,
            name: fields[1].trim().to_string(),
            temp: num(fields[2]),
            mem_temp: num(fields[3]),
            mem_used: num(fields[4]),
            mem_total: num(fields[5]),
            util: num(fields[6]),
            power: num(fields[7]),
            power_limit: num(fields[8]),
            pcie_gen: csv_gen.or(link.link_gen),
            pcie_width: csv_width.or(link.width),
            pcie_rx: link.rx,
            pcie_tx: link.tx,
            p2p: tp.paths,
        });
    }
    Ok(gpus)
}

#[derive(Default, Clone, Copy)]
struct PcieInfo {
    link_gen: Option<f64>,
    width: Option<f64>,
    rx: Option<f64>,
    tx: Option<f64>,
}

#[derive(Default, Clone)]
struct TopoInfo {
    paths: Option<Vec<String>>,
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            while let Some(&n) = chars.peek() {
                chars.next();
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn parse_topo(text: &str) -> Vec<TopoInfo> {
    fn rank(tok: &str) -> Option<u8> {
        match tok {
            "PIX" => Some(4),
            "PXB" => Some(3),
            "PHB" => Some(2),
            "SYS" => Some(1),
            _ => None,
        }
    }
    let is_gpu_row = |t: &str| {
        t.len() > 3 && t.starts_with("GPU") && t[3..].parse::<usize>().is_ok()
    };
    let mut ngpus = 0usize;
    for line in text.lines() {
        let line = strip_ansi(line);
        let tokens: Vec<&str> = line.split('\t').map(str::trim).collect();
        if tokens.first().is_some_and(|t| t.is_empty()) {
            let count = tokens.iter().filter(|t| is_gpu_row(t)).count();
            if count > ngpus {
                ngpus = count;
            }
        }
    }
    let mut out = vec![TopoInfo::default(); ngpus];
    for line in text.lines() {
        let line = strip_ansi(line);
        let tokens: Vec<&str> = line.split('\t').map(str::trim).collect();
        let Some(first) = tokens.first() else {
            continue;
        };
        if !is_gpu_row(first) {
            continue;
        }
        let Ok(k) = first[3..].parse::<usize>() else {
            continue;
        };
        if k >= ngpus {
            continue;
        }
        let mut parts: Vec<String> = Vec::new();
        for j in 0..ngpus {
            if j == k {
                parts.push("X".to_string());
                continue;
            }
            let Some(tok) = tokens.get(1 + j) else {
                parts.push("—".to_string());
                continue;
            };
            if *tok == "X" {
                parts.push("—".to_string());
                continue;
            }
            if tok.starts_with("NV") && tok[2..].parse::<u8>().is_ok() {
                parts.push(tok.to_string());
                continue;
            }
            if rank(tok).is_some() {
                parts.push(tok.to_string());
            } else {
                parts.push("—".to_string());
            }
        }
        let paths = (!parts.is_empty()).then_some(parts);
        out[k] = TopoInfo { paths };
    }
    out
}

fn parse_q_pcie(q: &str) -> Vec<PcieInfo> {
    let mut out = Vec::new();
    let mut cur: Option<PcieInfo> = None;
    for line in q.lines() {
        if line.starts_with("GPU ") {
            if let Some(info) = cur.take() {
                out.push(info);
            }
            cur = Some(PcieInfo::default());
            continue;
        }
        let Some(info) = cur.as_mut() else {
            continue;
        };
        let Some((label, value)) = line.split_once(':') else {
            continue;
        };
        let label = label.trim().to_lowercase();
        if label.contains("link gen") && !label.contains("max") {
            if info.link_gen.is_none() {
                info.link_gen = digits(value);
            }
        } else if label.contains("link width") && !label.contains("max") {
            if info.width.is_none() {
                info.width = digits(value);
            }
        } else if label.contains("rx throughput") {
            if info.rx.is_none() {
                info.rx = num(value).map(|kb| kb / 1000.0);
            }
        } else if label.contains("tx throughput")
            && info.tx.is_none()
        {
            info.tx = num(value).map(|kb| kb / 1000.0);
        }
    }
    if let Some(info) = cur.take() {
        out.push(info);
    }
    out
}

fn digits(s: &str) -> Option<f64> {
    let d: String = s
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    d.parse::<f64>().ok()
}

fn num(s: &str) -> Option<f64> {
    let s = s.trim();
    let end = s
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .unwrap_or(s.len());
    s[..end].parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> anyhow::Result<Vec<GpuInfo>> {
        let (csv, q, topo_sec) = split_sections(text);
        let topo_text = topo_sec.as_deref().filter(|t| !parse_topo(t).is_empty());
        build_gpus(&csv, &q, topo_text)
    }

    #[test]
    fn parses_nvidia_smi_csv() {
        let text = "0, NVIDIA GeForce RTX 4090, 45, N/A, 12345, 24564, 98, 142.50, 450, 55\n1, NVIDIA A100-SXM4-40GB, 71, 68, 32000, 40960, 100, 302.10, 400, N/A\n";
        let gpus = parse(text).unwrap();
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(gpus[0].temp, Some(45.0));
        assert_eq!(gpus[0].mem_temp, None);
        assert_eq!(gpus[0].mem_used, Some(12345.0));
        assert_eq!(gpus[0].mem_total, Some(24564.0));
        assert_eq!(gpus[0].util, Some(98.0));
        assert_eq!(gpus[0].power, Some(142.5));
        assert_eq!(gpus[0].power_limit, Some(450.0));
        assert_eq!(gpus[0].pcie_gen, None);
    }

    #[test]
    fn parses_topo_with_ansi_codes() {
        let text = "\t\u{1b}[4mGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\u{1b}[0m\nGPU0\t X \tPXB\tPXB\tPXB\t0-15\t0\t\tN/A\nGPU1\tPXB\t X \tPXB\tPXB\t0-15\t0\t\tN/A\nGPU2\tPXB\tPXB\t X \tPIX\t0-15\t0\t\tN/A\nGPU3\tPXB\tPXB\tPIX\t X \t0-15\t0\t\tN/A\n";
        let topo = parse_topo(text);
        assert_eq!(topo.len(), 4);
        assert_eq!(
            topo[0].paths,
            Some(vec!["X".to_string(), "PXB".into(), "PXB".into(), "PXB".into()])
        );
        assert_eq!(
            topo[2].paths,
            Some(vec!["PXB".into(), "PXB".into(), "X".to_string(), "PIX".into()])
        );
        assert_eq!(
            topo[3].paths,
            Some(vec!["PXB".into(), "PXB".into(), "PIX".into(), "X".to_string()])
        );
    }

    #[test]
    fn parses_topo_matrix() {
        let text = "\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\nGPU0\t X \tPIX \tPHB \tSYS \t0-23\tN/A\nGPU1\tPIX \t X \tPHB \tSYS \t0-23\tN/A\nGPU2\tPHB \tPHB \t X \tSYS \t0-23\tN/A\nGPU3\tSYS \tSYS \tSYS \t X \t0-23\tN/A\nLegend:\n  X    = Self\n  SYS  = Path traversing multiple PCIe bridges\n";
        let topo = parse_topo(text);
        assert_eq!(topo.len(), 4);
        assert_eq!(
            topo[0].paths,
            Some(vec!["X".to_string(), "PIX".into(), "PHB".into(), "SYS".into()])
        );
        assert_eq!(
            topo[1].paths,
            Some(vec!["PIX".into(), "X".to_string(), "PHB".into(), "SYS".into()])
        );
        assert_eq!(
            topo[2].paths,
            Some(vec!["PHB".into(), "PHB".into(), "X".to_string(), "SYS".into()])
        );
        assert_eq!(
            topo[3].paths,
            Some(vec!["SYS".into(), "SYS".into(), "SYS".into(), "X".to_string()])
        );
    }

    #[test]
    fn parses_topo_nvlink() {
        let text = "\tGPU0\tGPU1\nGPU0\t X \tNV2\nGPU1\tNV2\t X \n";
        let topo = parse_topo(text);
        assert_eq!(
            topo[0].paths,
            Some(vec!["X".to_string(), "NV2".into()])
        );
        assert_eq!(
            topo[1].paths,
            Some(vec!["NV2".into(), "X".to_string()])
        );
    }

    #[test]
    fn parses_pcie_from_csv() {
        let text = "0, NVIDIA H100 80GB HBM3, 52, 49, 41000, 81559, 96, 596.10, 700, 60, 5, 16\n---LLMWATCH-Q---\nGPU 00000000:1b:00.0\n    PCIe\n        TX Throughput                    : 1100 KB/s\n        RX Throughput                    : 3200 KB/s\n";
        let gpus = parse(text).unwrap();
        assert_eq!(gpus[0].pcie_gen, Some(5.0));
        assert_eq!(gpus[0].pcie_width, Some(16.0));
        assert_eq!(gpus[0].pcie_rx, Some(3.2));
        assert_eq!(gpus[0].pcie_tx, Some(1.1));
    }

    #[test]
    fn parses_pcie_fields_from_q() {
        let text = "0, NVIDIA H100 80GB HBM3, 52, 49, 41000, 81559, 96, 596.10, 700, 60\n1, NVIDIA A100-SXM4-40GB, 71, 68, 32000, 40960, 100, 302.10, 400, 50\n---LLMWATCH-Q---\nGPU 00000000:1b:00.0\n\tProduct Name\t: NVIDIA H100 80GB HBM3\n    PCIe\n        Link Gen Current                 : Gen4\n        Link Width Current               : 16x\n        Max Link Gen                     : Gen5\n        Max Link Width                   : 16x\n        TX Throughput                    : 1100 KB/s\n        RX Throughput                    : 3200 KB/s\nGPU 00000000:1c:00.0\n    PCIe\n        Link Gen                         : Gen3\n        Link Width                       : 8x\n        TX Throughput                    : 0 KB/s\n        RX Throughput                    : N/A\n";
        let gpus = parse(text).unwrap();
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].pcie_gen, Some(4.0));
        assert_eq!(gpus[0].pcie_width, Some(16.0));
        assert_eq!(gpus[0].pcie_rx, Some(3.2));
        assert_eq!(gpus[0].pcie_tx, Some(1.1));
        assert_eq!(gpus[1].pcie_gen, Some(3.0));
        assert_eq!(gpus[1].pcie_width, Some(8.0));
        assert_eq!(gpus[1].pcie_tx, Some(0.0));
        assert_eq!(gpus[1].pcie_rx, None);
    }
}
