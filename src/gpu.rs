use std::sync::Arc;
use std::time::Duration;

use tokio::process::Command;

use crate::state::{AppState, GpuInfo};

const QUERY_BASE: &str = "index,name,temperature.gpu,temperature.memory,memory.used,memory.total,utilization.gpu,power.draw,power.limit,fan.speed,pcie.link.gen.current,pcie.link.width.current";
const QUERY_PLAIN: &str = "index,name,temperature.gpu,temperature.memory,memory.used,memory.total,utilization.gpu,power.draw,power.limit,fan.speed";
const Q_MARKER: &str = "---LLMWATCH---";

pub async fn fetch_loop(state: Arc<AppState>, host: String, timeout_secs: f64) {
    loop {
        if !state.is_paused() {
            match fetch(&host, timeout_secs).await {
                Ok(gpus) => {
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

async fn fetch(host: &str, timeout_secs: f64) -> anyhow::Result<Vec<GpuInfo>> {
    let script = |query: &str| {
        format!(
            "nvidia-smi --query-gpu={query} --format=csv,noheader,nounits && echo '{Q_MARKER}' && nvidia-smi -q"
        )
    };
    match run_ssh(host, &script(QUERY_BASE), timeout_secs).await {
        Ok(text) => parse(&text),
        Err(first) => {
            let text = run_ssh(host, &script(QUERY_PLAIN), timeout_secs).await?;
            let _ = first;
            parse(&text)
        }
    }
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

pub fn parse(text: &str) -> anyhow::Result<Vec<GpuInfo>> {
    let (csv, q) = match text.split_once(Q_MARKER) {
        Some((c, q)) => (c, q),
        None => (text, ""),
    };
    let pcie = parse_q_pcie(q);
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
    fn parses_pcie_from_csv() {
        let text = "0, NVIDIA H100 80GB HBM3, 52, 49, 41000, 81559, 96, 596.10, 700, 60, 5, 16\n---LLMWATCH---\nGPU 00000000:1b:00.0\n    PCIe\n        TX Throughput                    : 1100 KB/s\n        RX Throughput                    : 3200 KB/s\n";
        let gpus = parse(text).unwrap();
        assert_eq!(gpus[0].pcie_gen, Some(5.0));
        assert_eq!(gpus[0].pcie_width, Some(16.0));
        assert_eq!(gpus[0].pcie_rx, Some(3.2));
        assert_eq!(gpus[0].pcie_tx, Some(1.1));
    }

    #[test]
    fn parses_pcie_fields_from_q() {
        let text = "0, NVIDIA H100 80GB HBM3, 52, 49, 41000, 81559, 96, 596.10, 700, 60\n1, NVIDIA A100-SXM4-40GB, 71, 68, 32000, 40960, 100, 302.10, 400, 50\n---LLMWATCH---\nGPU 00000000:1b:00.0\n\tProduct Name\t: NVIDIA H100 80GB HBM3\n    PCIe\n        Link Gen Current                 : Gen4\n        Link Width Current               : 16x\n        Max Link Gen                     : Gen5\n        Max Link Width                   : 16x\n        TX Throughput                    : 1100 KB/s\n        RX Throughput                    : 3200 KB/s\nGPU 00000000:1c:00.0\n    PCIe\n        Link Gen                         : Gen3\n        Link Width                       : 8x\n        TX Throughput                    : 0 KB/s\n        RX Throughput                    : N/A\n";
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
