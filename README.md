# llmwatch

Terminal UI monitor for a remote [vLLM](https://github.com/vllm-project/vllm) server and its GPUs.

llmwatch polls the vLLM Prometheus `/metrics` endpoint and renders live throughput, latency, cache,
speculative-decoding and API panels in the terminal, while GPU temperature, memory and PCIe stats
are collected over SSH via `nvidia-smi`.

![llmwatch TUI](visual.png?v=2)

## Panels

- **Throughput** — prompt tok/s (log scale, cache hits included) overlaid with the *true* computed
  prefill rate, generation tok/s, and percentile lines (min → max, zero samples ignored) for each series
- **Requests** — running / waiting (with wait reason), preemptions, req/s, tool-call rate, batch size,
  HTTP p50/p99 latency, error count
- **Latency** — TTFT avg/p50/p90/p99, TPOT avg, per-stream tok/s, E2E p99, prefill/decode averages
- **Context / Cache** — KV cache usage, prefix cache hit rate, p99 prompt & generation lengths,
  engine process CPU/RSS, KV pool size and memory utilization
- **Spec decode** — draft/accepted tok/s, acceptance rate, accepted tokens per step, per-position
  acceptance breakdown (shown when the server runs speculative decoding)
- **GPUs** — per-GPU core & memory temperature (threshold-colored), VRAM usage, utilization, power,
  PCIe gen × width and RX/TX throughput

## Usage

```bash
cargo build --release
./target/release/llmwatch --url http://gpu-box:8000 --ssh user@gpu-box
```

| Flag | Default | Description |
|---|---|---|
| `--url` | `http://127.0.0.1:8000` | Base URL of the vLLM server (metrics read from `<url>/metrics`) |
| `--ssh` | — | SSH target for GPU stats, e.g. `user@gpu-box` |
| `--interval` | `1.0` | Poll interval in seconds (0.5–30, adjustable at runtime) |
| `--timeout` | `5.0` | HTTP request timeout in seconds |

Keys: `q` quit · `r` refresh now · `p` pause · `+`/`-` poll interval · `m` list all metric names the
server exposes · `j`/`k` or arrows scroll the metric list.

## Requirements

- A vLLM server with `/metrics` exposed. Metric names from recent vLLM builds are matched directly,
  with fallbacks for older variants (e.g. `gpu_cache_usage_perc` vs `kv_cache_usage_perc`,
  `time_per_output_token_seconds` vs `inter_token_latency_seconds`, prefixed and unprefixed
  `spec_decode_*` names).
- For the GPU panel: key-based SSH access to the GPU host with `nvidia-smi` on `PATH`. If the driver
  rejects the PCIe throughput query fields, llmwatch falls back automatically and reads PCIe info
  from `nvidia-smi -q` instead.

## Disclaimer

This project was built **end-to-end by an AI coding agent** (opencode). It has not been audited by
a human — review the source before running it against production systems. Use at your own risk.

## Acknowledgements

Developed and tested against a GLM-5.3-Flash deployment set up following the
[glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe).
Metric-name compatibility — including the `spec_decode_*` counters, per-position acceptance counters
and `inter_token_latency_seconds` naming — was validated against that setup.
