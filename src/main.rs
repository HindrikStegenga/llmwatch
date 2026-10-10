mod gpu;
mod prom;
mod state;
mod ui;
mod vllm;

use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};

use state::AppState;

#[derive(Parser)]
#[command(
    name = "llmwatch",
    version,
    about = "TUI monitor for a remote vLLM server and its GPUs"
)]
struct Args {
    /// Base URL of the vLLM server (metrics are read from <url>/metrics)
    #[arg(short, long, default_value = "http://127.0.0.1:8000")]
    url: String,

    /// SSH target for GPU stats, e.g. user@gpu-box (runs nvidia-smi remotely)
    #[arg(short, long)]
    ssh: Option<String>,

    /// Poll interval in seconds
    #[arg(short, long, default_value_t = 1.0)]
    interval: f64,

    /// HTTP request timeout in seconds
    #[arg(long, default_value_t = 5.0)]
    timeout: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut state = AppState::new(args.interval);
    state.vllm_url = args.url.trim_end_matches('/').to_string();
    state.ssh_host = args.ssh.clone();
    let state = Arc::new(state);

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(args, state))
}

async fn run(args: Args, state: Arc<AppState>) -> anyhow::Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs_f64(args.timeout))
        .build()?;
    let metrics_url = format!("{}/metrics", state.vllm_url);
    tokio::spawn(vllm::fetch_loop(state.clone(), client, metrics_url));

    match &state.ssh_host {
        Some(host) => {
            let host = host.clone();
            tokio::spawn(gpu::fetch_loop(state.clone(), host, args.timeout));
        }
        None => {
            state.gpu.lock().unwrap().last_err =
                Some("no --ssh host configured; GPU stats disabled".to_string());
        }
    }

    let mut terminal = ratatui::init();
    let res = event_loop(&mut terminal, &state).await;
    ratatui::restore();
    res
}

async fn event_loop(
    terminal: &mut ratatui::DefaultTerminal,
    state: &Arc<AppState>,
) -> anyhow::Result<()> {
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    loop {
        tick.tick().await;
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c')
                {
                    return Ok(());
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('r') => state.refresh.notify_waiters(),
                    KeyCode::Char('p') => state.toggle_paused(),
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        state.bump_interval(0.5);
                        state.refresh.notify_waiters();
                    }
                    KeyCode::Char('-') | KeyCode::Char('_') => {
                        state.bump_interval(-0.5);
                        state.refresh.notify_waiters();
                    }
                    KeyCode::Char('w') => state.cycle_window(),
                    KeyCode::Char('m') => state.toggle_metrics_popup(),
                    KeyCode::Char('j') | KeyCode::Down => state.scroll_metrics(1),
                    KeyCode::Char('k') | KeyCode::Up => state.scroll_metrics(-1),
                    _ => {}
                }
            }
        }
        terminal.draw(|f| ui::draw(f, state))?;
    }
}
