from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Periodic Weak Spots 评分监控</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #111827; color: #e5e7eb; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 18px; font-size: 24px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .card { background: #1f2937; border: 1px solid #374151; border-radius: 10px; padding: 14px; }
    .label { color: #9ca3af; font-size: 13px; }
    .value { margin-top: 6px; font-size: 23px; font-variant-numeric: tabular-nums; }
    .wide { margin-top: 12px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 9px 8px; border-bottom: 1px solid #374151; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    .ok { color: #86efac; }
    .warn { color: #fbbf24; }
    .muted { color: #9ca3af; }
    .latest { white-space: pre-wrap; word-break: break-word; }
    .record-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 10px; }
    .record-value { margin-top: 4px; white-space: pre-wrap; word-break: break-word; font-variant-numeric: tabular-nums; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .table-wrap { overflow-x: auto; }
    pre { margin: 10px 0 0; white-space: pre-wrap; word-break: break-word; }
    details { margin-top: 12px; }
    summary { cursor: pointer; color: #93c5fd; }
    @media (max-width: 800px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 800px) { .record-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  </style>
</head>
<body>
<main>
  <h1>Periodic Weak Spots 评分监控</h1>
  <div class="grid">
    <div class="card"><div class="label">状态</div><div id="state" class="value">等待</div></div>
    <div class="card"><div class="label">完成数量</div><div id="completed" class="value">0 / 0</div>
    </div>
    <div class="card"><div class="label">当前准确率</div><div id="accuracy" class="value">-</div>
    </div>
    <div class="card"><div class="label">预计剩余</div><div id="eta" class="value">-</div></div>
  </div>
  <div class="grid wide">
    <div class="card"><div class="label">运行时间</div><div id="elapsed" class="value">-</div></div>
    <div class="card"><div class="label">处理速度</div><div id="rate" class="value">-</div></div>
    <div class="card"><div class="label">最后更新</div><div id="updated" class="value">-</div></div>
    <div class="card"><div class="label">最后样本</div><div id="sample" class="value">-</div></div>
  </div>
  <div class="card wide">
    <div class="label">各 phase 统计</div>
    <table>
      <thead><tr><th>phase</th><th>完成数量</th><th>正确数量</th><th>准确率</th></tr></thead>
      <tbody id="phases"><tr><td colspan="4" class="muted">等待评分进度文件</td></tr></tbody>
    </table>
  </div>
  <div class="card wide">
    <div class="label">最后生成文本</div>
    <div id="latest" class="latest muted">-</div>
  </div>
  <div class="card wide">
    <div class="label">运行参数与数据集</div>
    <div class="record-grid">
      <div><div class="label">模型</div><div id="runModel" class="record-value mono">-</div></div>
      <div><div class="label">引擎版本</div><div id="runVersions" class="record-value mono">-</div></div>
      <div><div class="label">批次大小</div><div id="runBatchSize" class="record-value">-</div></div>
      <div><div class="label">进度格式</div><div id="runSchema" class="record-value">-</div></div>
      <div><div class="label">数据集目录</div><div id="runDataset" class="record-value mono">-</div></div>
      <div><div class="label">Tokenizer</div><div id="runTokenizer" class="record-value mono">-</div></div>
      <div><div class="label">引擎参数</div><div id="runEngineArgs" class="record-value mono">-</div></div>
      <div><div class="label">采样参数</div><div id="runSampling" class="record-value mono">-</div></div>
    </div>
    <details>
      <summary>查看完整 run JSON</summary>
      <pre id="runJson" class="mono muted">-</pre>
    </details>
  </div>
  <div class="card wide">
    <div class="label">最后完整结果记录</div>
    <div class="record-grid">
      <div><div class="label">sample_id</div><div id="recordSample" class="record-value mono">-</div></div>
      <div><div class="label">phase / row_index</div><div id="recordPosition" class="record-value">-</div></div>
      <div><div class="label">gold_answer</div><div id="recordGold" class="record-value">-</div></div>
      <div><div class="label">correct</div><div id="recordCorrect" class="record-value">-</div></div>
      <div><div class="label">prompt token 数量</div><div id="recordPromptTokens" class="record-value">-</div></div>
      <div><div class="label">target key</div><div id="recordKey" class="record-value mono">-</div></div>
      <div><div class="label">target value</div><div id="recordValue" class="record-value mono">-</div></div>
      <div><div class="label">finish_reason</div><div id="recordFinish" class="record-value">-</div></div>
    </div>
    <div class="label wide">next token top10</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>rank</th><th>token_id</th><th>vocabulary_token</th><th>decoded_text</th><th>probability</th><th>logprob</th></tr></thead>
        <tbody id="top10"><tr><td colspan="6" class="muted">等待完整结果记录</td></tr></tbody>
      </table>
    </div>
    <details>
      <summary>查看最新完整记录 JSON</summary>
      <pre id="latestRecord" class="mono muted">-</pre>
    </details>
  </div>
</main>
<script>
const byId = (id) => document.getElementById(id);
const percent = (value) => value === null ? '-' : `${(value * 100).toFixed(2)}%`;
const jsonText = (value) => value === null || value === undefined ? '-' : JSON.stringify(value, null, 2);
const displayText = (value) => value === null || value === undefined ? '-' : String(value);
const escapeHtml = (value) => displayText(value).replace(/[&<>\"']/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', "'": '&#39;'
}[character]));
const seconds = (value) => {
  if (value === null || value === undefined || !Number.isFinite(value)) return '-';
  const total = Math.max(0, Math.round(value));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h ? `${h}小时 ${m}分` : `${m}分 ${s}秒`;
};
function render(data) {
  const state = byId('state');
  state.textContent = data.state;
  const stateClass = data.state === '完成' || data.state === '运行中' ? 'ok' : 'warn';
  state.className = `value ${stateClass}`;
  byId('completed').textContent = `${data.completed_rows} / ${data.total_rows}`;
  byId('accuracy').textContent = percent(data.accuracy);
  byId('eta').textContent = data.state === '完成' ? '已完成' : seconds(data.eta_seconds);
  byId('elapsed').textContent = seconds(data.elapsed_seconds);
  byId('rate').textContent = data.rate_rows_per_second === null
    ? '-' : `${data.rate_rows_per_second.toFixed(3)} 条/秒`;
  byId('updated').textContent = data.last_update_age_seconds === null
    ? '-' : `${Math.round(data.last_update_age_seconds)} 秒前`;
  byId('sample').textContent = data.last_sample_id || '-';
  byId('latest').textContent = data.last_completion_text || '-';
  const rows = Object.entries(data.phase_stats);
  byId('phases').innerHTML = rows.length ? rows.map(([phase, value]) =>
    `<tr><td>${phase}</td><td>${value.completed}</td><td>${value.correct}</td><td>${percent(value.accuracy)}</td></tr>`
  ).join('') : '<tr><td colspan="4" class="muted">等待评分进度文件</td></tr>';
  const run = data.run || {};
  byId('runModel').textContent = displayText(run.model || data.model);
  byId('runVersions').textContent = jsonText(run.versions);
  byId('runBatchSize').textContent = displayText(run.batch_size || data.batch_size);
  byId('runSchema').textContent = displayText(data.schema_version);
  byId('runDataset').textContent = displayText(run.dataset_path);
  byId('runTokenizer').textContent = jsonText({
    id: run.tokenizer_id,
    revision: run.tokenizer_revision,
    files: run.tokenizer_files,
  });
  byId('runEngineArgs').textContent = jsonText(run.engine_args);
  byId('runSampling').textContent = jsonText(run.sampling_params);
  byId('runJson').textContent = jsonText(run);

  const record = data.last_record;
  byId('recordSample').textContent = displayText(record && record.sample_id);
  byId('recordPosition').textContent = record ? `${displayText(record.phase)} / ${displayText(record.row_index)}` : '-';
  byId('recordGold').textContent = displayText(record && record.gold_answer);
  byId('recordCorrect').textContent = record && record.completion ? displayText(record.completion.correct) : '-';
  byId('recordPromptTokens').textContent = record && record.input ? displayText(record.input.prompt_token_count) : '-';
  byId('recordKey').textContent = record && record.input ? jsonText(record.input.target_key) : '-';
  byId('recordValue').textContent = record && record.input ? jsonText(record.input.target_value) : '-';
  byId('recordFinish').textContent = record && record.completion ? displayText(record.completion.finish_reason) : '-';
  byId('latestRecord').textContent = jsonText(record);
  const top10 = record && record.next_token_distribution ? record.next_token_distribution.top10 || [] : [];
  byId('top10').innerHTML = top10.length ? top10.map((item) =>
    `<tr><td>${escapeHtml(item.rank)}</td><td>${escapeHtml(item.token_id)}</td><td class="mono">${escapeHtml(item.vocabulary_token)}</td><td>${escapeHtml(item.decoded_text)}</td><td>${escapeHtml(Number(item.probability).toPrecision(8))}</td><td>${escapeHtml(Number(item.logprob).toFixed(6))}</td></tr>`
  ).join('') : '<tr><td colspan="6" class="muted">等待完整结果记录</td></tr>';
}
async function refresh() {
  const response = await fetch(`/api/status?ts=${Date.now()}`);
  render(await response.json());
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = path.read_bytes()
    if not payload:
        return []
    lines = payload.splitlines()
    if not payload.endswith(b"\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line]


def build_status(path: Path) -> dict[str, Any]:
    events = read_events(path)
    start = next((event for event in events if event.get("type") == "start"), None)
    records = [event for event in events if event.get("type") == "record"]
    complete = next((event for event in reversed(events) if event.get("type") == "complete"), None)
    last_record_event = records[-1] if records else None
    last_record = None if last_record_event is None else last_record_event.get("record")
    run = None if start is None else start.get("run")
    now = time.time()
    started_at = None if start is None else float(start["started_at"])
    finished_at = None if complete is None else float(complete["finished_at"])
    end_time = finished_at if finished_at is not None else now
    elapsed = None if started_at is None else max(0.0, end_time - started_at)
    completed_rows = len(records)
    total_rows = 0 if start is None else int(start["total_rows"])
    correct_rows = sum(bool(event["correct"]) for event in records)
    accuracy = None if completed_rows == 0 else correct_rows / completed_rows
    rate = None if elapsed is None or elapsed <= 0 else completed_rows / elapsed
    remaining = max(0, total_rows - completed_rows)
    eta = None if rate is None or rate <= 0 else remaining / rate
    phase_stats: dict[str, dict[str, Any]] = {}
    for event in records:
        phase = str(event["phase"])
        current = phase_stats.setdefault(phase, {"completed": 0, "correct": 0, "accuracy": None})
        current["completed"] += 1
        current["correct"] += int(bool(event["correct"]))
        current["accuracy"] = current["correct"] / current["completed"]
    last_event = last_record_event or start
    if last_event is None:
        last_update = None
    elif last_event.get("type") == "start":
        last_update = float(last_event["started_at"])
    else:
        last_update = float(last_event["finished_at"])
    state = "完成" if complete is not None else "运行中" if start is not None else "等待"
    schema_version = None
    if complete is not None:
        schema_version = complete.get("schema_version")
    elif last_record is not None:
        schema_version = 2
    return {
        "state": state,
        "progress_path": str(path.resolve()),
        "model": None if start is None else start.get("model"),
        "batch_size": None if start is None else start.get("batch_size"),
        "schema_version": schema_version,
        "run": run,
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "correct_rows": correct_rows,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "rate_rows_per_second": rate,
        "eta_seconds": eta,
        "last_update_age_seconds": None if last_update is None else max(0.0, now - last_update),
        "last_sample_id": None if not records else records[-1]["sample_id"],
        "last_completion_text": None if not records else records[-1]["completion_text"],
        "last_record": last_record,
        "phase_stats": {phase: phase_stats[phase] for phase in sorted(phase_stats, key=int)},
        "summary": None if complete is None else complete["summary"],
    }


def make_handler(progress_path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] in ("/", "/index.html"):
                payload = DASHBOARD_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path.split("?", 1)[0] == "/api/status":
                payload = json.dumps(
                    build_status(progress_path), ensure_ascii=False
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(404)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m periodic_weak_spots.niah_128k.monitor")
    parser.add_argument("--progress", type=Path, required=True, help="评分进度 JSONL 文件")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.progress))
    print(f"监控地址：http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
