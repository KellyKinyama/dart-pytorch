/// Local GPT-style chat server with document upload — "Copilot-lite" RAG.
///
/// Run it, open http://127.0.0.1:8090/ in a browser, drop a `.txt` /
/// `.md` file (or several) into the drop-zone, then chat about them.
/// The server serves its own single-file HTML UI at `/` so no
/// separate frontend build is needed — everything is offline.
///
/// The whole thing is the R6 chunked-RAG pipeline behind an HTTP API:
///
///   * On upload: token-chunk the document (200 tok / 100 stride),
///     embed each chunk with `lastTokenHidden` from
///     [bin/_lm_encoder.dart](_lm_encoder.dart), append to the
///     in-memory chunk table. Because centring uses a **corpus mean**
///     that shifts every time new documents arrive, the index is
///     rebuilt from scratch after each upload — trivial for the few
///     hundred chunks a chat session realistically holds.
///
///   * On chat: encode the user message → search top-K chunks →
///     prepend them plus the last few conversation turns to the
///     prompt → `.generate(...)` → return the reply plus which
///     chunks were retrieved (source citations).
///
/// The point of this file is **not** a production system — it's the
/// smallest Copilot-style chat surface you can build on top of the
/// primitives already in the tree. Every generation runs the whole
/// pretrained HuggingFace GPT-2 family checkpoint you point it at,
/// on CPU by default or GPU with `--gpu` on WSL.
///
/// Endpoints:
///
///   GET  /              index.html (chat UI)
///   GET  /health        `{status, model, device, ...}`
///   GET  /status        `{numDocs, numChunks, historyLen}`
///   POST /upload        text/plain body, header `X-Filename: name.txt`
///                       -> `{ok, docId, filename, chunks, totalChunks}`
///   POST /chat          JSON `{message}`
///                       -> `{reply, retrieved: [{docTitle, span, score, preview}], ms}`
///   POST /reset         clears the index + conversation history
///
/// Run:
///
/// ```sh
///   dart run bin/rag_chat_server.dart --port 8090
/// ```
///
/// Bigger checkpoint, GPU (WSL):
///
/// ```sh
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/rag_chat_server.dart \
///       --path models/gpt2-medium/model.safetensors \
///       --vocab models/gpt2-medium/tokenizer.json \
///       --preset medium --gpu --port 8090
/// ```
///
/// Then open http://127.0.0.1:8090/ in your browser.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

import '_lm_encoder.dart';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const int _chunkTokens = 200;
const int _chunkStride = 100;
const int _topKChunks = 4;
const int _historyTurns = 3; // last N user/assistant pairs to include
const int _maxNewTokens = 120;
const double _temperature = 0.7;
const int _generatorTopK = 40;
const int _maxUploadBytes = 5 * 1024 * 1024; // 5 MB per file
const int _maxTotalChunks = 500;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

class _Chunk {
  _Chunk({
    required this.id,
    required this.docId,
    required this.docTitle,
    required this.startTok,
    required this.endTok,
    required this.tokenIds,
    required this.text,
    required this.rawVec,
  });
  final int id;
  final int docId;
  final String docTitle;
  final int startTok;
  final int endTok;
  final List<int> tokenIds;
  final String text;
  final Float32List rawVec; // uncentered, unnormalised
}

class _Doc {
  _Doc({required this.id, required this.title, required this.numChunks});
  final int id;
  final String title;
  final int numChunks;
}

class _Turn {
  _Turn({required this.role, required this.text});
  final String role; // 'user' | 'assistant'
  final String text;
}

class _State {
  _State({
    required this.model,
    required this.tokenizer,
    required this.cfg,
    required this.presetLabel,
    required this.deviceLabel,
  });

  final GPT model;
  final HFBpeTokenizer tokenizer;
  final GPTConfig cfg;
  final String presetLabel;
  final String deviceLabel;

  final List<_Doc> docs = [];
  final List<_Chunk> chunks = [];
  IndexFlat index = IndexFlatIP(1); // recreated on first upload
  Float32List? corpusMean;
  final List<_Turn> history = [];
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

Future<void> main(List<String> args) async {
  final srvOpts = _parseServerArgs(args);
  final opts = parseEncoderArgs(srvOpts.encoderArgs, programHelp: _help);
  final loaded = loadEncoder(opts);

  final state = _State(
    model: loaded.model,
    tokenizer: loaded.tokenizer,
    cfg: loaded.config,
    presetLabel: opts.preset,
    deviceLabel: opts.gpu ? 'gpu' : 'cpu',
  );

  final server = await HttpServer.bind(srvOpts.host, srvOpts.port);
  stdout.writeln(
    '\nrag_chat_server (${state.presetLabel}, '
    '${state.deviceLabel}) listening on '
    'http://${srvOpts.host}:${srvOpts.port}',
  );
  stdout.writeln('  open the URL above in your browser.');
  stdout.writeln('  GET  /health   /status');
  stdout.writeln('  POST /upload   /chat   /reset');

  await for (final req in server) {
    try {
      await _handle(state, req);
    } catch (e, st) {
      stderr.writeln('handler error: $e\n$st');
      try {
        req.response
          ..statusCode = 500
          ..headers.contentType = ContentType.json
          ..write(jsonEncode({'error': e.toString()}));
        await req.response.close();
      } catch (_) {}
    }
  }
}

// ---------------------------------------------------------------------------
// Request dispatch
// ---------------------------------------------------------------------------

Future<void> _handle(_State state, HttpRequest req) async {
  final method = req.method;
  final path = req.uri.path;

  void writeJson(int status, Object body) {
    req.response
      ..statusCode = status
      ..headers.contentType = ContentType.json
      ..write(jsonEncode(body));
  }

  if (method == 'GET' && (path == '/' || path == '/index.html')) {
    req.response
      ..statusCode = 200
      ..headers.contentType = ContentType.html
      ..write(_indexHtml);
    await req.response.close();
    return;
  }

  if (method == 'GET' && path == '/health') {
    writeJson(200, {
      'status': 'ok',
      'model': state.presetLabel,
      'device': state.deviceLabel,
      'embedDim': state.cfg.embedDim,
      'numLayers': state.cfg.numLayers,
      'maxCtx': state.cfg.maxCtx,
    });
    await req.response.close();
    return;
  }

  if (method == 'GET' && path == '/status') {
    writeJson(200, {
      'numDocs': state.docs.length,
      'numChunks': state.chunks.length,
      'historyLen': state.history.length,
      'docs': [
        for (final d in state.docs)
          {'id': d.id, 'title': d.title, 'numChunks': d.numChunks},
      ],
    });
    await req.response.close();
    return;
  }

  if (method == 'POST' && path == '/upload') {
    await _handleUpload(state, req, writeJson);
    await req.response.close();
    return;
  }

  if (method == 'POST' && path == '/chat') {
    await _handleChat(state, req, writeJson);
    await req.response.close();
    return;
  }

  if (method == 'POST' && path == '/reset') {
    state.docs.clear();
    state.chunks.clear();
    state.index = IndexFlatIP(state.cfg.embedDim);
    state.corpusMean = null;
    state.history.clear();
    writeJson(200, {'ok': true});
    await req.response.close();
    return;
  }

  writeJson(404, {'error': 'no route for $method $path'});
  await req.response.close();
}

// ---------------------------------------------------------------------------
// /upload
// ---------------------------------------------------------------------------

Future<void> _handleUpload(
  _State state,
  HttpRequest req,
  void Function(int, Object) writeJson,
) async {
  // Header X-Filename gives the display title; body is raw utf8 text.
  final filename = req.headers.value('x-filename') ?? 'doc.txt';

  final buf = <int>[];
  var total = 0;
  await for (final chunk in req) {
    total += chunk.length;
    if (total > _maxUploadBytes) {
      writeJson(413, {
        'error': 'upload too large — max ${_maxUploadBytes ~/ 1024} KiB',
      });
      return;
    }
    buf.addAll(chunk);
  }
  final text = utf8.decode(buf, allowMalformed: true);
  if (text.trim().isEmpty) {
    writeJson(400, {'error': 'empty upload body'});
    return;
  }

  // Chunk this doc.
  final docId = state.docs.length;
  final tokens = state.tokenizer.encode(text);
  final newChunks = <_Chunk>[];

  void addChunk(int start, int end, List<int> ids) {
    final chunkText = state.tokenizer.decode(ids);
    final raw = lastTokenHidden(state.model, ids);
    newChunks.add(
      _Chunk(
        id: state.chunks.length + newChunks.length,
        docId: docId,
        docTitle: filename,
        startTok: start,
        endTok: end,
        tokenIds: ids,
        text: chunkText,
        rawVec: raw,
      ),
    );
  }

  if (tokens.length <= _chunkTokens) {
    addChunk(0, tokens.length, tokens);
  } else {
    for (var start = 0; start < tokens.length; start += _chunkStride) {
      final end = start + _chunkTokens < tokens.length
          ? start + _chunkTokens
          : tokens.length;
      addChunk(start, end, tokens.sublist(start, end));
      if (end == tokens.length) break;
    }
  }

  if (state.chunks.length + newChunks.length > _maxTotalChunks) {
    writeJson(413, {
      'error':
          'index would exceed $_maxTotalChunks chunks — '
          'POST /reset first',
    });
    return;
  }

  state.chunks.addAll(newChunks);
  state.docs.add(_Doc(id: docId, title: filename, numChunks: newChunks.length));

  // Rebuild the whole index: the corpus mean has moved.
  _rebuildIndex(state);

  writeJson(200, {
    'ok': true,
    'docId': docId,
    'filename': filename,
    'chunks': newChunks.length,
    'totalChunks': state.chunks.length,
    'tokens': tokens.length,
  });
}

void _rebuildIndex(_State state) {
  final d = state.cfg.embedDim;
  if (state.chunks.isEmpty) {
    state.index = IndexFlatIP(d);
    state.corpusMean = null;
    return;
  }
  final raws = [for (final c in state.chunks) c.rawVec];
  final mean = meanVector(raws, d);
  state.corpusMean = mean;
  final idx = IndexFlatIP(d);
  for (final v in raws) {
    idx.add([centerAndNormalize(v, mean)]);
  }
  state.index = idx;
}

// ---------------------------------------------------------------------------
// /chat
// ---------------------------------------------------------------------------

Future<void> _handleChat(
  _State state,
  HttpRequest req,
  void Function(int, Object) writeJson,
) async {
  final body = await utf8.decoder.bind(req).join();
  final json = jsonDecode(body) as Map<String, dynamic>;
  final message = (json['message'] as String?)?.trim() ?? '';
  if (message.isEmpty) {
    writeJson(400, {'error': 'empty message'});
    return;
  }

  final sw = Stopwatch()..start();

  // ---- 1. Retrieve (skip if no docs uploaded yet) --------------
  final retrieved = <Map<String, dynamic>>[];
  final ctxLines = <String>[];
  if (state.chunks.isNotEmpty && state.corpusMean != null) {
    final qRaw = lastTokenHidden(state.model, state.tokenizer.encode(message));
    final qVec = centerAndNormalize(qRaw, state.corpusMean!);
    final k = _topKChunks < state.chunks.length
        ? _topKChunks
        : state.chunks.length;
    final res = state.index.search([qVec], k);
    final ids = res.ids[0];
    final scores = res.distances[0];
    for (var i = 0; i < ids.length; i++) {
      final c = state.chunks[ids[i]];
      ctxLines.add('[${i + 1}] ${c.text.trim()}');
      final preview = c.text.length > 140
          ? '${c.text.substring(0, 140).replaceAll('\n', ' ')}...'
          : c.text.replaceAll('\n', ' ');
      retrieved.add({
        'docTitle': c.docTitle,
        'span': '${c.startTok}-${c.endTok}',
        'score': scores[i],
        'preview': preview,
      });
    }
  }

  // ---- 2. Build prompt: context + last N turns + current Q ------
  final buf = StringBuffer();
  if (ctxLines.isNotEmpty) {
    buf.writeln('Context:');
    for (final line in ctxLines) {
      buf.writeln(line);
    }
    buf.writeln();
  }
  if (state.history.isNotEmpty) {
    buf.writeln('Previous conversation:');
    final start = state.history.length > _historyTurns * 2
        ? state.history.length - _historyTurns * 2
        : 0;
    for (var i = start; i < state.history.length; i++) {
      final t = state.history[i];
      final tag = t.role == 'user' ? 'User' : 'Assistant';
      buf.writeln('$tag: ${t.text}');
    }
    buf.writeln();
  }
  buf.write('User: $message\nAssistant:');
  final prompt = buf.toString();
  var promptIds = state.tokenizer.encode(prompt);

  // ---- 3. Guard against overflow: trim history first, then chunks
  while (promptIds.length + _maxNewTokens > state.cfg.maxCtx &&
      state.history.isNotEmpty) {
    state.history.removeAt(0);
    // Rebuild prompt.
    final b = StringBuffer();
    if (ctxLines.isNotEmpty) {
      b.writeln('Context:');
      for (final line in ctxLines) {
        b.writeln(line);
      }
      b.writeln();
    }
    if (state.history.isNotEmpty) {
      b.writeln('Previous conversation:');
      for (final t in state.history) {
        b.writeln('${t.role == 'user' ? 'User' : 'Assistant'}: ${t.text}');
      }
      b.writeln();
    }
    b.write('User: $message\nAssistant:');
    promptIds = state.tokenizer.encode(b.toString());
  }
  while (promptIds.length + _maxNewTokens > state.cfg.maxCtx &&
      ctxLines.length > 1) {
    ctxLines.removeLast();
    retrieved.removeLast();
    final b = StringBuffer()
      ..writeln('Context:')
      ..writeAll(ctxLines.map((l) => '$l\n'))
      ..writeln()
      ..write('User: $message\nAssistant:');
    promptIds = state.tokenizer.encode(b.toString());
  }

  // ---- 4. Generate ---------------------------------------------
  final full = state.model.generate(
    promptIds.map((i) => i.toDouble()).toList(),
    maxNewTokens: _maxNewTokens,
    temperature: _temperature,
    topK: _generatorTopK,
  );
  final answerIds = full
      .sublist(promptIds.length)
      .map((d) => d.toInt())
      .toList();
  var answer = state.tokenizer.decode(answerIds).trim();

  // Trim at the next "User:" turn if the LM tried to keep the
  // conversation going on its own.
  final userStop = answer.indexOf('\nUser:');
  if (userStop > 0) answer = answer.substring(0, userStop).trim();

  state.history.add(_Turn(role: 'user', text: message));
  state.history.add(_Turn(role: 'assistant', text: answer));

  sw.stop();
  writeJson(200, {
    'reply': answer,
    'retrieved': retrieved,
    'ms': sw.elapsed.inMilliseconds,
    'promptTokens': promptIds.length,
    'newTokens': answerIds.length,
  });
}

// ---------------------------------------------------------------------------
// Server CLI (separate from encoder CLI so we can also pass --port)
// ---------------------------------------------------------------------------

class _ServerOpts {
  _ServerOpts({
    required this.host,
    required this.port,
    required this.encoderArgs,
  });
  final String host;
  final int port;
  final List<String> encoderArgs;
}

_ServerOpts _parseServerArgs(List<String> args) {
  var host = '127.0.0.1';
  var port = 8090;
  final encoderArgs = <String>[];
  for (var i = 0; i < args.length; i++) {
    final a = args[i];
    if (a == '--port' && i + 1 < args.length) {
      port = int.parse(args[++i]);
    } else if (a == '--host' && i + 1 < args.length) {
      host = args[++i];
    } else if (a == '-h' || a == '--help') {
      stdout.writeln(_help);
      exit(0);
    } else {
      encoderArgs.add(a);
    }
  }
  return _ServerOpts(host: host, port: port, encoderArgs: encoderArgs);
}

const String _help = '''
Local GPT-style chat server with document upload.

Usage:
  dart run bin/rag_chat_server.dart [flags]

Server:
  --host HOST      bind address     (default: 127.0.0.1 — local only)
  --port PORT      TCP port         (default: 8090)

Model:
  --path PATH      safetensors weights (default: models/distilgpt2/model.safetensors)
  --vocab PATH     tokenizer.json     (default: models/distilgpt2/tokenizer.json)
  --preset NAME    distilgpt2 | small | medium | large  (default: distilgpt2)
  --gpu            run on CUDA (default: CPU)

Then open http://127.0.0.1:PORT/ in your browser.
''';

// ---------------------------------------------------------------------------
// Embedded HTML UI — one string so this file has no external assets.
// ---------------------------------------------------------------------------

const String _indexHtml = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>dart-pytorch RAG chat</title>
<style>
  :root {
    --bg: #f6f7f9;
    --panel: #ffffff;
    --border: #e2e5ea;
    --text: #1c1e21;
    --muted: #6b7280;
    --accent: #2563eb;
    --user: #dbeafe;
    --assistant: #f3f4f6;
    --src: #fef3c7;
  }
  html, body { height: 100%; margin: 0; }
  body {
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI",
          Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: grid;
    grid-template-columns: 260px 1fr;
    grid-template-rows: 1fr;
    height: 100vh;
  }
  aside {
    background: var(--panel);
    border-right: 1px solid var(--border);
    padding: 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
  aside h1 { font-size: 15px; margin: 0 0 4px 0; }
  aside .sub { color: var(--muted); font-size: 12px; margin-bottom: 16px; }
  aside .drop {
    border: 2px dashed var(--border);
    border-radius: 8px;
    padding: 18px 12px;
    text-align: center;
    color: var(--muted);
    cursor: pointer;
    transition: border-color .15s, background .15s;
  }
  aside .drop.dragging { border-color: var(--accent); background: #eff6ff; }
  aside .drop input { display: none; }
  aside .doclist {
    margin-top: 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 13px;
  }
  aside .doc {
    padding: 6px 8px;
    background: var(--assistant);
    border-radius: 6px;
    display: flex;
    justify-content: space-between;
    gap: 8px;
  }
  aside .doc .name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  aside .doc .count { color: var(--muted); font-size: 11px; }
  aside button.reset {
    margin-top: auto;
    padding: 8px;
    border: 1px solid var(--border);
    background: white;
    border-radius: 6px;
    cursor: pointer;
  }
  aside button.reset:hover { background: #fff7ed; border-color: #fdba74; }

  main {
    display: grid;
    grid-template-rows: 1fr auto;
    height: 100vh;
    min-height: 0;
  }
  #log {
    padding: 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .msg { max-width: 780px; }
  .msg .role { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
  .msg .bubble {
    padding: 10px 14px;
    border-radius: 10px;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .msg.user .bubble { background: var(--user); }
  .msg.assistant .bubble { background: var(--assistant); }
  .msg .sources {
    margin-top: 6px;
    font-size: 12px;
    color: var(--muted);
  }
  .msg .sources details summary {
    cursor: pointer;
    padding: 2px 0;
  }
  .msg .sources .src {
    background: var(--src);
    padding: 6px 10px;
    border-radius: 6px;
    margin: 4px 0;
    color: #78350f;
  }
  .msg .sources .src .head {
    font-weight: 600;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px;
  }
  .msg .meta { color: var(--muted); font-size: 11px; margin-top: 4px; }

  form#chat {
    border-top: 1px solid var(--border);
    background: var(--panel);
    padding: 12px 20px;
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }
  form#chat textarea {
    flex: 1;
    resize: none;
    min-height: 40px;
    max-height: 200px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font: inherit;
    outline: none;
  }
  form#chat textarea:focus { border-color: var(--accent); }
  form#chat button {
    padding: 10px 18px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font: inherit;
  }
  form#chat button:disabled { opacity: .5; cursor: not-allowed; }

  .toast {
    position: fixed;
    right: 20px;
    bottom: 20px;
    background: #1f2937;
    color: white;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 13px;
    opacity: 0;
    transition: opacity .2s;
  }
  .toast.show { opacity: 1; }
</style>
</head>
<body>
<aside>
  <h1>dart-pytorch RAG chat</h1>
  <div class="sub" id="modelInfo">loading…</div>

  <label class="drop" id="drop">
    <input type="file" id="fileInput" accept=".txt,.md,.markdown,.log">
    <div><b>Click or drop</b><br>a .txt / .md file</div>
  </label>

  <div class="doclist" id="doclist"></div>

  <button class="reset" id="resetBtn" type="button">
    Reset (clear docs + history)
  </button>
</aside>

<main>
  <div id="log">
    <div class="msg assistant">
      <div class="role">assistant</div>
      <div class="bubble">Hi. Drop one or more text files on the left, then ask me questions about them. Everything runs locally against the loaded HuggingFace GPT-2 checkpoint.</div>
    </div>
  </div>

  <form id="chat">
    <textarea id="input" placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"></textarea>
    <button id="sendBtn" type="submit">Send</button>
  </form>
</main>

<div class="toast" id="toast"></div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const log = $('log');
  const form = $('chat');
  const input = $('input');
  const sendBtn = $('sendBtn');
  const drop = $('drop');
  const fileInput = $('fileInput');
  const doclist = $('doclist');
  const resetBtn = $('resetBtn');
  const modelInfo = $('modelInfo');
  const toastEl = $('toast');

  function toast(text, ms = 2000) {
    toastEl.textContent = text;
    toastEl.classList.add('show');
    clearTimeout(toast._t);
    toast._t = setTimeout(() => toastEl.classList.remove('show'), ms);
  }

  function addMsg(role, text, extras = {}) {
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;

    const roleEl = document.createElement('div');
    roleEl.className = 'role';
    roleEl.textContent = role;
    wrap.appendChild(roleEl);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text || '…';
    wrap.appendChild(bubble);

    if (extras.retrieved && extras.retrieved.length) {
      const src = document.createElement('div');
      src.className = 'sources';
      const det = document.createElement('details');
      const sum = document.createElement('summary');
      sum.textContent = `${extras.retrieved.length} source chunk${extras.retrieved.length === 1 ? '' : 's'} used`;
      det.appendChild(sum);
      for (const r of extras.retrieved) {
        const s = document.createElement('div');
        s.className = 'src';
        const head = document.createElement('div');
        head.className = 'head';
        head.textContent = `${r.docTitle}  tok ${r.span}  cos=${Number(r.score).toFixed(3)}`;
        s.appendChild(head);
        const body = document.createElement('div');
        body.textContent = r.preview;
        s.appendChild(body);
        det.appendChild(s);
      }
      src.appendChild(det);
      wrap.appendChild(src);
    }
    if (extras.meta) {
      const m = document.createElement('div');
      m.className = 'meta';
      m.textContent = extras.meta;
      wrap.appendChild(m);
    }

    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
    return { wrap, bubble };
  }

  async function refreshStatus() {
    try {
      const r = await fetch('/status');
      const j = await r.json();
      doclist.innerHTML = '';
      for (const d of j.docs) {
        const el = document.createElement('div');
        el.className = 'doc';
        const nm = document.createElement('span');
        nm.className = 'name';
        nm.textContent = d.title;
        const ct = document.createElement('span');
        ct.className = 'count';
        ct.textContent = `${d.numChunks} ch`;
        el.appendChild(nm);
        el.appendChild(ct);
        doclist.appendChild(el);
      }
    } catch (_) {}
  }

  async function refreshHealth() {
    try {
      const r = await fetch('/health');
      const j = await r.json();
      modelInfo.textContent = `${j.model} · ${j.device} · d=${j.embedDim} · L=${j.numLayers} · ctx=${j.maxCtx}`;
    } catch (_) {
      modelInfo.textContent = '(server unreachable)';
    }
  }

  async function uploadFile(file) {
    if (!file) return;
    toast(`uploading ${file.name}…`, 6000);
    const text = await file.text();
    const r = await fetch('/upload', {
      method: 'POST',
      headers: {
        'content-type': 'text/plain; charset=utf-8',
        'x-filename': file.name,
      },
      body: text,
    });
    const j = await r.json();
    if (!r.ok) {
      toast('upload failed: ' + (j.error || r.status));
      return;
    }
    toast(`indexed "${file.name}" — ${j.chunks} chunks (${j.tokens} tok)`);
    await refreshStatus();
  }

  async function send(msg) {
    if (!msg.trim()) return;
    addMsg('user', msg);
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;
    const { bubble } = addMsg('assistant', '', { meta: 'thinking…' });
    try {
      const r = await fetch('/chat', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ message: msg }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.status);
      bubble.textContent = j.reply || '(no reply)';
      bubble.parentElement.querySelectorAll('.meta,.sources')
        .forEach((n) => n.remove());
      if (j.retrieved && j.retrieved.length) {
        addSources(bubble.parentElement, j.retrieved);
      }
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = `${j.newTokens} tok in ${j.ms} ms · prompt ${j.promptTokens} tok`;
      bubble.parentElement.appendChild(meta);
    } catch (e) {
      bubble.textContent = 'error: ' + e.message;
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function addSources(msgEl, retrieved) {
    const src = document.createElement('div');
    src.className = 'sources';
    const det = document.createElement('details');
    const sum = document.createElement('summary');
    sum.textContent = `${retrieved.length} source chunk${retrieved.length === 1 ? '' : 's'} used`;
    det.appendChild(sum);
    for (const r of retrieved) {
      const s = document.createElement('div');
      s.className = 'src';
      const head = document.createElement('div');
      head.className = 'head';
      head.textContent = `${r.docTitle}  tok ${r.span}  cos=${Number(r.score).toFixed(3)}`;
      s.appendChild(head);
      const body = document.createElement('div');
      body.textContent = r.preview;
      s.appendChild(body);
      det.appendChild(s);
    }
    src.appendChild(det);
    msgEl.appendChild(src);
  }

  // ---- events ----------------------------------------------------
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    send(input.value);
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send(input.value);
    }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(200, input.scrollHeight) + 'px';
  });

  drop.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    for (const f of fileInput.files) uploadFile(f);
    fileInput.value = '';
  });
  drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('dragging');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('dragging'));
  drop.addEventListener('drop', async (e) => {
    e.preventDefault();
    drop.classList.remove('dragging');
    for (const f of e.dataTransfer.files) await uploadFile(f);
  });

  resetBtn.addEventListener('click', async () => {
    if (!confirm('Clear all uploaded documents and conversation history?')) return;
    await fetch('/reset', { method: 'POST' });
    log.innerHTML = '';
    addMsg('assistant', 'Reset. Drop new documents and start again.');
    await refreshStatus();
    toast('reset');
  });

  refreshHealth();
  refreshStatus();
})();
</script>
</body>
</html>
''';
