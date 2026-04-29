#!/usr/bin/env node
// Minimal localhost-only OpenAI Responses proxy for ResearchClaw.
// Purpose: ResearchClaw's stdlib urllib client expects non-streaming JSON, while
// the configured upstream requires `stream: true`. This proxy sends streaming
// requests upstream with Node fetch, aggregates response.output_text.delta events,
// and returns an OpenAI Responses-like JSON payload.

const http = require('http');
const fs = require('fs');

const PORT = Number(process.env.RESEARCHCLAW_PROXY_PORT || 18901);
const OPENCLAW_CONFIG = process.env.OPENCLAW_CONFIG || '/home/jcd/.openclaw/openclaw.json';
const cfg = JSON.parse(fs.readFileSync(OPENCLAW_CONFIG, 'utf8'));
const upstreamBase = cfg.models.providers.openai.baseUrl.replace(/\/$/, '');
const upstreamKey = cfg.models.providers.openai.apiKey;

function sendJson(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, {'content-type': 'application/json', 'content-length': Buffer.byteLength(body)});
  res.end(body);
}

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const text = Buffer.concat(chunks).toString('utf8');
  return text ? JSON.parse(text) : {};
}

function parseSseChunk(buffer, state) {
  state.buf += buffer;
  let idx;
  while ((idx = state.buf.indexOf('\n\n')) >= 0) {
    const raw = state.buf.slice(0, idx);
    state.buf = state.buf.slice(idx + 2);
    const lines = raw.split(/\r?\n/);
    let event = '';
    let dataLines = [];
    for (const line of lines) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    }
    const data = dataLines.join('\n');
    if (!data || data === '[DONE]') continue;
    let obj;
    try { obj = JSON.parse(data); } catch { continue; }
    if (obj.type === 'response.output_text.delta' && typeof obj.delta === 'string') {
      state.text += obj.delta;
    } else if (obj.type === 'response.completed' && obj.response) {
      state.completed = obj.response;
    } else if (obj.type === 'response.failed' && obj.response) {
      state.failed = obj.response;
    } else if ((event === 'response.completed') && obj.response) {
      state.completed = obj.response;
    }
  }
}

async function handleResponses(req, res) {
  const body = await readBody(req);
  body.stream = true;
  // This upstream currently requires streaming and rejects temperature for GPT-5.x.
  // ResearchClaw's generic OpenAI-compatible client sends temperature by default,
  // so strip it here for compatibility with the OpenClaw-configured provider.
  delete body.temperature;
  const r = await fetch(upstreamBase + '/responses', {
    method: 'POST',
    headers: {
      'authorization': 'Bearer ' + upstreamKey,
      'content-type': 'application/json',
      'user-agent': 'ResearchClaw-OpenClaw-Proxy/1.0'
    },
    body: JSON.stringify(body)
  });
  if (!r.ok) {
    const txt = await r.text();
    return sendJson(res, r.status, {error: {type: 'upstream_error', message: txt.slice(0, 2000)}});
  }
  const state = {buf: '', text: '', completed: null, failed: null};
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    parseSseChunk(decoder.decode(value, {stream: true}), state);
  }
  parseSseChunk(decoder.decode(), state);
  if (state.failed) {
    return sendJson(res, 500, {error: {type: 'upstream_failed', message: JSON.stringify(state.failed).slice(0, 2000)}});
  }
  const now = Math.floor(Date.now() / 1000);
  const response = state.completed || {};
  const id = response.id || ('resp_proxy_' + Date.now());
  const model = response.model || body.model || 'gpt-5.5';
  return sendJson(res, 200, {
    id,
    object: 'response',
    created_at: response.created_at || now,
    status: 'completed',
    model,
    output: [{
      id: 'msg_proxy_' + Date.now(),
      type: 'message',
      status: 'completed',
      role: 'assistant',
      content: [{type: 'output_text', text: state.text, annotations: []}]
    }],
    usage: response.usage || {input_tokens: 0, output_tokens: 0, total_tokens: 0}
  });
}

const server = http.createServer(async (req, res) => {
  try {
    if (req.method === 'GET' && req.url === '/v1/models') {
      return sendJson(res, 200, {object: 'list', data: [
        {id: 'gpt-5.5', object: 'model', owned_by: 'openclaw-proxy'},
        {id: 'gpt-5.4', object: 'model', owned_by: 'openclaw-proxy'}
      ]});
    }
    if (req.method === 'POST' && req.url === '/v1/responses') return await handleResponses(req, res);
    return sendJson(res, 404, {error: {message: 'not found'}});
  } catch (err) {
    return sendJson(res, 500, {error: {type: 'proxy_error', message: String(err && err.stack || err)}});
  }
});
server.listen(PORT, '127.0.0.1', () => {
  console.error(`ResearchClaw OpenClaw stream proxy listening on 127.0.0.1:${PORT}`);
});
