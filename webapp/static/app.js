'use strict';
const $ = (id) => document.getElementById(id);

const ACTIONS = {
  0: { name: 'Stand', cls: 'stand' },
  1: { name: 'Hit', cls: 'hit' },
  2: { name: 'Double', cls: 'double' },
  3: { name: 'Split', cls: 'split' },
};

async function api(url, body) {
  const opts = body !== undefined
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : {};
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
  return data;
}

/* ------------------------------- tabs -------------------------------- */
function showView(view) {
  document.body.dataset.view = view;
  $('view-casino').hidden = view !== 'casino';
  $('view-lab').hidden = view !== 'lab';
  $('tab-casino').classList.toggle('active', view === 'casino');
  $('tab-lab').classList.toggle('active', view === 'lab');
}
$('tab-casino').onclick = () => showView('casino');
$('tab-lab').onclick = () => showView('lab');

/* ------------------------------ models ------------------------------- */
let modelsCache = [];
async function refreshModels() {
  let models = [];
  try { models = await api('/api/models'); } catch (e) { /* server still up? ignore */ }
  modelsCache = models;
  for (const [sel, none] of [[$('play-model'), '— no advisor —'], [$('base-model'), '— train a new model —']]) {
    const prev = sel.value;
    sel.innerHTML = '';
    const opt0 = document.createElement('option');
    opt0.value = ''; opt0.textContent = none;
    sel.appendChild(opt0);
    for (const m of models) {
      const o = document.createElement('option');
      o.value = m.name;
      o.textContent = `${m.name} (${m.rules}, [${m.hidden_layers.join('·')}])`;
      sel.appendChild(o);
    }
    if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  }
}

/* ============================== CASINO =============================== */
const SUITS = ['♠', '♥', '♦', '♣'];
const TENS = ['10', 'J', 'Q', 'K'];
let game = null;
let seenCards = new Set();
let busy = false;

function getBankroll() { return parseInt(localStorage.getItem('betql_bankroll') || '100', 10); }
function setBankroll(v) { localStorage.setItem('betql_bankroll', String(v)); $('bankroll').textContent = v; }

function hash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

function cardFace(value, key) {
  const h = hash(key);
  const suit = SUITS[h % 4];
  let rank;
  if (value === 1) rank = 'A';
  else if (value === 10) rank = TENS[(h >>> 4) % 4];
  else rank = String(value);
  return { rank, suit, red: suit === '♥' || suit === '♦' };
}

function cardEl(value, key, { back = false, anim = null } = {}) {
  const el = document.createElement('div');
  el.className = 'card';
  if (back) {
    el.classList.add('back');
  } else {
    const f = cardFace(value, key);
    if (f.red) el.classList.add('red');
    el.innerHTML =
      `<span class="corner">${f.rank}<br>${f.suit}</span>` +
      `<span class="pip">${f.suit}</span>` +
      `<span class="corner br">${f.rank}<br>${f.suit}</span>`;
  }
  if (anim) el.classList.add(anim);
  return el;
}

function trackedCard(value, key, back) {
  const fresh = !seenCards.has(key);
  if (fresh) seenCards.add(key);
  return cardEl(value, key, { back, anim: fresh ? (back ? 'deal-in' : (key.includes('reveal') ? 'flip-in' : 'deal-in')) : null });
}

function showToast(msg) {
  const t = $('play-error');
  t.textContent = msg; t.hidden = false;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => { t.hidden = true; }, 3500);
}

function renderGame(state) {
  // dealer
  const dc = $('dealer-cards');
  dc.innerHTML = '';
  state.dealer.forEach((v, i) => {
    const key = state.done && i > 0 ? `${state.game_id}:dealer-reveal:${i}` : `${state.game_id}:dealer:${i}`;
    dc.appendChild(trackedCard(v, key, false));
  });
  if (state.dealer_hidden) dc.appendChild(trackedCard(0, `${state.game_id}:dealer-hole`, true));

  const ds = $('dealer-score');
  if (state.done) {
    ds.hidden = false;
    ds.textContent = state.dealer_score > 21 ? 'BUST' : state.dealer_score;
    ds.classList.toggle('bust', state.dealer_score > 21);
  } else { ds.hidden = true; }
  $('dealer-note').textContent =
    !state.done && state.rules === 'european' ? 'European rules — no hole card' : '';

  // player hands
  const ph = $('player-hands');
  ph.innerHTML = '';
  state.hands.forEach((hand, hi) => {
    const box = document.createElement('div');
    box.className = 'hand';
    if (!state.done && state.hands.length > 1 && hi === state.current) box.classList.add('current');

    const cards = document.createElement('div');
    cards.className = 'cards';
    hand.forEach((v, ci) => cards.appendChild(trackedCard(v, `${state.game_id}:p${hi}:${ci}:${v}`, false)));
    box.appendChild(cards);

    const tags = document.createElement('div');
    tags.className = 'hand-tags';
    const score = state.scores[hi];
    const pill = document.createElement('span');
    pill.className = 'score-pill' + (score > 21 ? ' bust' : '');
    pill.textContent = score > 21 ? 'BUST' : score;
    tags.appendChild(pill);
    if (state.doubled[hi]) {
      const t = document.createElement('span');
      t.className = 'tag doubled'; t.textContent = '2× BET';
      tags.appendChild(t);
    }
    if (state.done && state.outcomes) {
      const t = document.createElement('span');
      t.className = 'tag ' + state.outcomes[hi];
      t.textContent = state.outcomes[hi].toUpperCase();
      tags.appendChild(t);
    }
    box.appendChild(tags);
    ph.appendChild(box);
  });

  // actions + advisor
  const act = $('actions');
  act.innerHTML = '';
  const bubble = $('advisor-bubble');
  if (!state.done) {
    for (const a of state.valid_actions) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = `action-btn ${ACTIONS[a].cls}` + (a === state.suggestion ? ' suggested' : '');
      b.innerHTML = ACTIONS[a].name + (a === state.suggestion ? '<span class="mini">ADVISOR</span>' : '');
      b.onclick = () => playAction(a);
      act.appendChild(b);
    }
    if (state.suggestion !== null && state.suggestion !== undefined) {
      bubble.hidden = false;
      bubble.textContent = `🤖 ${state.advisor} suggests: ${ACTIONS[state.suggestion].name.toUpperCase()}`;
    } else { bubble.hidden = true; }
  } else { bubble.hidden = true; }
}

function showBanner(state) {
  const chips = Math.round(state.reward * 10);
  const title = $('banner-title');
  const blackjack = (state.outcomes || []).includes('blackjack');
  if (state.reward > 0) {
    title.textContent = blackjack ? '♠ BLACKJACK! ♠' : 'YOU WIN!';
    title.className = 'banner-title win';
    $('banner-sub').textContent = `+${chips} chips`;
    confetti();
  } else if (state.reward < 0) {
    title.textContent = 'DEALER WINS';
    title.className = 'banner-title lose';
    $('banner-sub').textContent = `${chips} chips`;
  } else {
    title.textContent = 'PUSH';
    title.className = 'banner-title push';
    $('banner-sub').textContent = 'Your bet is returned';
  }
  setBankroll(getBankroll() + chips);
  $('banner').hidden = false;
}

function confetti() {
  const box = $('confetti');
  const colors = ['#fcd34d', '#f87171', '#34d399', '#60a5fa', '#c084fc', '#fdfdf8'];
  for (let i = 0; i < 90; i++) {
    const s = document.createElement('span');
    s.style.left = Math.random() * 100 + '%';
    s.style.background = colors[i % colors.length];
    s.style.animationDuration = 1.6 + Math.random() * 1.6 + 's';
    s.style.animationDelay = Math.random() * 0.5 + 's';
    box.appendChild(s);
  }
  setTimeout(() => { box.innerHTML = ''; }, 4000);
}

async function deal() {
  if (busy) return;
  busy = true;
  try {
    $('banner').hidden = true;
    seenCards = new Set();
    game = await api('/api/game/new', {
      rules: $('play-rules').value,
      model: $('play-model').value || null,
    });
    renderGame(game);
  } catch (e) { showToast(e.message); }
  finally { busy = false; }
}

async function playAction(action) {
  if (!game || game.done || busy) return;
  busy = true;
  for (const b of $('actions').querySelectorAll('button')) b.disabled = true;
  try {
    game = await api(`/api/game/${game.game_id}/action`, { action });
    renderGame(game);
    if (game.done) setTimeout(() => showBanner(game), 700);
  } catch (e) { showToast(e.message); }
  finally { busy = false; }
}

$('btn-deal').onclick = deal;
$('btn-again').onclick = deal;
$('btn-reset-bank').onclick = () => setBankroll(100);

/* ================================ LAB ================================ */
let pollTimer = null;

function fmtTime(sec) {
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function setRunningUI(running) {
  $('btn-train').disabled = running;
  $('btn-stop').disabled = !running;
  for (const id of ['base-model', 'train-rules', 'layers', 'dropout', 'lr', 'steps', 'eval-episodes', 'model-name'])
    $(id).disabled = running || ($(id).dataset.lockedByBase === '1' && id !== 'base-model');
}

$('base-model').onchange = () => {
  const usingBase = !!$('base-model').value;
  if (usingBase) {
    // continuing a model: the architecture and rules are fixed by the
    // base model, so show its real values instead of whatever was typed
    const m = modelsCache.find((x) => x.name === $('base-model').value);
    if (m) {
      $('train-rules').value = m.rules;
      $('layers').value = m.hidden_layers.join(',');
      $('dropout').value = m.dropout !== null && m.dropout !== undefined ? m.dropout : 0.5;
    }
  }
  for (const id of ['train-rules', 'layers', 'dropout']) {
    $(id).disabled = usingBase;
    $(id).dataset.lockedByBase = usingBase ? '1' : '0';
  }
};

function drawChart(series) {
  const cv = $('chart'), ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const padL = 56, padR = 16, padT = 12, padB = 30;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);

  ctx.font = '11px ui-monospace, Consolas, monospace';
  if (!series || series.length === 0) {
    ctx.fillStyle = '#6e7681';
    ctx.textAlign = 'center';
    ctx.fillText('waiting for episode data…', W / 2, H / 2);
    return;
  }
  const xs = series.map((p) => p[0]), ys = series.map((p) => p[1]);
  const xMin = xs[0], xMax = Math.max(xs[xs.length - 1], xMin + 1);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  const span = Math.max(yMax - yMin, 0.1), padY = span * 0.15;
  yMin -= padY; yMax += padY;

  const X = (x) => padL + ((x - xMin) / (xMax - xMin)) * (W - padL - padR);
  const Y = (y) => padT + (1 - (y - yMin) / (yMax - yMin)) * (H - padT - padB);

  // grid + y labels
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const yv = yMin + (i / 4) * (yMax - yMin);
    const y = Y(yv);
    ctx.strokeStyle = '#21262d'; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = '#6e7681'; ctx.fillText(yv.toFixed(2), padL - 8, y);
  }
  // zero line
  if (yMin < 0 && yMax > 0) {
    ctx.strokeStyle = '#3d444d'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(W - padR, Y(0)); ctx.stroke();
    ctx.setLineDash([]);
  }
  // x labels
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let i = 0; i <= 4; i++) {
    const xv = Math.round(xMin + (i / 4) * (xMax - xMin));
    ctx.fillStyle = '#6e7681'; ctx.fillText(String(xv), X(xv), H - padB + 8);
  }
  ctx.fillText('episode', (padL + W - padR) / 2, H - 14);

  // line
  ctx.strokeStyle = '#2dd4bf'; ctx.lineWidth = 2;
  ctx.beginPath();
  series.forEach((p, i) => (i === 0 ? ctx.moveTo(X(p[0]), Y(p[1])) : ctx.lineTo(X(p[0]), Y(p[1]))));
  ctx.stroke();
  ctx.lineWidth = 1;

  // last point marker + value
  const last = series[series.length - 1];
  ctx.fillStyle = '#2dd4bf';
  ctx.beginPath(); ctx.arc(X(last[0]), Y(last[1]), 3.5, 0, Math.PI * 2); ctx.fill();
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  ctx.fillText(' ' + last[1].toFixed(3), Math.min(X(last[0]), W - 60), Y(last[1]));
}

function renderStatus(s) {
  const badge = $('phase-badge');
  badge.textContent = s.phase;
  badge.className = 'badge ' + s.phase;

  const pct = s.total_steps ? Math.min(100, (100 * s.step) / s.total_steps) : 0;
  $('progress-fill').style.width = pct + '%';
  let ptext = s.total_steps ? `${s.step.toLocaleString()} / ${s.total_steps.toLocaleString()} steps (${pct.toFixed(1)}%)` : '—';
  if (s.running && s.steps_per_sec > 0 && s.phase === 'training') {
    ptext += ` · ETA ${fmtTime((s.total_steps - s.step) / s.steps_per_sec)}`;
  }
  if (s.best) {
    ptext += ` · best ckpt ${s.best.mean.toFixed(3)} @ ${s.best.step.toLocaleString()}`;
  }
  $('progress-text').textContent = ptext;

  $('st-episodes').textContent = s.episodes ? s.episodes.toLocaleString() : '—';
  $('st-sps').textContent = s.steps_per_sec || '—';
  $('st-reward').textContent = s.recent_mean !== null && s.recent_mean !== undefined ? s.recent_mean.toFixed(3) : '—';
  $('st-elapsed').textContent = s.elapsed ? fmtTime(s.elapsed) : '—';

  drawChart(s.series);

  if (s.live_rates) {
    $('live-rates').hidden = false;
    $('live-n').textContent = `(last ${s.live_rates.episodes.toLocaleString()} training episodes, exploring policy)`;
    $('lrate-win').style.width = (s.live_rates.win_rate * 100) + '%';
    $('lrate-push').style.width = (s.live_rates.push_rate * 100) + '%';
    $('lrate-loss').style.width = (s.live_rates.loss_rate * 100) + '%';
    $('llg-win').textContent = (s.live_rates.win_rate * 100).toFixed(1) + '%';
    $('llg-push').textContent = (s.live_rates.push_rate * 100).toFixed(1) + '%';
    $('llg-loss').textContent = (s.live_rates.loss_rate * 100).toFixed(1) + '%';
  } else {
    $('live-rates').hidden = true;
  }

  if (s.eval) {
    $('eval-panel').hidden = false;
    $('eval-n').textContent = `(${s.eval.episodes.toLocaleString()} episodes, greedy policy)`;
    $('eval-mean').textContent = s.eval.mean.toFixed(4);
    $('eval-std').textContent = s.eval.sem !== undefined
      ? `${(1.96 * s.eval.sem).toFixed(4)} (95% CI) · per-hand std ${s.eval.std.toFixed(2)}`
      : s.eval.std.toFixed(4);
    $('rate-win').style.width = (s.eval.win_rate * 100) + '%';
    $('rate-push').style.width = (s.eval.push_rate * 100) + '%';
    $('rate-loss').style.width = (s.eval.loss_rate * 100) + '%';
    $('lg-win').textContent = (s.eval.win_rate * 100).toFixed(1) + '%';
    $('lg-push').textContent = (s.eval.push_rate * 100).toFixed(1) + '%';
    $('lg-loss').textContent = (s.eval.loss_rate * 100).toFixed(1) + '%';
    $('eval-saved').textContent = s.saved_as
      ? `✔ saved as weights/${s.saved_as}` +
        (s.used_best && s.best ? ` (best checkpoint, step ${s.best.step.toLocaleString()})` : '')
      : '';
  } else {
    $('eval-panel').hidden = true;
  }

  if (s.phase === 'error' && s.error) {
    $('train-error').hidden = false;
    $('train-error').textContent = s.error;
  }
}

async function poll() {
  let s;
  try { s = await api('/api/train/status'); } catch (e) { return; }
  renderStatus(s);
  setRunningUI(s.running);
  if (!s.running && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
    if (s.phase === 'done' || s.phase === 'stopped') refreshModels();
  }
}

function startPolling() {
  if (!pollTimer) pollTimer = setInterval(poll, 1000);
}

$('btn-train').onclick = async () => {
  $('train-error').hidden = true;
  const layers = $('layers').value.split(',').map((x) => parseInt(x.trim(), 10)).filter((x) => !isNaN(x));
  try {
    await api('/api/train/start', {
      rules: $('train-rules').value,
      layers,
      dropout: parseFloat($('dropout').value),
      lr: parseFloat($('lr').value),
      steps: parseInt($('steps').value, 10),
      eval_episodes: parseInt($('eval-episodes').value, 10),
      name: $('model-name').value.trim(),
      base_model: $('base-model').value || null,
    });
    setRunningUI(true);
    startPolling();
    poll();
  } catch (e) {
    $('train-error').hidden = false;
    $('train-error').textContent = e.message;
  }
};

$('btn-stop').onclick = async () => {
  try { await api('/api/train/stop', {}); } catch (e) { /* already finished */ }
};

/* ------------------------------- init -------------------------------- */
setBankroll(getBankroll());
refreshModels();
poll().then(() => {
  // resume live polling if a run is already in progress (e.g. page reload)
  api('/api/train/status').then((s) => { if (s.running) { setRunningUI(true); startPolling(); } }).catch(() => {});
});
