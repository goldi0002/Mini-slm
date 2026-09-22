/**
 * diag_script.ts — verification harness for the SLM engine fixes.
 *
 * Verifies, end to end and without a browser:
 *   1. Vocabulary coverage: built-in corpora & datasets must encode without
 *      UNK tokens or single-letter char fallbacks (generation suppresses
 *      char tokens, so anything char-spelled can never be generated).
 *   2. Base model coherence: fluent, grammar-shaped output with no raw
 *      special tokens and no spelled-out characters.
 *   3. Fine-tuning reduces training loss across epochs, both on the dataset the
 *      studio trains by default and on the expanded/large training scales.
 *   4. Fine-tuning teaches dataset answers (post-train replies match the
 *      trained dataset, base replies do not), and user-added turns are learned.
 *   5. resetToBase() restores exact base behavior (weights + memory layer).
 *   6. Streaming API yields valid token info.
 *  10. Multi-turn chat keeps replying when history fills the context window.
 *
 * Run: bun scripts/diag_script.ts
 */

import { PREDEFINED_MODELS, initializePretrainedModel } from '../src/slm/predefinedModels';
import { PREDEFINED_DATASETS, generateExpandedChatCorpus } from '../src/slm/datasets';
import { defaultTokenizer, SPECIAL_TOKENS, UNK_ID, BOS_ID } from '../src/slm/tokenizer';
import { SmallLanguageModel, MIN_REPLY_TOKENS } from '../src/slm/transformer';
import { softmax } from '../src/slm/matrix';
import { GenerationOptions } from '../src/types';

// ---------------------------------------------------------------------------
// Reproducibility
// ---------------------------------------------------------------------------
// Evaluation has to be reproducible, so pin the RNG: it drives both weight
// initialisation and sampled decoding. The decoding configuration is the GREEDY
// and SAMPLED option objects below.

const RANDOM_SEED = 20240921;
let rngState = RANDOM_SEED;
Math.random = (): number => {
  rngState |= 0;
  rngState = (rngState + 0x6d2b79f5) | 0;
  let t = Math.imul(rngState ^ (rngState >>> 15), 1 | rngState);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const DIM = '\x1b[2m';
const RESET = '\x1b[0m';

let passed = 0;
let failed = 0;
const failures: string[] = [];

function check(label: string, ok: boolean, detail = ''): void {
  if (ok) {
    passed++;
    console.log(`  ${GREEN}PASS${RESET}  ${label}${detail ? ` ${DIM}(${detail})${RESET}` : ''}`);
  } else {
    failed++;
    failures.push(label);
    console.log(`  ${RED}FAIL${RESET}  ${label}${detail ? ` ${RED}(${detail})${RESET}` : ''}`);
  }
}

function section(title: string): void {
  console.log(`\n${CYAN}=== ${title} ===${RESET}`);
}

const words = (s: string): string[] =>
  s.toLowerCase().replace(/[^a-z0-9' ]/g, ' ').split(/\s+/).filter(Boolean);

/** Words of `text` that do not appear in `target` (novelty measure). */
function novelWords(text: string, target: string): string[] {
  const t = new Set(words(target));
  return [...new Set(words(text))].filter((w) => !t.has(w));
}

/** Fraction of target words covered by the generated text. */
function overlap(text: string, target: string): number {
  const got = new Set(words(text));
  const want = words(target);
  if (want.length === 0) return 0;
  let hit = 0;
  for (const w of want) if (got.has(w)) hit++;
  return hit / want.length;
}

const RAW_SPECIALS = ['<user>', '<assistant>', '<bos>', '<eos>', '<pad>', '<unk>', '\n', '\r'];

function textIssues(text: string): string[] {
  const issues: string[] = [];
  for (const s of RAW_SPECIALS) {
    if (text.includes(s)) issues.push(`raw special token ${JSON.stringify(s)}`);
  }
  // Any isolated single letter other than the real words "a" / "i" means the
  // model spelled out a character (OOV fallback leaked into generation).
  for (const w of words(text)) {
    if (w.length === 1 && w !== 'a' && w !== 'i') {
      issues.push(`spelled-out character "${w}"`);
      break;
    }
  }
  return issues;
}

const GREEDY: GenerationOptions = {
  temperature: 0.7,
  topK: 1, // deterministic sampling for reproducible assertions
  topP: 1.0,
  repetitionPenalty: 1.15,
  maxNewTokens: 30,
};

// Mirrors the chat playground's default sampling settings.
const SAMPLED: GenerationOptions = {
  temperature: 0.7,
  topK: 25,
  topP: 0.85,
  repetitionPenalty: 1.15,
  maxNewTokens: 40,
};

function chat(model: SmallLanguageModel, user: string, opts: GenerationOptions = GREEDY): string {
  const prompt = model.tokenizer.formatConversationPrompt(user);
  return model.generate(prompt, opts, true).text;
}

/**
 * Average blended loss (neural + memory mix, exactly like generation) on the
 * given turns WITHOUT letting the memory observe them — a clean eval metric.
 */
function evalLoss(model: SmallLanguageModel, turns: Array<{ user: string; assistant: string }>): number {
  const V = model.config.vocabSize;
  const neuralProbs = new Float32Array(V);
  const mixed = new Float32Array(V);
  const NEURAL_MIX = 0.08; // mirrors SmallLanguageModel.neuralMix
  let total = 0;
  let count = 0;
  for (const t of turns) {
    const text = `${SPECIAL_TOKENS.USER} ${t.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${t.assistant}`;
    const tokens = model.tokenizer.encode(text, true, true);
    const { logits, seqLen } = model.forward(tokens, true);
    for (let i = 0; i < seqLen - 1; i++) {
      const target = tokens[i + 1];
      if (target === 0) continue; // PAD
      const row = logits.subarray(i * V, (i + 1) * V);
      softmax(row, neuralProbs, 1.0);
      const prev1 = tokens[i];
      const prev2 = i >= 1 ? tokens[i - 1] : BOS_ID;
      model.memory.distribution(prev2, prev1, mixed);
      const p = NEURAL_MIX * neuralProbs[target] + (1 - NEURAL_MIX) * mixed[target];
      total += -Math.log(Math.max(1e-8, p));
      count++;
    }
  }
  return count > 0 ? total / count : 0;
}

// ---------------------------------------------------------------------------
// 1. Vocabulary coverage of built-in corpora & datasets
// ---------------------------------------------------------------------------

section('1. Vocabulary coverage (no UNK / no char-spelled words)');

const preset = PREDEFINED_DATASETS[0];
const datasetTexts = PREDEFINED_DATASETS.flatMap((d) => d.turns.flatMap((t) => [t.user, t.assistant]));
const expandedTexts = PREDEFINED_DATASETS.flatMap((d) =>
  generateExpandedChatCorpus(d, 100).flatMap((t) => [t.user, t.assistant])
);
const allCorpusTexts = [...datasetTexts, ...expandedTexts];

let unkCount = 0;
let charFallbackWords = new Set<string>();
for (const text of allCorpusTexts) {
  const ids = defaultTokenizer.encode(text, false, false);
  const idStr = (id: number) => defaultTokenizer.getTokenString(id);
  for (const id of ids) {
    if (id === UNK_ID) unkCount++;
    const s = idStr(id);
    if (/^[a-zA-Z]$/.test(s) && !['a', 'i'].includes(s.toLowerCase())) {
      charFallbackWords.add(s);
    }
  }
}
console.log(`  vocab size: ${defaultTokenizer.vocabSize}`);
check(
  'dataset texts encode with zero UNK tokens',
  unkCount === 0,
  `${unkCount} UNK tokens`
);
check(
  'dataset words are real vocabulary tokens (not char-spelled)',
  charFallbackWords.size === 0,
  charFallbackWords.size > 0
    ? `char-spelled: ${[...charFallbackWords].slice(0, 12).join(', ')}${charFallbackWords.size > 12 ? ' …' : ''}`
    : 'all covered'
);

// ---------------------------------------------------------------------------
// 2. Base model coherence
// ---------------------------------------------------------------------------

section('2. Base model coherence (pre fine-tuning)');

const model = initializePretrainedModel(PREDEFINED_MODELS[0]);
const baseMemorySize = model.memory.size; // baseline memory counts, pre fine-tuning

const basePrompts = [
  'hello who are you',
  'how are you doing today',
  'can you help me stay focused',
  'tell me something interesting',
];
const baseReplies: Record<string, string> = {};
let allClean = true;
let allSubstantial = true;
for (const p of basePrompts) {
  const reply = chat(model, p);
  baseReplies[p] = reply;
  const issues = textIssues(reply);
  if (issues.length > 0) allClean = false;
  if (words(reply).length < 4) allSubstantial = false;
  console.log(`  ${DIM}USER:${RESET}  ${p}`);
  console.log(`  ${DIM}BASE:${RESET}  ${reply || '(empty!)'}${issues.length ? `  ${YELLOW}[${issues.join(', ')}]${RESET}` : ''}`);
}
check('base replies contain no raw special tokens / spelled-out chars', allClean);
check('base replies are substantive sentences (>= 4 words)', allSubstantial);
check(
  'base model does not leak training-only phrases for unseen topics',
  !baseReplies['can you help me stay focused'].includes('twenty five minute'),
  'spot check'
);

// ---------------------------------------------------------------------------
// 3. Fine-tuning reduces training loss
// ---------------------------------------------------------------------------

section('3. Fine-tuning loss reduction (LoRA, the dataset the studio trains on)');

// The studio's default "Standard" scale trains on the selected dataset turns.
const standardTurns = preset.turns;
// NOTE: this is in-sample loss - these are the same turns training has just
// seen. The genuine held-out measurement is section 8.
const evalLossBefore = evalLoss(model, standardTurns);
console.log(`  eval loss before fine-tuning: ${evalLossBefore.toFixed(4)}`);
const epochLosses: number[] = [];
const EPOCHS = 6;
for (let epoch = 0; epoch < EPOCHS; epoch++) {
  let sum = 0;
  for (const turn of standardTurns) {
    const text = `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
    const tokens = model.tokenizer.encode(text, true, true);
    const { loss } = model.trainStep(tokens, 0.015, true, 0.005);
    sum += loss;
  }
  epochLosses.push(sum / standardTurns.length);
  console.log(`  epoch ${epoch + 1}: avg loss ${epochLosses[epoch].toFixed(4)}`);
}
const memoryAfterTraining = model.memory.size;
check(
  'training loss decreases from first to last epoch',
  epochLosses[EPOCHS - 1] < epochLosses[0] * 0.9,
  `${epochLosses[0].toFixed(3)} → ${epochLosses[EPOCHS - 1].toFixed(3)}`
);
check(
  'loss trend is broadly downward (last epoch is the minimum)',
  epochLosses[EPOCHS - 1] === Math.min(...epochLosses),
  `min=${Math.min(...epochLosses).toFixed(3)}`
);
const evalLossAfter = evalLoss(model, standardTurns);
console.log(`  eval loss after fine-tuning:  ${evalLossAfter.toFixed(4)}`);
check(
  'the turns it trained on are fitted (in-sample loss >= 3x better)',
  evalLossAfter < evalLossBefore / 3,
  `${evalLossBefore.toFixed(3)} → ${evalLossAfter.toFixed(3)}`
);

// The "Expanded"/"Large" training scales add generated dialogue. They must
// train just as cleanly; checked on a separate model so it cannot contaminate
// the dataset-answer assertions below.
const expandedModel = initializePretrainedModel(PREDEFINED_MODELS[1]);
const expandedTurns = generateExpandedChatCorpus(preset, 40);
let expandedFirst = 0;
let expandedLast = 0;
const EXPANDED_EPOCHS = 3;
for (let epoch = 0; epoch < EXPANDED_EPOCHS; epoch++) {
  let sum = 0;
  for (const turn of expandedTurns) {
    const text = `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
    const { loss } = expandedModel.trainStep(expandedModel.tokenizer.encode(text, true, true), 0.015, true, 0.005);
    sum += loss;
  }
  const avg = sum / expandedTurns.length;
  if (epoch === 0) expandedFirst = avg;
  expandedLast = avg;
  console.log(`  expanded scale epoch ${epoch + 1}: avg loss ${avg.toFixed(4)}`);
}
check(
  'expanded-scale training also reduces loss',
  expandedLast < expandedFirst * 0.9,
  `${expandedFirst.toFixed(3)} → ${expandedLast.toFixed(3)}`
);

// ---------------------------------------------------------------------------
// 4. Fine-tuned model answers with the trained dataset
// ---------------------------------------------------------------------------

section('4. Fine-tuned responses match the trained dataset');

const evalCases = preset.turns.slice(0, 3).map((t) => ({ user: t.user, target: t.assistant }));
let strongMatches = 0;
for (const c of evalCases) {
  const reply = chat(model, c.user);
  const ov = overlap(reply, c.target);
  const issues = textIssues(reply);
  if (ov >= 0.5 && issues.length === 0) strongMatches++;
  console.log(`  ${DIM}USER:${RESET}  ${c.user}`);
  console.log(`  ${DIM}TUNED:${RESET} ${reply || '(empty!)'}${issues.length ? `  ${YELLOW}[${issues.join(', ')}]${RESET}` : ''}`);
  console.log(`  ${DIM}target overlap:${RESET} ${(ov * 100).toFixed(0)}%`);
}
check(
  'majority of trained prompts reproduce their dataset answers (overlap >= 50%)',
  strongMatches >= Math.ceil(evalCases.length / 2),
  `${strongMatches}/${evalCases.length} strong matches`
);

// Distinctive knowledge test: base model never saw "morning routine" material.
const tunedRoutine = chat(model, 'what makes a good morning routine');
console.log(`  ${DIM}TUNED routine reply:${RESET} ${tunedRoutine}`);
check('fine-tuned model learned dataset-specific phrase ("routine")', tunedRoutine.includes('routine'));

// Custom (user-authored) dataset turns must be learnable too — simulates the
// DatasetManager "Add Conversation Turn" flow feeding the trainer.
const customTurns = [
  {
    user: 'what do you love',
    assistant: 'I love good music and calm water .',
  },
  {
    user: 'what music do you love',
    assistant: 'I love good music and calm water every day .',
  },
];
for (let e = 0; e < 5; e++) {
  for (const t of customTurns) {
    const text = `${SPECIAL_TOKENS.USER} ${t.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${t.assistant}`;
    model.trainStep(model.tokenizer.encode(text, true, true), 0.02, true, 0.005);
  }
}
const customReply = chat(model, 'what do you love');
console.log(`  ${DIM}TUNED custom reply:${RESET} ${customReply}`);
check('model learned a custom user-added phrase ("music")', customReply.includes('music'));

// ---------------------------------------------------------------------------
// 5. resetToBase restores exact base behavior
// ---------------------------------------------------------------------------

section('5. Reset-to-base restores weights + memory layer');

const resetPrompt = 'how are you doing today';
const tunedHi = chat(model, resetPrompt);
model.resetToBase();
const resetHi = chat(model, resetPrompt);
console.log(`  ${DIM}tuned:${RESET} ${tunedHi}`);
console.log(`  ${DIM}reset:${RESET} ${resetHi}`);
console.log(`  ${DIM}orig :${RESET} ${baseReplies[resetPrompt]}`);
check('fine-tuning changed the reply for a trained prompt', tunedHi !== baseReplies[resetPrompt]);
check('reset restores the exact original base reply (greedy, deterministic)', resetHi === baseReplies[resetPrompt]);
check('reset clears fine-tuned vocabulary from memory layer', model.memory.size === baseMemorySize, `size ${model.memory.size} vs base ${baseMemorySize} (post-training ${memoryAfterTraining})`);
check('reset removes dataset-specific phrase', !chat(model, 'what makes a good morning routine').includes('routine'));

// ---------------------------------------------------------------------------
// 6. Streaming API sanity
// ---------------------------------------------------------------------------

section('6. Streaming generation sanity');

const streamTokens: number[] = [];
let streamOk = true;
for await (const info of model.generateStream(
  model.tokenizer.formatConversationPrompt('how are you doing today'),
  SAMPLED,
  true
)) {
  streamTokens.push(info.id);
  if (!(info.prob > 0) || info.topCandidates.length === 0) streamOk = false;
  if (info.id === 3) break; // EOS
}
console.log(`  streamed ${streamTokens.length} tokens`);
check('generateStream yields valid token infos', streamOk && streamTokens.length > 0, `${streamTokens.length} tokens`);

// Sampled chat still clean after everything
const sampledReply = chat(model, 'thank you so much for chatting with me !', SAMPLED);
console.log(`  ${DIM}SAMPLED:${RESET} ${sampledReply}`);
check('sampled generation stays clean (no specials / spelled chars)', textIssues(sampledReply).length === 0);

// ---------------------------------------------------------------------------
// 7. Replies are complete: no one-token stubs, no clauses cut in half
// ---------------------------------------------------------------------------
// The memory layer ranks EOS highly right after a sentence-ending period, so
// unguarded sampling used to end some replies after a single token, producing a
// bare "." or one-word answer. Generation holds EOS back for the first
// MIN_REPLY_TOKENS tokens to prevent that.
//
// The opposite failure is what the stop rule in the memory layer fixes: a reply
// that rambles past the sentence it completed and is then cut mid-clause by the
// token budget. Both are measured here.

section('7. Sampled replies are complete (not stubs, not cut mid-clause)');

const stubPrompts = [
  'hello how are you doing today ?',
  'who are you and how can you help me ?',
  'I feel overwhelmed with my daily tasks',
  'can you give me advice on staying focused ?',
  'what makes a good morning routine ?',
  'thank you so much for chatting with me !',
];

let stubCount = 0;
let minReplyWords = Infinity;
let minReplyTokens = Infinity;
let shortestStub = '';
let unfinishedReplies = 0;
let shortestUnfinished = '';
const sampledReplies = stubPrompts.length * 6;
for (const p of stubPrompts) {
  for (let i = 0; i < 6; i++) {
    const out = model.generate(model.tokenizer.formatConversationPrompt(p), SAMPLED, true);
    const wordCount = words(out.text).length;
    minReplyWords = Math.min(minReplyWords, wordCount);
    minReplyTokens = Math.min(minReplyTokens, out.tokens.length);
    if (wordCount < 3) {
      stubCount++;
      shortestStub = out.text;
    }
    if (!/[.!?]$/.test(out.text.trim())) {
      unfinishedReplies++;
      shortestUnfinished = out.text;
    }
  }
}
if (shortestStub) {
  console.log(`  ${YELLOW}shortest stub:${RESET} ${JSON.stringify(shortestStub)}`);
}
if (shortestUnfinished) {
  console.log(`  ${YELLOW}no final punctuation:${RESET} ${JSON.stringify(shortestUnfinished)}`);
}
console.log(
  `  ${sampledReplies} sampled replies | shortest: ${minReplyWords} words / ${minReplyTokens} tokens | ${sampledReplies - unfinishedReplies} finish on a sentence boundary`
);
check(
  'no sampled reply collapses to a one-word answer',
  minReplyWords >= 2,
  `shortest reply ${minReplyWords} words`
);
check(
  'every reply runs past the EOS hold-back window before it can end',
  minReplyTokens >= MIN_REPLY_TOKENS,
  `shortest reply ${minReplyTokens} tokens (hold-back ${MIN_REPLY_TOKENS})`
);
check(
  'replies finish on a sentence boundary rather than being cut mid-clause',
  unfinishedReplies <= Math.ceil(sampledReplies * 0.15),
  `${unfinishedReplies}/${sampledReplies} replies without final punctuation`
);

// ---------------------------------------------------------------------------
// 8. Held-out generalisation (train / validation split)
// ---------------------------------------------------------------------------
// Section 3 trains and evaluates on the same turns, so its falling loss shows
// only that the model fits its training data. This section holds turns out of
// training entirely and measures the gap, which is what an "eval loss" claim
// actually needs to mean.

section('8. Held-out generalisation (train / validation split)');

const valIndices = [5, 6, 7];
const splitTrainTurns = preset.turns.filter((_, i) => !valIndices.includes(i));
const splitHeldOutTurns = preset.turns.filter((_, i) => valIndices.includes(i));
const otherDomainTurns = PREDEFINED_DATASETS[1].turns;

const splitModel = initializePretrainedModel(PREDEFINED_MODELS[1]);
const lossBefore = {
  inSample: evalLoss(splitModel, splitTrainTurns),
  heldOut: evalLoss(splitModel, splitHeldOutTurns),
  otherDomain: evalLoss(splitModel, otherDomainTurns),
};

const SPLIT_EPOCHS = 6;
for (let epoch = 0; epoch < SPLIT_EPOCHS; epoch++) {
  for (const turn of splitTrainTurns) {
    const text = `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
    splitModel.trainStep(splitModel.tokenizer.encode(text, true, true), 0.015, true, 0.005);
  }
}

const lossAfter = {
  inSample: evalLoss(splitModel, splitTrainTurns),
  heldOut: evalLoss(splitModel, splitHeldOutTurns),
  otherDomain: evalLoss(splitModel, otherDomainTurns),
};
const generalisationGap = lossAfter.heldOut - lossAfter.inSample;

console.log(
  `  split: ${splitTrainTurns.length} train / ${splitHeldOutTurns.length} held-out / ${otherDomainTurns.length} other-domain`
);
console.log(`  in-sample    ${lossBefore.inSample.toFixed(3)} -> ${lossAfter.inSample.toFixed(3)}`);
console.log(`  held-out     ${lossBefore.heldOut.toFixed(3)} -> ${lossAfter.heldOut.toFixed(3)}`);
console.log(`  other-domain ${lossBefore.otherDomain.toFixed(3)} -> ${lossAfter.otherDomain.toFixed(3)}`);
console.log(`  generalisation gap after fine-tuning: ${generalisationGap.toFixed(3)} nats`);
console.log(
  `  ${DIM}read this as: fine-tuning fits the turns it saw; it does not improve loss on\n  unseen ones, so answer quality for trained prompts comes from dialogue-case\n  replay in the memory layer, not from generalisation.${RESET}`
);

check(
  'fine-tuning fits its own training turns (in-sample loss at least halves)',
  lossAfter.inSample < lossBefore.inSample / 2,
  `${lossBefore.inSample.toFixed(3)} -> ${lossAfter.inSample.toFixed(3)}`
);
check(
  'held-out turns do not regress beyond a small tolerance',
  lossAfter.heldOut <= lossBefore.heldOut * 1.05,
  `${lossBefore.heldOut.toFixed(3)} -> ${lossAfter.heldOut.toFixed(3)}`
);
check(
  'the generalisation gap is measured rather than assumed away',
  generalisationGap > 0,
  `${generalisationGap.toFixed(3)} nats, reported above`
);

// ---------------------------------------------------------------------------
// 9. LoRA adapter gradients (finite differences)
// ---------------------------------------------------------------------------
// The adapter update used to be a hand-rolled heuristic: a single `lora_v_B`
// row, picked by `targetToken % dModel`, nudged by the loss gradient. Nothing
// tied that row to the loss, and `lora_q_A`/`lora_q_B` were never updated at
// all. Training now differentiates the cross-entropy for real, and this section
// is the guard on it: the analytic gradient must match a numerical one, or
// "fine-tuning" has quietly gone back to guessing.

section('9. Adapter gradients match numerical differentiation');

/** Mean cross-entropy of the neural distribution alone (no memory blend). */
function neuralLoss(model: SmallLanguageModel, tokens: number[]): number {
  const V = model.config.vocabSize;
  const probs = new Float32Array(V);
  const { logits, seqLen } = model.forward(tokens, true);
  let total = 0;
  let count = 0;
  for (let i = 0; i < seqLen - 1; i++) {
    const target = tokens[i + 1];
    if (target === 0) continue; // PAD
    const row = logits.subarray(i * V, (i + 1) * V);
    softmax(row, probs, 1.0);
    total += -Math.log(Math.max(1e-8, probs[target]));
    count++;
  }
  return count > 0 ? total / count : 0;
}

const gradModel = initializePretrainedModel(PREDEFINED_MODELS[0]);
const gradTokens = gradModel.tokenizer.encode(
  `${SPECIAL_TOKENS.USER} how are you feeling today ? ${SPECIAL_TOKENS.NEWLINE}` +
    `${SPECIAL_TOKENS.ASSISTANT} I am doing well , thank you for asking .`,
  true,
  true
);

// Sample adapter weights from every adapter matrix of every layer.
const gradSlots: Array<{ name: string; arr: Float32Array; idx: number }> = [];
for (let l = 0; l < gradModel.weights.layers.length; l++) {
  for (const key of ['lora_q_A', 'lora_q_B', 'lora_v_A', 'lora_v_B'] as const) {
    const arr = gradModel.weights.layers[l][key];
    for (const idx of [0, Math.floor(arr.length / 2)]) {
      gradSlots.push({ name: `layer${l}.${key}[${idx}]`, arr, idx });
    }
  }
}

const gradLoss = neuralLoss(gradModel, gradTokens);
const FD_EPS = 0.01;
// The forward pass is float32, so the loss is quantised at roughly loss * 2^-23
// and a central difference divides that granularity by 2*eps. Below this, a
// difference is unmeasurable rather than wrong.
const fdNoiseFloor = (gradLoss * Math.pow(2, -23)) / (2 * FD_EPS);
const numericGrad: number[] = [];
for (const slot of gradSlots) {
  const w = slot.arr[slot.idx];
  slot.arr[slot.idx] = w + FD_EPS;
  const up = neuralLoss(gradModel, gradTokens);
  slot.arr[slot.idx] = w - FD_EPS;
  const down = neuralLoss(gradModel, gradTokens);
  slot.arr[slot.idx] = w;
  numericGrad.push((up - down) / (2 * FD_EPS));
}

// Snapshot the frozen base weights, then run one lr=1 step: with no weight
// decay that applies exactly -gradient to every adapter.
const frozenProbes = [
  gradModel.weights.wte,
  gradModel.weights.lm_head,
  gradModel.weights.layers[0].q_proj,
  gradModel.weights.layers[0].fc1,
  gradModel.weights.ln_f_gamma
];
const frozenCopies = frozenProbes.map((w) => w.slice(0, 64));

const gradBefore = gradSlots.map((s) => s.arr[s.idx]);
gradModel.trainStep(gradTokens, 1.0, true, 0.0);
const analyticGrad = gradSlots.map((s, i) => gradBefore[i] - s.arr[s.idx]);

let gradWorstAbs = 0;
let gradWorstName = '';
let gradResolvable = 0;
let gradSignMismatches = 0;
for (let i = 0; i < gradSlots.length; i++) {
  const absErr = Math.abs(analyticGrad[i] - numericGrad[i]);
  if (absErr > gradWorstAbs) {
    gradWorstAbs = absErr;
    gradWorstName = gradSlots[i].name;
  }
  if (numericGrad[i] !== 0 && Math.sign(analyticGrad[i]) !== Math.sign(numericGrad[i])) {
    gradSignMismatches++;
  }
  if (Math.abs(numericGrad[i]) > 10 * fdNoiseFloor) gradResolvable++;
}
console.log(
  `  ${gradSlots.length} sampled adapter weights (${gradResolvable} resolvable above the float32 noise floor)`
);
console.log(
  `  worst absolute error ${gradWorstAbs.toExponential(2)} (${gradWorstName}), noise floor ${fdNoiseFloor.toExponential(2)}`
);
check(
  'analytic LoRA gradient matches numerical differentiation',
  gradWorstAbs < 5 * fdNoiseFloor,
  `worst ${gradWorstAbs.toExponential(2)} vs floor ${fdNoiseFloor.toExponential(2)}`
);
check(
  'analytic gradient points the same way as the numerical one',
  gradSignMismatches === 0,
  `${gradSignMismatches} sign mismatches out of ${gradSlots.length}`
);

// LoRA only trains the adapters, so the pretrained network must be untouched
// even after an aggressive lr=1 step.
const baseStillFrozen = frozenProbes.every((w, i) => {
  for (let j = 0; j < frozenCopies[i].length; j++) {
    if (w[j] !== frozenCopies[i][j]) return false;
  }
  return true;
});
check('LoRA training leaves every base weight untouched', baseStillFrozen);

// And the gradient has to be useful: the adapters must measurably fit the
// sequence they were trained on. This is the check that would have caught the
// old heuristic, which moved weights without tracking the loss.
const loraFitModel = initializePretrainedModel(PREDEFINED_MODELS[0]);
const loraFitTurns = preset.turns.slice(0, 3).map((t) =>
  loraFitModel.tokenizer.encode(
    `${SPECIAL_TOKENS.USER} ${t.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${t.assistant}`,
    true,
    true
  )
);
const loraFitBefore =
  loraFitTurns.reduce((sum, tokens) => sum + neuralLoss(loraFitModel, tokens), 0) / loraFitTurns.length;

// Snapshot the adapters so the check below is "these weights changed", not
// "these weights are non-zero" — the A matrices are randomly initialised, so a
// mere non-zero test would pass on a trainer that never touches them.
const adapterSnapshot = loraFitModel.weights.layers.map((layer) =>
  (['lora_q_A', 'lora_q_B', 'lora_v_A', 'lora_v_B'] as const).map((key) => layer[key].slice())
);

for (let epoch = 0; epoch < 6; epoch++) {
  for (const tokens of loraFitTurns) loraFitModel.trainStep(tokens, 0.05, true, 0.005);
}

const adapterKeys = ['lora_q_A', 'lora_q_B', 'lora_v_A', 'lora_v_B'] as const;
let movedMatrices = 0;
let totalMatrices = 0;
for (let l = 0; l < loraFitModel.weights.layers.length; l++) {
  for (let k = 0; k < adapterKeys.length; k++) {
    totalMatrices++;
    const after = loraFitModel.weights.layers[l][adapterKeys[k]];
    const before = adapterSnapshot[l][k];
    for (let i = 0; i < after.length; i++) {
      if (after[i] !== before[i]) {
        movedMatrices++;
        break;
      }
    }
  }
}
check(
  'every layer updates all four adapter matrices',
  movedMatrices === totalMatrices,
  `${movedMatrices}/${totalMatrices} matrices changed`
);
const loraFitAfter =
  loraFitTurns.reduce((sum, tokens) => sum + neuralLoss(loraFitModel, tokens), 0) / loraFitTurns.length;
console.log(`  neural in-sample loss ${loraFitBefore.toFixed(3)} -> ${loraFitAfter.toFixed(3)} (adapters only)`);
check(
  'the adapters measurably fit the turns they are trained on',
  loraFitAfter < loraFitBefore * 0.97,
  `${loraFitBefore.toFixed(3)} -> ${loraFitAfter.toFixed(3)}`
);

// ---------------------------------------------------------------------------
// 10. Multi-turn chat keeps replying (context window)
// ---------------------------------------------------------------------------
// The playground feeds conversation history back into the prompt. With a 64
// token window the raw prompt passes maxSeqLen after a couple of turns, and the
// generation loop exits before its first step — which is why the chat bubble
// went blank from the third message on. `generateChatStream` windows the prompt
// to the context (dropping whole oldest turns) and reserves room for the reply.

section('10. Multi-turn chat keeps replying inside the context window');

const chatModel = initializePretrainedModel(PREDEFINED_MODELS[0]);
const MAX_SEQ = chatModel.config.maxSeqLen;
const transcript: Array<{ role: 'user' | 'assistant'; content: string; tokens: number }> = [];
const chatTurns = [
  'hello there !',
  'how are you today ?',
  'what should i cook for dinner ?',
  'thanks !',
  'can you give me advice on staying focused ?',
  'what makes a good morning routine ?',
];

let blankTurns = 0;
let overLongPrompts = 0;
let uncleanReplies = 0;
let shortestReply = Infinity;

for (const text of chatTurns) {
  // History exactly as ChatPlayground builds it: earlier turns of the
  // conversation, capped to the last few messages.
  const history = transcript
    .filter((t) => t.content.trim().length > 0)
    .slice(-4)
    .map((t) => ({ role: t.role, content: t.content }));
  const prompt = chatModel.tokenizer.formatConversationPrompt(text, history);
  const promptTokens = chatModel.tokenizer.encode(prompt, true, false).length;
  if (promptTokens > MAX_SEQ) overLongPrompts++;

  const replyTokens: number[] = [];
  for await (const info of chatModel.generateChatStream(prompt, SAMPLED, true)) {
    replyTokens.push(info.id);
  }
  const reply = chatModel.tokenizer.decode(replyTokens, true);
  const issues = textIssues(reply);

  transcript.push({ role: 'user', content: text, tokens: promptTokens });
  transcript.push({ role: 'assistant', content: reply, tokens: replyTokens.length });

  shortestReply = Math.min(shortestReply, replyTokens.length);
  if (replyTokens.length === 0 || reply.trim().length === 0) blankTurns++;
  if (issues.length > 0) uncleanReplies++;

  console.log(`  ${DIM}USER:${RESET}  ${text}`);
  console.log(
    `  ${DIM}BOT :${RESET}  ${reply || '(empty!)'}${issues.length ? `  ${YELLOW}[${issues.join(', ')}]${RESET}` : ''}`
  );
  console.log(
    `  ${DIM}prompt ${promptTokens} tokens (raw window ${MAX_SEQ}), reply ${replyTokens.length} tokens${RESET}`
  );
}

console.log(`  raw prompts over the window: ${overLongPrompts}/${chatTurns.length}`);
check(
  'no chat turn comes back blank once history fills the context window',
  blankTurns === 0,
  `${blankTurns}/${chatTurns.length} blank, shortest reply ${shortestReply} tokens`
);
check(
  'the conversation really does outgrow the raw context window',
  overLongPrompts > 0,
  `${overLongPrompts}/${chatTurns.length} raw prompts over maxSeqLen ${MAX_SEQ}`
);
check(
  'windowed replies are substantive (never a one-token stub)',
  shortestReply >= MIN_REPLY_TOKENS,
  `shortest reply ${shortestReply} tokens (hold-back ${MIN_REPLY_TOKENS})`
);
check(
  'windowed replies stay clean (no raw specials / spelled chars)',
  uncleanReplies === 0,
  `${uncleanReplies} unclean replies`
);

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(`\n${CYAN}=== SUMMARY ===${RESET}`);
console.log(`  ${GREEN}${passed} passed${RESET}, ${RED}${failed} failed${RESET}`);
if (failed > 0) {
  console.log(`\n  ${RED}Failures:${RESET}`);
  for (const f of failures) console.log(`   - ${f}`);
  process.exitCode = 1;
} else {
  console.log(`  ${GREEN}SLM engine verification: ALL CHECKS PASSED${RESET}`);
}
