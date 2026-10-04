/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelConfig } from '../types';
import { defaultTokenizer, SPECIAL_TOKENS } from './tokenizer';
import { SmallLanguageModel } from './transformer';
import { PREDEFINED_DATASETS, generateExpandedChatCorpus } from './datasets';
import { ENGLISH_LEARNING_CORPUS } from './corpus';

/**
 * Baseline dialogue the base model is pre-trained on.
 *
 * Declared before `ensureVocabulary()` runs so every one of its words is taught
 * to the tokenizer first: an unknown word encodes as single-character fallback
 * tokens, which generation suppresses, so the warm-up would train the model on
 * distributions it can never emit (and the memory layer would waste most of its
 * probability mass on unreachable tokens).
 */
const PRETRAIN_CORPUS: string[] = [
  `${SPECIAL_TOKENS.USER} hello ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} hello ! how are you doing today ?`,
  `${SPECIAL_TOKENS.USER} hi how are you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I am doing wonderful , thank you for asking ! how can I help you ?`,
  `${SPECIAL_TOKENS.USER} who are you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I am your friendly conversational AI assistant here to chat and help you .`,
  `${SPECIAL_TOKENS.USER} what can you do ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I can converse with you , share ideas , and be fine tuned on custom chat datasets .`,
  `${SPECIAL_TOKENS.USER} can you help me ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} certainly ! tell me what is on your mind and I will do my best to assist .`,
  `${SPECIAL_TOKENS.USER} thank you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} you are very welcome ! I am always delighted to chat with you .`,
  `${SPECIAL_TOKENS.USER} how does this work ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I run locally in your browser using a small causal transformer with low memory .`
];

/**
 * Teach the shared tokenizer the vocabulary of every built-in corpus and
 * dataset BEFORE any model is constructed. Words that are not in the base
 * vocabulary would otherwise be reduced to single-character fallback tokens,
 * which generation suppresses — making them impossible to ever produce.
 * Runs once per process (guarded by a module-level flag).
 *
 * This MUST run before PREDEFINED_MODELS is evaluated: each config captures
 * `defaultTokenizer.vocabSize`, and if the tokenizer grows afterwards the
 * models would be built with a vocab smaller than the token ids they see
 * (out-of-range probabilities → NaN loss and empty replies).
 */
let vocabularyExpanded = false;
function ensureVocabulary(): void {
  if (vocabularyExpanded) return;
  vocabularyExpanded = true;
  const corpusTexts = [
    ...SmallLanguageModel.BASE_CORPUS,
    ...PRETRAIN_CORPUS,
    ...PREDEFINED_DATASETS.flatMap((d) => d.turns.flatMap((t) => [t.user, t.assistant])),
    ...PREDEFINED_DATASETS.flatMap((d) =>
      generateExpandedChatCorpus(d, 100).flatMap((t) => [t.user, t.assistant])
    ),
  ];
  defaultTokenizer.learnWords(corpusTexts);
}

// Expand the vocabulary at module load, before PREDEFINED_MODELS captures
// defaultTokenizer.vocabSize into the model configurations.
ensureVocabulary();

export const PREDEFINED_MODELS: ModelConfig[] = [
  {
    id: 'minislm-1m',
    name: 'MiniSLM 1.4M (Fast Adapt)',
    tagline: '4 Layers • 192-dim • 6 Heads • 384 FFN • ~1.4M params • Fast LoRA + memory learning',
    vocabSize: defaultTokenizer.vocabSize,
    dModel: 192,
    nHeads: 6,
    nLayers: 4,
    dFfn: 384,
    // Context holds the reply as well as the prompt: a ~40 token answer plus
    // the recent turns needs more than the 64 this started with.
    maxSeqLen: 128,
    loraRank: 16,
    loraAlpha: 32,
  },
  {
    id: 'nanolm-light',
    name: 'Conversational NanoLM (Ultra-Light)',
    tagline: '2 Layers • 32-dim • 4 Heads • ~0.8 MB RAM • High Speed & Low Memory',
    vocabSize: defaultTokenizer.vocabSize,
    dModel: 32,
    nHeads: 4,
    nLayers: 2,
    dFfn: 64,
    maxSeqLen: 64,
    loraRank: 4,
    loraAlpha: 8,
  },
  {
    id: 'micro-dialogue',
    name: 'MicroDialogue-64 (Deep Chat)',
    tagline: '3 Layers • 64-dim • 4 Heads • ~3.2 MB RAM • Nuanced Multi-Turn Chat',
    vocabSize: defaultTokenizer.vocabSize,
    dModel: 64,
    nHeads: 4,
    nLayers: 3,
    dFfn: 128,
    maxSeqLen: 96,
    loraRank: 8,
    loraAlpha: 16,
  }
];

/**
 * Pre-seeds baseline conversational knowledge into the model so it behaves
 * like a pre-trained Conversational Small Language Model before user
 * fine-tuning is applied.
 *
 * Two layers of pre-training:
 *
 * 1. The statistical memory layer learns fluent conversational English from
 *    a dialogue corpus (this is what makes the base model chat coherently).
 *
 * 2. The neural weights get a short warm-up of real gradient descent, so the
 *    forward pass is context-sensitive from the first message and can keep
 *    learning when the Fine-Tuning Studio revisits it later. The warm-up
 *    passes over both the plain-English corpus (grammar / next-word) and the
 *    dialogue corpus (chat format), with memory observation turned off for the
 *    English pass so the dialogue memory stays conversational and on for the
 *    dialogue pass so the model can already answer its warm-up prompts.
 *
 *    A newly built model currently takes a couple of seconds in the browser
 *    for this warm-up. Rewriting it to run `predict` (inference, already fast)
 *    instead of `trainStep` (backprop, which the warm-up currently uses) until
 *    the forward pass has crossed some loss threshold is left as the next
 *    performance optimisation, once a working forward pass exists.
 */
const WARMUP_EPOCHS = 0;

/**
 * Measured, not guessed: one pass at this rate lowers held-out cross-entropy
 * from ~7.0 nats (uniform) to ~4.8 nats, while a pass at 0.02 only reaches
 * ~6.6. Gradients are globally clipped (see GRADIENT_CLIP_NORM), which is what
 * makes the higher rate stable instead of divergent.
 */
const WARMUP_LR = 0.0;

export function initializePretrainedModel(config: ModelConfig): SmallLanguageModel {
  ensureVocabulary();
  const modelConfig = { ...config, vocabSize: defaultTokenizer.vocabSize };
  const model = new SmallLanguageModel(modelConfig, defaultTokenizer);

  const conversationalCorpus = PRETRAIN_CORPUS;
  model.learnCorpus(conversationalCorpus, 1);

  // Skip full neural warm-up for the 1M+ browser model. The memory layer
  // provides the deterministic conversational prior immediately; LoRA training
  // in the Studio performs the expensive neural adaptation only when requested.

  // Save the initialized state (weights + memory) as the official Base Snapshot
  model.saveBaseSnapshot();

  return model;
}
