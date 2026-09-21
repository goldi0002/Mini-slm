/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelConfig } from '../types';
import { defaultTokenizer, SPECIAL_TOKENS } from './tokenizer';
import { SmallLanguageModel } from './transformer';
import { PREDEFINED_DATASETS, generateExpandedChatCorpus } from './datasets';

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
    id: 'assistant-48',
    name: 'Assistant-48 (Balanced Chat)',
    tagline: '2 Layers • 48-dim • 4 Heads • ~1.8 MB RAM • Fluent Daily Dialogue',
    vocabSize: defaultTokenizer.vocabSize,
    dModel: 48,
    nHeads: 4,
    nLayers: 2,
    dFfn: 96,
    maxSeqLen: 64,
    loraRank: 4,
    loraAlpha: 8,
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
    maxSeqLen: 48,
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
    maxSeqLen: 64,
    loraRank: 8,
    loraAlpha: 16,
  }
];

/**
 * Pre-seeds baseline conversational knowledge into the model so it behaves
 * like a pre-trained Conversational Small Language Model before user
 * fine-tuning is applied. Two layers of pre-training:
 *
 * 1. The statistical memory layer learns fluent conversational English from
 *    a dialogue corpus (this is what makes the base model chat coherently).
 * 2. The neural weights get a short warm-up so the forward pass is
 *    context-sensitive from the first message.
 */
export function initializePretrainedModel(config: ModelConfig): SmallLanguageModel {
  ensureVocabulary();
  const model = new SmallLanguageModel(config, defaultTokenizer);

  // Pre-seed natural conversational dialogues into the memory layer
  const conversationalCorpus = [
    `${SPECIAL_TOKENS.USER} hello ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} hello ! how are you doing today ?`,
    `${SPECIAL_TOKENS.USER} hi how are you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I am doing wonderful , thank you for asking ! how can I help you ?`,
    `${SPECIAL_TOKENS.USER} who are you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I am your friendly conversational AI assistant here to chat and help you .`,
    `${SPECIAL_TOKENS.USER} what can you do ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I can converse with you , share ideas , and be fine tuned on custom chat datasets .`,
    `${SPECIAL_TOKENS.USER} can you help me ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} certainly ! tell me what is on your mind and I will do my best to assist .`,
    `${SPECIAL_TOKENS.USER} thank you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} you are very welcome ! I am always delighted to chat with you .`,
    `${SPECIAL_TOKENS.USER} how does this work ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I run locally in your browser using a small causal transformer with low memory .`
  ];
  model.learnCorpus(conversationalCorpus, 1);

  // Warm up neural weights with a few quick conversational pre-training steps
  for (let epoch = 0; epoch < 3; epoch++) {
    for (const sample of conversationalCorpus) {
      const tokens = defaultTokenizer.encode(sample, true, true);
      model.trainStep(tokens, 0.08, false, 0.001);
    }
  }

  // Save the warm-up state (weights + memory) as the official Base Snapshot
  model.saveBaseSnapshot();

  return model;
}
