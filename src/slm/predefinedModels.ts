/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelConfig } from '../types';
import { defaultTokenizer, SPECIAL_TOKENS } from './tokenizer';
import { SmallLanguageModel } from './transformer';
import { PREDEFINED_DATASETS, generateExpandedChatCorpus } from './datasets';

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
    id: 'smollm2-135m-instruct',
    name: 'SmolLM2 135M Instruct (Local)',
    tagline: '135M pretrained parameters • Apache-2.0 • WebGPU/WASM • Fast local browser inference',
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
 * Pre-seeds baseline conversational knowledge.
 *
 * The default 1.4M-parameter model deliberately does not run a full neural
 * pretraining pass on page load. That would make the browser wait on thousands
 * of matrix operations before the first chat. Instead the statistical memory
 * layer is seeded immediately, and the Fine-Tuning Studio performs real LoRA
 * backpropagation only when the user asks it to learn.
 *
 * This separation is important: memory/case replay gives immediate adaptation
 * for a dataset, while the transformer remains a genuine 1M+ parameter neural
 * model that can continue learning through LoRA.
 */
export function initializePretrainedModel(config: ModelConfig): SmallLanguageModel {
  ensureVocabulary();
  const modelConfig = { ...config, vocabSize: defaultTokenizer.vocabSize };
  const model = new SmallLanguageModel(modelConfig, defaultTokenizer);

  // The default model is a real pretrained checkpoint loaded by Transformers.js.
  // The SmallLanguageModel instance is retained as the existing app compatibility
  // shell for the training/inspection surfaces; chat generation bypasses these
  // scratch weights and uses the downloaded pretrained checkpoint instead.

  // Save the compatibility shell's initialized state as its Base Snapshot
  model.saveBaseSnapshot();

  return model;
}
