/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelConfig } from '../types';
import { defaultTokenizer, SPECIAL_TOKENS } from './tokenizer';
import { SmallLanguageModel } from './transformer';
import { PREDEFINED_DATASETS, generateExpandedChatCorpus } from './datasets';

const PRETRAIN_CORPUS: string[] = [
  `${SPECIAL_TOKENS.USER} hello ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} hello ! how are you doing today ?`,
  `${SPECIAL_TOKENS.USER} hi how are you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I am doing wonderful , thank you for asking ! how can I help you ?`,
  `${SPECIAL_TOKENS.USER} who are you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I am your friendly conversational AI assistant here to chat and help you .`,
  `${SPECIAL_TOKENS.USER} what can you do ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I can converse with you , share ideas , and be fine tuned on custom chat datasets .`,
  `${SPECIAL_TOKENS.USER} can you help me ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} certainly ! tell me what is on your mind and I will do my best to assist .`,
  `${SPECIAL_TOKENS.USER} thank you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} you are very welcome ! I am always delighted to chat with you .`,
  `${SPECIAL_TOKENS.USER} how does this work ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I run locally in your browser using a small causal transformer with low memory .`
];

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
    maxSeqLen: 128,
    loraRank: 16,
    loraAlpha: 32,
    displayParameterCount: 135_000_000,
    displayMemory: '~117–181 MB weights',
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

export function initializePretrainedModel(config: ModelConfig): SmallLanguageModel {
  ensureVocabulary();
  const modelConfig = { ...config, vocabSize: defaultTokenizer.vocabSize };
  const model = new SmallLanguageModel(modelConfig, defaultTokenizer);
  model.saveBaseSnapshot();
  return model;
}
