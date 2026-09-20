/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelConfig } from '../types';
import { defaultTokenizer, SPECIAL_TOKENS } from './tokenizer';
import { SmallLanguageModel } from './transformer';

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
 * Pre-seeds baseline conversational dialogue knowledge into model weights so it behaves
 * like a pre-trained Conversational Small Language Model before user fine-tuning is applied.
 */
export function initializePretrainedModel(config: ModelConfig): SmallLanguageModel {
  const model = new SmallLanguageModel(config, defaultTokenizer);

  // Pre-seed natural conversational dialogues
  const conversationalCorpus = [
    `${SPECIAL_TOKENS.USER} hello ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} hello ! how are you doing today ?`,
    `${SPECIAL_TOKENS.USER} hi how are you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I am doing wonderful , thank you for asking ! how can I help you ?`,
    `${SPECIAL_TOKENS.USER} who are you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I am your friendly conversational AI assistant here to chat and help you .`,
    `${SPECIAL_TOKENS.USER} what can you do ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I can converse with you , share ideas , and be fine tuned on custom chat datasets .`,
    `${SPECIAL_TOKENS.USER} can you help me ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} certainly ! tell me what is on your mind and I will do my best to assist .`,
    `${SPECIAL_TOKENS.USER} thank you ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} you are very welcome ! I am always delighted to chat with you .`,
    `${SPECIAL_TOKENS.USER} how does this work ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} I run locally in your browser using a small causal transformer with low memory .`
  ];

  // Warm up weights with quick conversational pre-training
  for (let epoch = 0; epoch < 6; epoch++) {
    for (const sample of conversationalCorpus) {
      const tokens = defaultTokenizer.encode(sample, true, true);
      model.trainStep(tokens, 0.08, false, 0.001);
    }
  }

  // Save the warm-up state as the official Base Snapshot
  model.saveBaseSnapshot();

  return model;
}
