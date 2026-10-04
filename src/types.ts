/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface ModelConfig {
  id: string;
  name: string;
  tagline: string;
  vocabSize: number;
  dModel: number; // Embedding dimension
  nHeads: number; // Multi-head attention heads
  nLayers: number; // Transformer blocks
  dFfn: number; // Feed-forward hidden dimension
  maxSeqLen: number; // Maximum context length
  loraRank: number; // LoRA rank r
  loraAlpha: number; // LoRA scaling
  /** Optional real checkpoint size shown in the UI when this config is a shell. */
  displayParameterCount?: number;
  /** Optional downloaded weight size shown in the UI when this config is a shell. */
  displayMemory?: string;
}

export interface GenerationOptions {
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  maxNewTokens: number;
}

export interface GeneratedTokenInfo {
  token: string;
  id: number;
  prob: number;
  topCandidates: Array<{ token: string; id: number; prob: number }>;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  tokens?: GeneratedTokenInfo[];
  modelSource?: 'base' | 'finetuned';
  /** Local knowledge sources retrieved for this response. */
  knowledgeSources?: string[];
}

export interface TrainingHyperparams {
  epochs: number;
  learningRate: number;
  batchSize: number;
  weightDecay: number;
  loraMode: boolean; // true = LoRA, false = Full model fine-tuning
  loraRank: number;
}

export interface LossPoint {
  step: number;
  epoch: number;
  loss: number;
  perplexity: number;
  lr: number;
}

export interface TrainingState {
  isTraining: boolean;
  isPaused: boolean;
  currentEpoch: number;
  totalEpochs: number;
  currentStep: number;
  totalSteps: number;
  lossHistory: LossPoint[];
  currentLoss: number;
  currentPerplexity: number;
  /** Diagnostic NLL of the neural + memory blend; not the training objective. */
  currentBlendedLoss: number;
  /** Average assistant-response loss across the most recent completed epoch. */
  epochAverageLoss: number;
  /** True only when the measured training objective reaches TARGET_LOSS. */
  targetReached: boolean;
  sampleOutputs: Array<{ epoch: number; prompt: string; response: string }>;
}

export interface ConversationTurn {
  id: string;
  user: string;
  assistant: string;
  category?: string;
}

export interface DatasetPreset {
  id: string;
  name: string;
  description: string;
  iconName: string;
  badge: string;
  turns: ConversationTurn[];
}

export interface AttentionMapData {
  tokens: string[];
  matrix: number[][]; // [seq_len, seq_len]
  layer: number;
  head: number;
}
