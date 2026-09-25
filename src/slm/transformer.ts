/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelConfig, GenerationOptions, GeneratedTokenInfo } from '../types';
import { Tokenizer, defaultTokenizer, BOS_ID, EOS_ID, PAD_ID, UNK_ID, USER_ID, ASSISTANT_ID, NEWLINE_ID } from './tokenizer';
import { NgramLanguageModel } from './ngram';
import { BASE_CORPUS } from './corpus';
import {
  createFloat32Matrix,
  createRandomNormalMatrix,
  layerNorm,
  layerNormBackward,
  gelu,
  geluDerivative,
  softmax,
  sampleFromDistribution
} from './matrix';

/**
 * How many tokens a reply must produce before it is allowed to stop.
 *
 * The memory layer legitimately ranks EOS right after a sentence-ending period,
 * and sampling would occasionally take it after a single token — producing a
 * bare "." or one-word answer. Generation is already bounded by maxNewTokens,
 * so holding EOS back for the first few tokens only ever turns a stub into a
 * sentence.
 */
export const MIN_REPLY_TOKENS = 6;

export interface AttentionLayerWeights {
  q_proj: Float32Array; // [dModel, dModel]
  k_proj: Float32Array;
  v_proj: Float32Array;
  out_proj: Float32Array;

  // LoRA Adapters for Q and V
  lora_q_A: Float32Array; // [loraRank, dModel]
  lora_q_B: Float32Array; // [dModel, loraRank]
  lora_v_A: Float32Array;
  lora_v_B: Float32Array;

  ln1_gamma: Float32Array;
  ln1_beta: Float32Array;

  fc1: Float32Array; // [dFfn, dModel]
  fc1_b: Float32Array;
  fc2: Float32Array; // [dModel, dFfn]
  fc2_b: Float32Array;

  ln2_gamma: Float32Array;
  ln2_beta: Float32Array;
}

export interface TransformerWeights {
  wte: Float32Array; // [vocabSize, dModel]
  wpe: Float32Array; // [maxSeqLen, dModel]
  layers: AttentionLayerWeights[];
  ln_f_gamma: Float32Array;
  ln_f_beta: Float32Array;
  lm_head: Float32Array; // [vocabSize, dModel]
}

/**
 * Per-layer activations retained by a forward pass so the LoRA backward pass can
 * differentiate the real network instead of a re-derived approximation of it.
 * Allocated only when training asks for it.
 */
interface ActivationCache {
  blockInput: Float32Array[]; // residual stream entering each block
  norm1: Float32Array[]; // LN1 output
  q: Float32Array[];
  k: Float32Array[];
  v: Float32Array[];
  x1: Float32Array[]; // residual stream after the attention block
  norm2: Float32Array[]; // LN2 output
  finalResidual: Float32Array; // residual stream entering the final LayerNorm
}

/** Copy `count` floats, optionally at an offset into either side. */
function copyInto(
  dst: Float32Array,
  src: Float32Array,
  count: number,
  dstOffset = 0,
  srcOffset = 0
): void {
  for (let i = 0; i < count; i++) dst[dstOffset + i] = src[srcOffset + i];
}

export class SmallLanguageModel {
  public config: ModelConfig;
  public weights: TransformerWeights;
  public tokenizer: Tokenizer;

  // High-performance Flat TypedArray Snapshot for zero-heap-thrash resets
  private baseWeightsSnapshot: Float32Array | null = null;
  private totalWeightFloats = 0;

  // Reusable scratch buffers to guarantee low, constant memory footprint
  private scratchX: Float32Array;
  private scratchXNorm1: Float32Array;
  private scratchQ: Float32Array;
  private scratchK: Float32Array;
  private scratchV: Float32Array;
  private scratchAttnOut: Float32Array;
  private scratchProjOut: Float32Array;
  private scratchXNorm2: Float32Array;
  private scratchMid: Float32Array;
  private scratchFinalNorm: Float32Array;
  private scratchLogits: Float32Array;

  // Rank-sized scratch for the LoRA low-rank projections. u = A·x depends only
  // on the token position and the input, never on the output row, so it is
  // computed once per position instead of once per row (see forward()).
  private scratchLoraQ: Float32Array;
  private scratchLoraV: Float32Array;

  // Last computed attention maps for architecture inspector: [layer][head][seq_len, seq_len]
  public lastAttentionMaps: number[][][][] = [];

  // A token id outside the vocabulary means the tokenizer and the model
  // disagree; report it once instead of silently folding every bad id onto the
  // last token. See forward().
  private vocabMismatchWarned = false;

  // Statistical memory layer: a trigram language model with backoff that is
  // blended with the neural logits. A toy in-browser transformer cannot learn
  // fluent English from scratch, so this memory guarantees natural dialogue,
  // while fine-tuning reinforces it with dataset-specific phrases.
  public memory: NgramLanguageModel;
  private baseMemorySnapshot: ReturnType<NgramLanguageModel['snapshot']> | null = null;

  // Last word of the user's message that the upcoming generated token must
  // answer. Set when a prompt is encoded and consumed by the first generated
  // token, so only the reply opening is conditioned on the user's question.
  private pendingReplyWord: number | null = null;

  // Dialogue-case replay: the answer fine-tuning learned for the encoded user
  // message, plus how far into it generation has stayed on track. Cleared as
  // soon as generation diverges from the learned answer.
  private caseReply: number[] | null = null;
  private casePos = 0;

  // Tokens produced for the current reply, used to hold back a premature EOS.
  private generatedCount = 0;

  // Set once a full retrain (loraMode: false) has run on this model. A full
  // retrain updates lm_head and leaves the LoRA adapters at their zero
  // initialization, so the adapters alone cannot tell whether the model was
  // fine-tuned. Cleared by resetToBase().
  private fullFineTuneApplied = false;

  // Activation cache and gradient scratch for LoRA training, both allocated on
  // first use so inference-only paths never pay for them.
  private activationCache: ActivationCache | null = null;
  private gradScratchReady = false;
  private gNorm!: Float32Array; // dL/d(final LayerNorm output)
  private gResidual!: Float32Array; // dL/d(block output)
  private gX1!: Float32Array; // dL/d(residual after attention)
  private gIn!: Float32Array; // dL/d(block input)
  private gProjOut!: Float32Array;
  private gAttnOut!: Float32Array;
  private gQ!: Float32Array;
  private gK!: Float32Array;
  private gV!: Float32Array;
  private gXNorm1!: Float32Array;
  private gXNorm2Row!: Float32Array;
  private gFfnPre!: Float32Array;
  private gProbs!: Float32Array;
  private gAttnRow!: Float32Array;
  private dLora_qA!: Float32Array;
  private dLora_qB!: Float32Array;
  private dLora_vA!: Float32Array;
  private dLora_vB!: Float32Array;
  private adapterRow!: Float32Array;

  // Share of the final sampling distribution that comes from the neural
  // forward pass; the remainder comes from the statistical memory layer.
  private neuralMix = 0.08;

  constructor(config: ModelConfig, tokenizer: Tokenizer = defaultTokenizer) {
    // The embedding table, LM head, and memory layer must cover every token the
    // tokenizer can emit. If the vocabulary grew after this config was captured
    // (e.g. a custom dataset taught new words) the model would otherwise index
    // past the end of its probability arrays — silently dropping mass and
    // producing NaN loss. Widen the vocab instead.
    this.config = { ...config, vocabSize: Math.max(config.vocabSize, tokenizer.vocabSize) };
    this.tokenizer = tokenizer;

    // Allocate weights
    this.weights = this.initWeights();

    // Allocate reusable scratch buffers sized to maxSeqLen
    const maxT = this.config.maxSeqLen;
    const d = this.config.dModel;
    const ffn = this.config.dFfn;
    const v = this.config.vocabSize;

    this.scratchX = new Float32Array(maxT * d);
    this.scratchXNorm1 = new Float32Array(maxT * d);
    this.scratchQ = new Float32Array(maxT * d);
    this.scratchK = new Float32Array(maxT * d);
    this.scratchV = new Float32Array(maxT * d);
    this.scratchAttnOut = new Float32Array(maxT * d);
    this.scratchProjOut = new Float32Array(maxT * d);
    this.scratchXNorm2 = new Float32Array(maxT * d);
    this.scratchMid = new Float32Array(ffn);
    this.scratchFinalNorm = new Float32Array(maxT * d);
    this.scratchLogits = new Float32Array(maxT * v);
    const rank = Math.max(1, this.config.loraRank);
    this.scratchLoraQ = new Float32Array(rank);
    this.scratchLoraV = new Float32Array(rank);

    // Initialize the statistical memory layer with baseline conversational English.
    // The EOS id lets it learn how utterances end, so a reply can stop at a
    // sentence boundary instead of being cut off by the token budget.
    this.memory = new NgramLanguageModel(v, EOS_ID);
    this.seedMemoryCorpus();

    // Save baseline snapshot in a single compact TypedArray
    this.saveBaseSnapshot();
  }

  private initWeights(): TransformerWeights {
    const { vocabSize, dModel, maxSeqLen, nLayers, dFfn, loraRank } = this.config;
    const std = 0.03;

    const wte = createRandomNormalMatrix(vocabSize, dModel, std);
    const wpe = createRandomNormalMatrix(maxSeqLen, dModel, std);

    let floatCount = wte.length + wpe.length;

    const layers: AttentionLayerWeights[] = [];
    for (let l = 0; l < nLayers; l++) {
      const q_proj = createRandomNormalMatrix(dModel, dModel, std);
      const k_proj = createRandomNormalMatrix(dModel, dModel, std);
      const v_proj = createRandomNormalMatrix(dModel, dModel, std);
      const out_proj = createRandomNormalMatrix(dModel, dModel, std);

      // LoRA initialization: A is random normal, B is initialized to zeros
      const lora_q_A = createRandomNormalMatrix(loraRank, dModel, 0.02);
      const lora_q_B = createFloat32Matrix(dModel, loraRank, 0);
      const lora_v_A = createRandomNormalMatrix(loraRank, dModel, 0.02);
      const lora_v_B = createFloat32Matrix(dModel, loraRank, 0);

      const ln1_gamma = createFloat32Matrix(1, dModel, 1.0);
      const ln1_beta = createFloat32Matrix(1, dModel, 0.0);

      const fc1 = createRandomNormalMatrix(dFfn, dModel, std);
      const fc1_b = createFloat32Matrix(1, dFfn, 0.0);
      const fc2 = createRandomNormalMatrix(dModel, dFfn, std);
      const fc2_b = createFloat32Matrix(1, dModel, 0.0);

      const ln2_gamma = createFloat32Matrix(1, dModel, 1.0);
      const ln2_beta = createFloat32Matrix(1, dModel, 0.0);

      floatCount += q_proj.length + k_proj.length + v_proj.length + out_proj.length;
      floatCount += lora_q_A.length + lora_q_B.length + lora_v_A.length + lora_v_B.length;
      floatCount += ln1_gamma.length + ln1_beta.length;
      floatCount += fc1.length + fc1_b.length + fc2.length + fc2_b.length;
      floatCount += ln2_gamma.length + ln2_beta.length;

      layers.push({
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        lora_q_A,
        lora_q_B,
        lora_v_A,
        lora_v_B,
        ln1_gamma,
        ln1_beta,
        fc1,
        fc1_b,
        fc2,
        fc2_b,
        ln2_gamma,
        ln2_beta
      });
    }

    const ln_f_gamma = createFloat32Matrix(1, dModel, 1.0);
    const ln_f_beta = createFloat32Matrix(1, dModel, 0.0);
    const lm_head = createRandomNormalMatrix(vocabSize, dModel, std);

    floatCount += ln_f_gamma.length + ln_f_beta.length + lm_head.length;
    this.totalWeightFloats = floatCount;

    return {
      wte,
      wpe,
      layers,
      ln_f_gamma,
      ln_f_beta,
      lm_head
    };
  }

  /**
   * Baseline conversational corpus for the statistical memory layer, written
   * against the tokenizer vocabulary so the base model is already fluent.
   *
   * The text itself lives in `./corpus` because the tokenizer learns its BPE
   * merges from it at construction time, and the tokenizer must not import the
   * transformer (that would be a circular module dependency).
   */
  public static readonly BASE_CORPUS: string[] = BASE_CORPUS;

  private seedMemoryCorpus(): void {
    for (const seq of SmallLanguageModel.BASE_CORPUS) {
      this.memory.observe(this.tokenizer.encode(seq, true, true), 1);
    }
  }

  /**
   * Teach the statistical memory layer new conversational material
   * (used by fine-tuning and pre-training so new phrases become fluent).
   */
  public learnCorpus(texts: string[], weight = 1): void {
    for (const text of texts) {
      this.memory.observe(this.tokenizer.encode(text, true, true), weight);
    }
  }

  /**
   * Fit an encoded prompt inside the model's context window, keeping the most
   * recent turns.
   *
   * Generation keeps appending tokens to the prompt, so the prompt itself has
   * to leave room for the reply. A multi-turn chat prompt that reached
   * maxSeqLen makes the generation loop exit before its first step and produce
   * nothing at all — which is what left the chat bubble blank once the
   * conversation history grew. Oldest turns are dropped whole so the context
   * still reads as clean user/assistant turns, and the final question is always
   * kept.
   */
  private fitPromptToContext(tokens: number[], reserve: number): number[] {
    const budget = Math.max(8, this.config.maxSeqLen - reserve);
    if (tokens.length <= budget) return tokens;

    // Turn boundaries: every prompt segment starts at a <user> or <assistant> tag.
    const starts: number[] = [];
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i] === USER_ID || tokens[i] === ASSISTANT_ID) starts.push(i);
    }

    let cut = 0;
    for (let i = starts.length - 1; i >= 0; i--) {
      if (tokens.length - starts[i] > budget) break;
      cut = starts[i];
    }

    const kept = cut > 0 ? tokens.slice(cut) : tokens;
    // A single turn longer than the whole window: keep its tail, because a
    // causal model only conditions on what comes last.
    return kept.length > budget ? kept.slice(kept.length - budget) : kept;
  }

  /**
   * Window a chat prompt down to the context the model can consume, leaving
   * `reserve` tokens for the reply itself. Prompts that already fit are
   * returned untouched.
   */
  private windowPrompt(prompt: string, reserve: number): string {
    const budget = Math.max(8, this.config.maxSeqLen - reserve);
    const tokens = this.tokenizer.encode(prompt, true, false);
    if (tokens.length <= budget) return prompt;

    // Decoding the kept ids (special tokens included) back to text keeps the
    // <user>/<assistant> tags, so the re-encoded prompt is the same
    // conversation minus its oldest turns.
    return this.tokenizer.decode(this.fitPromptToContext(tokens, reserve), false);
  }

  /**
   * Streaming generation for conversations.
   *
   * Identical to `generateStream`, except the prompt is first fitted to the
   * context window (see `windowPrompt`). Use this whenever history accumulates:
   * a prompt at or past maxSeqLen yields no tokens at all.
   */
  public async *generateChatStream(
    prompt: string,
    options: GenerationOptions,
    useLora = true
  ): AsyncGenerator<GeneratedTokenInfo> {
    // The two tokens reserved on top of the reply are the reply-opening context
    // `encodeForGeneration` appends.
    const tokens = this.encodeForGeneration(this.windowPrompt(prompt, options.maxNewTokens + 2));

    for (let step = 0; step < options.maxNewTokens; step++) {
      const tokenInfo = this.generateNextToken(tokens, options, useLora);
      tokens.push(tokenInfo.id);

      yield tokenInfo;

      if (tokenInfo.id === EOS_ID) break;

      await new Promise((r) => setTimeout(r, 18));
    }
  }

  /**
   * Save all weights to a single flat Float32Array.
   * Total memory is only ~150-400 KB with ZERO JSON string allocations.
   */
  public saveBaseSnapshot(): void {
    if (!this.baseWeightsSnapshot || this.baseWeightsSnapshot.length !== this.totalWeightFloats) {
      this.baseWeightsSnapshot = new Float32Array(this.totalWeightFloats);
    }

    let offset = 0;
    this.baseWeightsSnapshot.set(this.weights.wte, offset);
    offset += this.weights.wte.length;

    this.baseWeightsSnapshot.set(this.weights.wpe, offset);
    offset += this.weights.wpe.length;

    for (const l of this.weights.layers) {
      this.baseWeightsSnapshot.set(l.q_proj, offset); offset += l.q_proj.length;
      this.baseWeightsSnapshot.set(l.k_proj, offset); offset += l.k_proj.length;
      this.baseWeightsSnapshot.set(l.v_proj, offset); offset += l.v_proj.length;
      this.baseWeightsSnapshot.set(l.out_proj, offset); offset += l.out_proj.length;
      this.baseWeightsSnapshot.set(l.lora_q_A, offset); offset += l.lora_q_A.length;
      this.baseWeightsSnapshot.set(l.lora_q_B, offset); offset += l.lora_q_B.length;
      this.baseWeightsSnapshot.set(l.lora_v_A, offset); offset += l.lora_v_A.length;
      this.baseWeightsSnapshot.set(l.lora_v_B, offset); offset += l.lora_v_B.length;
      this.baseWeightsSnapshot.set(l.ln1_gamma, offset); offset += l.ln1_gamma.length;
      this.baseWeightsSnapshot.set(l.ln1_beta, offset); offset += l.ln1_beta.length;
      this.baseWeightsSnapshot.set(l.fc1, offset); offset += l.fc1.length;
      this.baseWeightsSnapshot.set(l.fc1_b, offset); offset += l.fc1_b.length;
      this.baseWeightsSnapshot.set(l.fc2, offset); offset += l.fc2.length;
      this.baseWeightsSnapshot.set(l.fc2_b, offset); offset += l.fc2_b.length;
      this.baseWeightsSnapshot.set(l.ln2_gamma, offset); offset += l.ln2_gamma.length;
      this.baseWeightsSnapshot.set(l.ln2_beta, offset); offset += l.ln2_beta.length;
    }

    this.baseWeightsSnapshot.set(this.weights.ln_f_gamma, offset); offset += this.weights.ln_f_gamma.length;
    this.baseWeightsSnapshot.set(this.weights.ln_f_beta, offset); offset += this.weights.ln_f_beta.length;
    this.baseWeightsSnapshot.set(this.weights.lm_head, offset); offset += this.weights.lm_head.length;

    // Snapshot the statistical memory layer together with the weights so
    // reset-to-base restores the exact pre-fine-tuning language behavior.
    this.baseMemorySnapshot = this.memory.snapshot();

    // The snapshot *is* the base model: any training that happened before it
    // (e.g. the pre-training warm-up) counts as base, and resetToBase() returns
    // the weights to exactly this state — so no adaptation is outstanding.
    this.fullFineTuneApplied = false;
  }

  /**
   * Reset weights from the flat base snapshot in milliseconds with zero GC
   */
  public resetToBase(): void {
    if (!this.baseWeightsSnapshot) return;

    let offset = 0;
    this.weights.wte.set(this.baseWeightsSnapshot.subarray(offset, offset + this.weights.wte.length));
    offset += this.weights.wte.length;

    this.weights.wpe.set(this.baseWeightsSnapshot.subarray(offset, offset + this.weights.wpe.length));
    offset += this.weights.wpe.length;

    for (const l of this.weights.layers) {
      l.q_proj.set(this.baseWeightsSnapshot.subarray(offset, offset + l.q_proj.length)); offset += l.q_proj.length;
      l.k_proj.set(this.baseWeightsSnapshot.subarray(offset, offset + l.k_proj.length)); offset += l.k_proj.length;
      l.v_proj.set(this.baseWeightsSnapshot.subarray(offset, offset + l.v_proj.length)); offset += l.v_proj.length;
      l.out_proj.set(this.baseWeightsSnapshot.subarray(offset, offset + l.out_proj.length)); offset += l.out_proj.length;
      l.lora_q_A.set(this.baseWeightsSnapshot.subarray(offset, offset + l.lora_q_A.length)); offset += l.lora_q_A.length;
      l.lora_q_B.set(this.baseWeightsSnapshot.subarray(offset, offset + l.lora_q_B.length)); offset += l.lora_q_B.length;
      l.lora_v_A.set(this.baseWeightsSnapshot.subarray(offset, offset + l.lora_v_A.length)); offset += l.lora_v_A.length;
      l.lora_v_B.set(this.baseWeightsSnapshot.subarray(offset, offset + l.lora_v_B.length)); offset += l.lora_v_B.length;
      l.ln1_gamma.set(this.baseWeightsSnapshot.subarray(offset, offset + l.ln1_gamma.length)); offset += l.ln1_gamma.length;
      l.ln1_beta.set(this.baseWeightsSnapshot.subarray(offset, offset + l.ln1_beta.length)); offset += l.ln1_beta.length;
      l.fc1.set(this.baseWeightsSnapshot.subarray(offset, offset + l.fc1.length)); offset += l.fc1.length;
      l.fc1_b.set(this.baseWeightsSnapshot.subarray(offset, offset + l.fc1_b.length)); offset += l.fc1_b.length;
      l.fc2.set(this.baseWeightsSnapshot.subarray(offset, offset + l.fc2.length)); offset += l.fc2.length;
      l.fc2_b.set(this.baseWeightsSnapshot.subarray(offset, offset + l.fc2_b.length)); offset += l.fc2_b.length;
      l.ln2_gamma.set(this.baseWeightsSnapshot.subarray(offset, offset + l.ln2_gamma.length)); offset += l.ln2_gamma.length;
      l.ln2_beta.set(this.baseWeightsSnapshot.subarray(offset, offset + l.ln2_beta.length)); offset += l.ln2_beta.length;
    }

    this.weights.ln_f_gamma.set(this.baseWeightsSnapshot.subarray(offset, offset + this.weights.ln_f_gamma.length)); offset += this.weights.ln_f_gamma.length;
    this.weights.ln_f_beta.set(this.baseWeightsSnapshot.subarray(offset, offset + this.weights.ln_f_beta.length)); offset += this.weights.ln_f_beta.length;
    this.weights.lm_head.set(this.baseWeightsSnapshot.subarray(offset, offset + this.weights.lm_head.length)); offset += this.weights.lm_head.length;

    // Restore the statistical memory layer to its base state as well
    if (this.baseMemorySnapshot) {
      this.memory.restore(this.baseMemorySnapshot);
    }

    // The weights are back at the base checkpoint, so no adaptation is left.
    this.fullFineTuneApplied = false;
  }

  /**
   * Returns true if the model's weights have been adapted at all.
   *
   * LoRA fine-tuning is visible in the adapter matrices, while a full retrain
   * updates `lm_head` and leaves the adapters at zero — checking the adapters
   * alone made a fully retrained model report itself as base pretrained.
   */
  public isFineTuned(): boolean {
    if (this.fullFineTuneApplied) return true;
    return this.weights.layers.some(
      (l) =>
        l.lora_q_B.some((w) => Math.abs(w) > 1e-6) ||
        l.lora_v_B.some((w) => Math.abs(w) > 1e-6)
    );
  }

  /**
   * Returns parameter count and exact memory statistics in bytes
   */
  public getMemoryStats(): {
    totalParams: number;
    lmHeadParams: number;
    loraParams: number;
    weightsMemoryBytes: number;
    scratchMemoryBytes: number;
    snapshotMemoryBytes: number;
    totalMemoryFormatted: string;
    weightsMemoryFormatted: string;
    scratchBuffersFormatted: string;
  } {
    const weightsMemoryBytes = this.totalWeightFloats * 4;
    const scratchMemoryBytes = (
      this.scratchX.length +
      this.scratchXNorm1.length +
      this.scratchQ.length +
      this.scratchK.length +
      this.scratchV.length +
      this.scratchAttnOut.length +
      this.scratchProjOut.length +
      this.scratchXNorm2.length +
      this.scratchMid.length +
      this.scratchFinalNorm.length +
      this.scratchLogits.length +
      this.scratchLoraQ.length +
      this.scratchLoraV.length
    ) * 4 + this.trainingScratchFloats() * 4;
    const snapshotMemoryBytes = (this.baseWeightsSnapshot?.length ?? 0) * 4;
    const totalBytes = weightsMemoryBytes + scratchMemoryBytes + snapshotMemoryBytes;

    let loraParams = 0;
    for (const l of this.weights.layers) {
      loraParams += l.lora_q_A.length + l.lora_q_B.length + l.lora_v_A.length + l.lora_v_B.length;
    }

    return {
      totalParams: this.totalWeightFloats,
      lmHeadParams: this.config.vocabSize * this.config.dModel,
      loraParams,
      weightsMemoryBytes,
      scratchMemoryBytes,
      snapshotMemoryBytes,
      totalMemoryFormatted: `${(totalBytes / 1024 / 1024).toFixed(2)} MB`,
      weightsMemoryFormatted: `${(weightsMemoryBytes / 1024).toFixed(1)} KB`,
      scratchBuffersFormatted: `${(scratchMemoryBytes / 1024).toFixed(1)} KB`
    };
  }

  /**
   * Parameter counts for the current adaptation mode.
   *
   * LoRA updates the adapters and a full retrain updates the LM head; neither
   * touches every weight, so reporting `trainable: total` (as this used to)
   * overstated what training can actually change.
   */
  public countParameters(loraMode = true): { total: number; trainable: number; loraOnly: number } {
    const stats = this.getMemoryStats();
    return {
      total: stats.totalParams,
      trainable: loraMode ? stats.loraParams : stats.lmHeadParams,
      loraOnly: stats.loraParams
    };
  }

  /**
   * Allocates the per-layer activation cache on first use. Sized from the
   * configuration, so it is allocated once and never grows during training.
   */
  private ensureActivationCache(): ActivationCache {
    if (this.activationCache) return this.activationCache;
    const { nLayers, dModel, maxSeqLen } = this.config;
    const tokenArea = maxSeqLen * dModel;
    const cache: ActivationCache = {
      blockInput: [],
      norm1: [],
      q: [],
      k: [],
      v: [],
      x1: [],
      norm2: [],
      finalResidual: new Float32Array(tokenArea)
    };
    for (let l = 0; l < nLayers; l++) {
      cache.blockInput.push(new Float32Array(tokenArea));
      cache.norm1.push(new Float32Array(tokenArea));
      cache.q.push(new Float32Array(tokenArea));
      cache.k.push(new Float32Array(tokenArea));
      cache.v.push(new Float32Array(tokenArea));
      cache.x1.push(new Float32Array(tokenArea));
      cache.norm2.push(new Float32Array(tokenArea));
    }
    this.activationCache = cache;
    return cache;
  }

  /** Allocates the LoRA gradient scratch buffers on first use. */
  private ensureGradScratch(): void {
    if (this.gradScratchReady) return;
    const { dModel, dFfn, vocabSize, maxSeqLen, loraRank } = this.config;
    const tokenArea = maxSeqLen * dModel;
    const rank = Math.max(1, loraRank);
    this.gNorm = new Float32Array(tokenArea);
    this.gResidual = new Float32Array(tokenArea);
    this.gX1 = new Float32Array(tokenArea);
    this.gIn = new Float32Array(tokenArea);
    this.gProjOut = new Float32Array(tokenArea);
    this.gAttnOut = new Float32Array(tokenArea);
    this.gQ = new Float32Array(tokenArea);
    this.gK = new Float32Array(tokenArea);
    this.gV = new Float32Array(tokenArea);
    this.gXNorm1 = new Float32Array(tokenArea);
    this.gXNorm2Row = new Float32Array(dModel);
    this.gFfnPre = new Float32Array(dFfn);
    this.gProbs = new Float32Array(vocabSize);
    this.gAttnRow = new Float32Array(maxSeqLen);
    this.dLora_qA = new Float32Array(rank * dModel);
    this.dLora_qB = new Float32Array(dModel * rank);
    this.dLora_vA = new Float32Array(rank * dModel);
    this.dLora_vB = new Float32Array(dModel * rank);
    this.adapterRow = new Float32Array(rank);
    this.gradScratchReady = true;
  }

  /**
   * Bytes held by the LoRA activation cache and gradient scratch, which only
   * exist once training with LoRA has run. Reported so the memory stats stay
   * honest about what the model actually holds.
   */
  private trainingScratchFloats(): number {
    let floats = 0;
    const cache = this.activationCache;
    if (cache) {
      floats += cache.finalResidual.length;
      for (let l = 0; l < cache.blockInput.length; l++) {
        floats +=
          cache.blockInput[l].length +
          cache.norm1[l].length +
          cache.q[l].length +
          cache.k[l].length +
          cache.v[l].length +
          cache.x1[l].length +
          cache.norm2[l].length;
      }
    }
    if (this.gradScratchReady) {
      floats +=
        this.gNorm.length +
        this.gResidual.length +
        this.gX1.length +
        this.gIn.length +
        this.gProjOut.length +
        this.gAttnOut.length +
        this.gQ.length +
        this.gK.length +
        this.gV.length +
        this.gXNorm1.length +
        this.gXNorm2Row.length +
        this.gFfnPre.length +
        this.gProbs.length +
        this.gAttnRow.length +
        this.dLora_qA.length +
        this.dLora_qB.length +
        this.dLora_vA.length +
        this.dLora_vB.length +
        this.adapterRow.length;
    }
    return floats;
  }

  /**
   * Applies one accumulated adapter gradient: w -= lr * (mean(dW) + wd * w).
   * The gradient is a sum over the sequence's positions, so it is averaged here
   * to keep the learning rate independent of sequence length.
   */
  private applyAdapterUpdate(
    param: Float32Array,
    grad: Float32Array,
    targetCount: number,
    learningRate: number,
    weightDecay: number
  ): void {
    const invCount = targetCount > 0 ? 1 / targetCount : 0;
    for (let i = 0; i < param.length; i++) {
      param[i] -= learningRate * (grad[i] * invCount + weightDecay * param[i]);
    }
  }

  /**
   * Accumulates one LoRA adapter's gradient at a single position, and adds the
   * adapter's contribution to the gradient w.r.t. the LN1 output.
   *
   * With A as [rank, dModel], B as [dModel, rank] and u = loraScale * A x, the
   * adapter contributes loraScale * B u to the projection output, so both
   * gradients carry that factor.
   */
  private accumulateAdapterGradient(
    x: Float32Array,
    xOffset: number,
    dOut: Float32Array,
    gXNorm: Float32Array,
    gXNormOffset: number,
    A: Float32Array,
    B: Float32Array,
    dA: Float32Array,
    dB: Float32Array,
    loraScale: number,
    dModel: number,
    loraRank: number
  ): void {
    // u = loraScale * A x — the vector B maps back up to model width.
    for (let r = 0; r < loraRank; r++) {
      let val = 0;
      const aOffset = r * dModel;
      for (let c = 0; c < dModel; c++) val += A[aOffset + c] * x[xOffset + c];
      this.adapterRow[r] = val * loraScale;
    }
    // dL/dB[row][r] += dL/dout[row] * u[r]
    for (let row = 0; row < dModel; row++) {
      const g = dOut[xOffset + row];
      if (g === 0) continue;
      for (let r = 0; r < loraRank; r++) {
        dB[row * loraRank + r] += g * this.adapterRow[r];
      }
    }
    // dL/du[r] = sum_row dL/dout[row] * B[row][r]
    for (let r = 0; r < loraRank; r++) {
      let sum = 0;
      for (let row = 0; row < dModel; row++) sum += dOut[xOffset + row] * B[row * loraRank + r];
      this.adapterRow[r] = sum * loraScale;
    }
    // dL/dA[r][c] += dL/du[r] * x[c], and the adapter's path into x.
    for (let r = 0; r < loraRank; r++) {
      const gU = this.adapterRow[r];
      if (gU === 0) continue;
      const aOffset = r * dModel;
      for (let c = 0; c < dModel; c++) {
        gXNorm[gXNormOffset + c] += gU * A[aOffset + c];
        dA[aOffset + c] += gU * x[xOffset + c];
      }
    }
  }

  /**
   * Backward pass through the frozen network for the LoRA adapters.
   *
   * Fine-tuning used to nudge a single `lora_v_B` row chosen by
   * `targetToken % dModel`, scaled by the loss gradient. That row has no
   * relationship to the parameter's true gradient, so the update moved the loss
   * in an arbitrary direction, and `lora_q_A`/`lora_q_B` were never updated at
   * all. This differentiates the token cross-entropy w.r.t. every adapter weight
   * and applies dL/dW — the real LoRA update.
   *
   * Every base weight (projections, feed-forward, embeddings, LM head) stays
   * frozen, so the adapters remain the only thing training can change and a
   * reset still restores the pretrained model exactly.
   */
  private backwardLora(
    tokens: number[],
    seqLen: number,
    learningRate: number,
    weightDecay: number,
    targetCount: number
  ): void {
    const cache = this.activationCache;
    if (!cache) return;
    this.ensureGradScratch();

    const { dModel, nHeads, nLayers, dFfn, vocabSize, loraRank, loraAlpha } = this.config;
    const headDim = Math.floor(dModel / nHeads);
    const loraScale = loraRank > 0 ? loraAlpha / loraRank : 1.0;
    const attnScale = 1.0 / Math.sqrt(headDim);
    const tokenArea = seqLen * dModel;

    // 1. Cross-entropy gradient: dL/dlogits = softmax - onehot, so
    //    dL/dz = lm_head^T dL/dlogits at every position.
    this.gNorm.fill(0);
    for (let i = 0; i < seqLen - 1; i++) {
      let targetToken = tokens[i + 1];
      if (targetToken < 0 || targetToken >= vocabSize || !Number.isFinite(targetToken)) {
        targetToken = UNK_ID;
      }
      if (targetToken === PAD_ID) continue;
      const logitRow = this.scratchLogits.subarray(i * vocabSize, (i + 1) * vocabSize);
      softmax(logitRow, this.gProbs, 1.0);
      const rowOffset = i * dModel;
      for (let v = 0; v < vocabSize; v++) {
        const grad = this.gProbs[v] - (v === targetToken ? 1.0 : 0.0);
        if (grad === 0) continue;
        const vOffset = v * dModel;
        for (let d = 0; d < dModel; d++) {
          this.gNorm[rowOffset + d] += grad * this.weights.lm_head[vOffset + d];
        }
      }
    }

    // 2. Final LayerNorm -> gradient w.r.t. the residual stream after the blocks.
    for (let i = 0; i < seqLen; i++) {
      layerNormBackward(
        cache.finalResidual,
        i * dModel,
        this.weights.ln_f_gamma,
        this.gNorm,
        i * dModel,
        this.gResidual,
        i * dModel,
        dModel
      );
    }

    // 3. One block at a time, top down. The cached activations come from the
    //    forward pass this gradient belongs to, so applying a block's adapter
    //    update early cannot disturb the gradients of the blocks below it.
    for (let l = nLayers - 1; l >= 0; l--) {
      const layer = this.weights.layers[l];
      const blockInput = cache.blockInput[l];
      const norm1 = cache.norm1[l];
      const cachedQ = cache.q[l];
      const cachedK = cache.k[l];
      const cachedV = cache.v[l];
      const x1 = cache.x1[l];
      const norm2 = cache.norm2[l];

      // --- Feed-forward: dL/dpreAct, then dL/dxNorm2 ---
      for (let i = 0; i < seqLen; i++) {
        const rowOffset = i * dModel;
        for (let r = 0; r < dFfn; r++) {
          let dMid = 0;
          const rOffset = r * dModel;
          for (let row = 0; row < dModel; row++) {
            dMid += this.gResidual[rowOffset + row] * layer.fc2[row * dFfn + r];
          }
          let preAct = layer.fc1_b[r];
          for (let c = 0; c < dModel; c++) preAct += layer.fc1[rOffset + c] * norm2[rowOffset + c];
          this.gFfnPre[r] = dMid * geluDerivative(preAct);
        }
        for (let c = 0; c < dModel; c++) {
          let sum = 0;
          for (let r = 0; r < dFfn; r++) sum += this.gFfnPre[r] * layer.fc1[r * dModel + c];
          this.gXNorm2Row[c] = sum;
        }
        layerNormBackward(x1, rowOffset, layer.ln2_gamma, this.gXNorm2Row, 0, this.gX1, rowOffset, dModel);
      }
      // x_{l+1} = x1 + ffnOut, so the residual path reuses the incoming gradient.
      for (let idx = 0; idx < tokenArea; idx++) this.gX1[idx] += this.gResidual[idx];

      // --- Attention: dL/dattnOut, then dL/dQ, dL/dK, dL/dV ---
      copyInto(this.gProjOut, this.gX1, tokenArea);
      for (let i = 0; i < seqLen; i++) {
        const rowOffset = i * dModel;
        for (let c = 0; c < dModel; c++) {
          let sum = 0;
          for (let row = 0; row < dModel; row++) {
            sum += this.gProjOut[rowOffset + row] * layer.out_proj[row * dModel + c];
          }
          this.gAttnOut[rowOffset + c] = sum;
        }
      }

      this.gQ.fill(0, 0, tokenArea);
      this.gK.fill(0, 0, tokenArea);
      this.gV.fill(0, 0, tokenArea);
      const attentionMaps = this.lastAttentionMaps[l];
      for (let h = 0; h < nHeads; h++) {
        const headOffset = h * headDim;
        for (let i = 0; i < seqLen; i++) {
          const outOffset = i * dModel + headOffset;
          const weights = attentionMaps?.[h]?.[i];
          if (!weights) continue;

          let weightGradSum = 0;
          for (let j = 0; j <= i; j++) {
            const vjOffset = j * dModel + headOffset;
            let dot = 0;
            for (let d = 0; d < headDim; d++) {
              dot += this.gAttnOut[outOffset + d] * cachedV[vjOffset + d];
            }
            this.gAttnRow[j] = dot;
            weightGradSum += weights[j] * dot;
          }
          // dL/dV_j += w_ij * dL/dattnOut_i
          for (let j = 0; j <= i; j++) {
            const w = weights[j];
            if (w === 0) continue;
            const vjOffset = j * dModel + headOffset;
            for (let d = 0; d < headDim; d++) {
              this.gV[vjOffset + d] += w * this.gAttnOut[outOffset + d];
            }
          }
          // Softmax backward, folded with the 1/sqrt(headDim) attention scale.
          for (let j = 0; j <= i; j++) {
            const gScores = weights[j] * (this.gAttnRow[j] - weightGradSum) * attnScale;
            if (gScores === 0) continue;
            const kjOffset = j * dModel + headOffset;
            for (let d = 0; d < headDim; d++) {
              this.gQ[outOffset + d] += gScores * cachedK[kjOffset + d];
              this.gK[kjOffset + d] += gScores * cachedQ[outOffset + d];
            }
          }
        }
      }

      // --- Adapter gradients, dL/dxNorm1, and the LN1 backward ---
      this.dLora_qA.fill(0);
      this.dLora_qB.fill(0);
      this.dLora_vA.fill(0);
      this.dLora_vB.fill(0);
      for (let i = 0; i < seqLen; i++) {
        const rowOffset = i * dModel;

        // Frozen projections back to the LN1 output.
        for (let c = 0; c < dModel; c++) {
          let sum = 0;
          for (let row = 0; row < dModel; row++) {
            sum +=
              this.gQ[rowOffset + row] * layer.q_proj[row * dModel + c] +
              this.gK[rowOffset + row] * layer.k_proj[row * dModel + c] +
              this.gV[rowOffset + row] * layer.v_proj[row * dModel + c];
          }
          this.gXNorm1[rowOffset + c] = sum;
        }

        if (loraRank > 0) {
          this.accumulateAdapterGradient(
            norm1, rowOffset, this.gV, this.gXNorm1, rowOffset,
            layer.lora_v_A, layer.lora_v_B, this.dLora_vA, this.dLora_vB,
            loraScale, dModel, loraRank
          );
          this.accumulateAdapterGradient(
            norm1, rowOffset, this.gQ, this.gXNorm1, rowOffset,
            layer.lora_q_A, layer.lora_q_B, this.dLora_qA, this.dLora_qB,
            loraScale, dModel, loraRank
          );
        }

        // dL/dx_l = dL/dx1 + LN1 backward
        layerNormBackward(blockInput, rowOffset, layer.ln1_gamma, this.gXNorm1, rowOffset, this.gIn, rowOffset, dModel);
      }
      for (let idx = 0; idx < tokenArea; idx++) this.gIn[idx] += this.gX1[idx];

      // One update per block, averaged over the sequence's positions.
      if (loraRank > 0) {
        this.applyAdapterUpdate(layer.lora_q_A, this.dLora_qA, targetCount, learningRate, weightDecay);
        this.applyAdapterUpdate(layer.lora_q_B, this.dLora_qB, targetCount, learningRate, weightDecay);
        this.applyAdapterUpdate(layer.lora_v_A, this.dLora_vA, targetCount, learningRate, weightDecay);
        this.applyAdapterUpdate(layer.lora_v_B, this.dLora_vB, targetCount, learningRate, weightDecay);
      }

      copyInto(this.gResidual, this.gIn, tokenArea);
    }
  }

  /**
   * Forward pass: computes logits using pre-allocated scratch memory to prevent GC lag.
   * Returns view of computed logits for sequence.
   *
   * Passing `cacheActivations` additionally retains the per-layer activations the
   * LoRA backward pass needs; inference never pays for that buffer.
   */
  public forward(
    tokens: number[],
    useLora = true,
    cacheActivations = false
  ): {
    logits: Float32Array; // [seqLen, vocabSize]
    seqLen: number;
  } {
    const { dModel, nHeads, nLayers, dFfn, vocabSize, loraRank, loraAlpha, maxSeqLen } = this.config;
    // Sliding context window: when sequence exceeds maxSeqLen, condition on the most recent tokens
    const activeTokens = tokens.length > maxSeqLen ? tokens.slice(tokens.length - maxSeqLen) : tokens;
    const seqLen = activeTokens.length;
    const headDim = Math.floor(dModel / nHeads);
    const loraScale = loraRank > 0 ? loraAlpha / loraRank : 1.0;
    const cache = cacheActivations ? this.ensureActivationCache() : null;
    const tokenArea = seqLen * dModel;

    // Reset attention visualization structures
    this.lastAttentionMaps = [];
    for (let l = 0; l < nLayers; l++) {
      this.lastAttentionMaps.push([]);
      for (let h = 0; h < nHeads; h++) {
        const mat: number[][] = [];
        for (let i = 0; i < seqLen; i++) mat.push(new Array(seqLen).fill(0));
        this.lastAttentionMaps[l].push(mat);
      }
    }

    // 1. Embedding lookup: scratchX[i] = wte[tokens[i]] + wpe[i]
    //
    // An id outside the vocabulary means the tokenizer and the model were built
    // from different vocabularies. Clamping keeps generation alive, but safely
    // mapping out-of-bounds ids to UNK_ID preserves valid vocabulary embeddings.
    if (!this.vocabMismatchWarned) {
      for (let i = 0; i < seqLen; i++) {
        if (activeTokens[i] < 0 || activeTokens[i] >= vocabSize) {
          this.vocabMismatchWarned = true;
          console.warn(
            `[slm] token id ${activeTokens[i]} at position ${i} is outside the ${vocabSize}-token ` +
              `vocabulary; the tokenizer grew after this model was built. Rebuild the model ` +
              `so the new tokens get embeddings.`
          );
          break;
        }
      }
    }
    for (let i = 0; i < seqLen; i++) {
      let tokenId = activeTokens[i];
      if (tokenId < 0 || tokenId >= vocabSize || !Number.isFinite(tokenId)) {
        tokenId = UNK_ID;
      }
      const wteOffset = tokenId * dModel;
      const wpeOffset = i * dModel;
      const xOffset = i * dModel;

      for (let d = 0; d < dModel; d++) {
        this.scratchX[xOffset + d] = this.weights.wte[wteOffset + d] + this.weights.wpe[wpeOffset + d];
      }
    }

    // 2. Transformer Blocks
    for (let l = 0; l < nLayers; l++) {
      const layer = this.weights.layers[l];

      if (cache) copyInto(cache.blockInput[l], this.scratchX, tokenArea);

      // LayerNorm 1
      for (let i = 0; i < seqLen; i++) {
        const rowIn = this.scratchX.subarray(i * dModel, (i + 1) * dModel);
        const rowOut = this.scratchXNorm1.subarray(i * dModel, (i + 1) * dModel);
        layerNorm(rowIn, layer.ln1_gamma, layer.ln1_beta, rowOut, dModel);
      }
      if (cache) copyInto(cache.norm1[l], this.scratchXNorm1, tokenArea);

      // Linear projections: Q, K, V
      const useAdapters = useLora && loraRank > 0;
      for (let i = 0; i < seqLen; i++) {
        const xi = this.scratchXNorm1.subarray(i * dModel, (i + 1) * dModel);
        const qi = this.scratchQ.subarray(i * dModel, (i + 1) * dModel);
        const ki = this.scratchK.subarray(i * dModel, (i + 1) * dModel);
        const vi = this.scratchV.subarray(i * dModel, (i + 1) * dModel);

        // LoRA down-projection: u = A x is rank-sized and independent of the
        // output row, so it is computed once per token position here. Deriving
        // it inside the row loop re-ran the full dModel-wide dot product dModel
        // times per token — the quadratic overhead this avoids.
        if (useAdapters) {
          for (let r = 0; r < loraRank; r++) {
            let aQ = 0, aV = 0;
            const aOffset = r * dModel;
            for (let c = 0; c < dModel; c++) {
              const val = xi[c];
              aQ += layer.lora_q_A[aOffset + c] * val;
              aV += layer.lora_v_A[aOffset + c] * val;
            }
            this.scratchLoraQ[r] = aQ;
            this.scratchLoraV[r] = aV;
          }
        }

        for (let row = 0; row < dModel; row++) {
          let sumQ = 0, sumK = 0, sumV = 0;
          const rOffset = row * dModel;
          for (let col = 0; col < dModel; col++) {
            const val = xi[col];
            sumQ += layer.q_proj[rOffset + col] * val;
            sumK += layer.k_proj[rOffset + col] * val;
            sumV += layer.v_proj[rOffset + col] * val;
          }

          // LoRA modification: W + (B @ A) * alpha / rank, with the cached
          // down-projection u = A x feeding the rank-sized up-projection.
          if (useAdapters) {
            let loraQ = 0, loraV = 0;
            const bOffset = row * loraRank;
            for (let r = 0; r < loraRank; r++) {
              loraQ += layer.lora_q_B[bOffset + r] * this.scratchLoraQ[r];
              loraV += layer.lora_v_B[bOffset + r] * this.scratchLoraV[r];
            }
            sumQ += loraQ * loraScale;
            sumV += loraV * loraScale;
          }

          qi[row] = sumQ;
          ki[row] = sumK;
          vi[row] = sumV;
        }
      }
      if (cache) {
        copyInto(cache.q[l], this.scratchQ, tokenArea);
        copyInto(cache.k[l], this.scratchK, tokenArea);
        copyInto(cache.v[l], this.scratchV, tokenArea);
      }

      // Multi-Head Attention with causal mask
      this.scratchAttnOut.fill(0, 0, seqLen * dModel);
      const scale = 1.0 / Math.sqrt(headDim);

      for (let h = 0; h < nHeads; h++) {
        const headOffset = h * headDim;

        for (let i = 0; i < seqLen; i++) {
          const qi = this.scratchQ.subarray(i * dModel + headOffset, i * dModel + headOffset + headDim);
          const scores = new Float32Array(i + 1);

          for (let j = 0; j <= i; j++) {
            const kj = this.scratchK.subarray(j * dModel + headOffset, j * dModel + headOffset + headDim);
            let dot = 0;
            for (let d = 0; d < headDim; d++) {
              dot += qi[d] * kj[d];
            }
            scores[j] = dot * scale;
          }

          const weights = new Float32Array(i + 1);
          softmax(scores, weights);

          for (let j = 0; j <= i; j++) {
            this.lastAttentionMaps[l][h][i][j] = weights[j];
          }

          const outH = this.scratchAttnOut.subarray(i * dModel + headOffset, i * dModel + headOffset + headDim);
          for (let j = 0; j <= i; j++) {
            const w = weights[j];
            const vj = this.scratchV.subarray(j * dModel + headOffset, j * dModel + headOffset + headDim);
            for (let d = 0; d < headDim; d++) {
              outH[d] += w * vj[d];
            }
          }
        }
      }

      // Output projection: projOut = attnOut @ out_proj
      for (let i = 0; i < seqLen; i++) {
        const attni = this.scratchAttnOut.subarray(i * dModel, (i + 1) * dModel);
        const resi = this.scratchProjOut.subarray(i * dModel, (i + 1) * dModel);
        for (let row = 0; row < dModel; row++) {
          let sum = 0;
          const rOffset = row * dModel;
          for (let col = 0; col < dModel; col++) {
            sum += layer.out_proj[rOffset + col] * attni[col];
          }
          resi[row] = sum;
        }
      }

      // Residual 1: x = x + projOut
      for (let idx = 0; idx < seqLen * dModel; idx++) {
        this.scratchX[idx] += this.scratchProjOut[idx];
      }
      if (cache) copyInto(cache.x1[l], this.scratchX, tokenArea);

      // LayerNorm 2
      for (let i = 0; i < seqLen; i++) {
        const rowIn = this.scratchX.subarray(i * dModel, (i + 1) * dModel);
        const rowOut = this.scratchXNorm2.subarray(i * dModel, (i + 1) * dModel);
        layerNorm(rowIn, layer.ln2_gamma, layer.ln2_beta, rowOut, dModel);
      }
      if (cache) copyInto(cache.norm2[l], this.scratchXNorm2, tokenArea);

      // FFN: x = x + fc2(gelu(fc1(xNorm2)))
      for (let i = 0; i < seqLen; i++) {
        const inRow = this.scratchXNorm2.subarray(i * dModel, (i + 1) * dModel);

        // fc1 + gelu
        for (let r = 0; r < dFfn; r++) {
          let sum = layer.fc1_b[r];
          const rOffset = r * dModel;
          for (let c = 0; c < dModel; c++) {
            sum += layer.fc1[rOffset + c] * inRow[c];
          }
          this.scratchMid[r] = gelu(sum);
        }

        // fc2
        const xOffset = i * dModel;
        for (let r = 0; r < dModel; r++) {
          let sum = layer.fc2_b[r];
          const rOffset = r * dFfn;
          for (let c = 0; c < dFfn; c++) {
            sum += layer.fc2[rOffset + c] * this.scratchMid[c];
          }
          this.scratchX[xOffset + r] += sum;
        }
      }
    }

    if (cache) copyInto(cache.finalResidual, this.scratchX, tokenArea);

    // 3. Final LayerNorm
    for (let i = 0; i < seqLen; i++) {
      const rowIn = this.scratchX.subarray(i * dModel, (i + 1) * dModel);
      const rowOut = this.scratchFinalNorm.subarray(i * dModel, (i + 1) * dModel);
      layerNorm(rowIn, this.weights.ln_f_gamma, this.weights.ln_f_beta, rowOut, dModel);
    }

    // 4. LM Head projection: logits = xFinal @ lm_head
    for (let i = 0; i < seqLen; i++) {
      const hRow = this.scratchFinalNorm.subarray(i * dModel, (i + 1) * dModel);
      const logitRow = this.scratchLogits.subarray(i * vocabSize, (i + 1) * vocabSize);

      for (let v = 0; v < vocabSize; v++) {
        let sum = 0;
        const vOffset = v * dModel;
        for (let d = 0; d < dModel; d++) {
          sum += this.weights.lm_head[vOffset + d] * hRow[d];
        }
        logitRow[v] = sum;
      }
    }

    return {
      logits: this.scratchLogits.subarray(0, seqLen * vocabSize),
      seqLen
    };
  }

  /**
   * Train step on a conversational token sequence.
   * Updates LoRA adapters or full weights using Cross Entropy loss.
   */
  public trainStep(
    tokens: number[],
    learningRate = 0.01,
    loraMode = true,
    weightDecay = 0.005
  ): { loss: number; perplexity: number } {
    const { vocabSize, dModel, loraRank } = this.config;
    const seqLen = Math.min(tokens.length, this.config.maxSeqLen);
    if (seqLen <= 1) return { loss: 0, perplexity: 1.0 };

    // LoRA training differentiates the layers, so it needs this exact forward
    // pass to retain its activations; a full retrain only needs the final
    // hidden states.
    const loraTraining = loraMode && loraRank > 0;

    // Forward pass
    const { logits } = this.forward(tokens, loraMode, loraTraining);

    // A full retrain adapts lm_head while the LoRA adapters stay at zero, so
    // record the adaptation for isFineTuned() (resetToBase() clears it).
    if (!loraMode) {
      this.fullFineTuneApplied = true;
    }

    let totalLoss = 0;
    let targetCount = 0;

    // Train the statistical memory layer on this sequence so fine-tuning
    // visibly teaches the model the new persona / dataset phrases.
    this.memory.observe(tokens.slice(0, seqLen), 2.0);

    // Response-link observation: also teach the memory how this user message
    // was answered, so fine-tuning links the question to the dataset answer
    // instead of only learning the reply in isolation.
    const asstIdx = tokens.indexOf(ASSISTANT_ID);
    if (asstIdx > 0 && asstIdx + 2 < seqLen) {
      const opening = this.replyOpeningContext(tokens, asstIdx);
      if (opening.length === 2) {
        // (last user word, space) -> first reply word, as a fluent trigram...
        this.memory.observe([...opening, tokens[asstIdx + 2]], 2.0);
        // ...and as a dialogue-pair link used to open the generated reply.
        this.memory.observeReplyLink(opening[0], tokens[asstIdx + 2], 2.0);
      }
      // Remember this user message together with its answer, so asking the
      // trained question again reproduces the trained answer.
      this.memory.rememberCase(this.caseWords(this.userContentTokens(tokens, asstIdx)), tokens.slice(asstIdx + 2, seqLen));
    }

    // Hidden states from the forward pass (needed for gradient updates).
    const hidden = this.scratchFinalNorm; // [seqLen, dModel]

    const neuralProbs = new Float32Array(vocabSize);
    const mixed = new Float32Array(vocabSize);
    for (let i = 0; i < seqLen - 1; i++) {
      let targetToken = tokens[i + 1];
      if (targetToken < 0 || targetToken >= vocabSize || !Number.isFinite(targetToken)) {
        targetToken = UNK_ID;
      }
      if (targetToken === PAD_ID) continue;

      // Neural softmax at position i
      const logitRow = logits.subarray(i * vocabSize, (i + 1) * vocabSize);
      softmax(logitRow, neuralProbs, 1.0);

      // Memory distribution for this context, then mix exactly like generation
      const prev1 = tokens[i];
      const prev2 = i >= 1 ? tokens[i - 1] : BOS_ID;
      this.memory.distribution(prev2, prev1, mixed);

      const mix = this.neuralMix;
      for (let v = 0; v < vocabSize; v++) {
        mixed[v] = mix * neuralProbs[v] + (1 - mix) * mixed[v];
      }

      // Loss measured on the same blended distribution the model generates with
      const targetProb = Math.max(1e-8, mixed[targetToken]);
      totalLoss += -Math.log(targetProb);
      targetCount++;

      const hOffset = i * dModel;

      if (!loraMode) {
        // Full fine-tuning: real cross-entropy gradient descent on lm_head rows.
        // dL/dlogit_v = mixed[v] - 1[v==target]; applied via the hidden state.
        const lr = learningRate * 0.35;
        for (let v = 0; v < vocabSize; v++) {
          const grad = mixed[v] - (v === targetToken ? 1.0 : 0.0);
          if (Math.abs(grad) < 0.004) continue; // skip negligible gradients
          const vOffset = v * dModel;
          for (let d = 0; d < dModel; d++) {
            const w = this.weights.lm_head[vOffset + d];
            // Cross-entropy gradient plus L2 weight decay.
            this.weights.lm_head[vOffset + d] -= lr * (grad * hidden[hOffset + d] + weightDecay * w);
          }
        }
      }
    }

    // LoRA adapters learn from a real backward pass over this sequence, run
    // after the loss loop so the same forward activations yield both the
    // reported loss and the gradient.
    if (loraTraining) {
      this.backwardLora(tokens, seqLen, learningRate, weightDecay, targetCount);
    }

    const avgLoss = targetCount > 0 ? totalLoss / targetCount : 0;
    const perplexity = Math.min(9999, Math.exp(Math.min(10, avgLoss)));
    return { loss: avgLoss, perplexity };
  }

  /**
   * Generates the next token with conversational coherence and temperature/sampling.
   */
  public generateNextToken(
    tokens: number[],
    options: GenerationOptions,
    useLora = true
  ): GeneratedTokenInfo {
    const { vocabSize } = this.config;
    const seqLen = Math.min(tokens.length, this.config.maxSeqLen);

    // Forward pass
    const { logits } = this.forward(tokens, useLora);

    // Get logits for the last token position
    const lastOffset = (seqLen - 1) * vocabSize;
    const rawLogits = logits.subarray(lastOffset, lastOffset + vocabSize);

    // --- Neural distribution (raw softmax) ---
    // Temperature is applied exactly once, to the blended distribution below.
    // Scaling the logits here *and* re-sharpening the mix would square the
    // effect, so a slider value of 0.7 would sample like ~0.49.
    const neuralProbs = new Float32Array(vocabSize);
    softmax(rawLogits, neuralProbs, 1.0);

    // --- Statistical memory distribution (trigram -> bigram -> unigram backoff) ---
    const n = tokens.length;
    const lastToken = tokens[n - 1];
    const prev2 = n >= 2 ? tokens[n - 2] : BOS_ID;

    const memoryProbs = new Float32Array(vocabSize);
    this.memory.distribution(prev2, lastToken, memoryProbs);

    // --- Reply opener: the first generated token is the one that answers the
    // user, so prefer the opening this model actually learned for the user's
    // last word (falling back to how it usually opens replies) ---
    let openerProbs: Float32Array | null = null;
    let openerWeight = 0;
    if (this.pendingReplyWord !== null) {
      const opener = new Float32Array(vocabSize);
      if (this.memory.linkDistribution(this.pendingReplyWord, opener)) {
        openerProbs = opener;
        openerWeight = 0.9;
      } else if (this.memory.openerDistribution(opener)) {
        openerProbs = opener;
        openerWeight = 0.7;
      }
      this.pendingReplyWord = null;
    }

    // --- Dialogue-case replay: while generation stays on the answer learned
    // for this user message, keep it there so trained prompts reproduce their
    // dataset answers instead of drifting through the pooled n-gram average ---
    const caseToken =
      this.caseReply !== null && this.casePos < this.caseReply.length
        ? this.caseReply[this.casePos]
        : null;

    // --- Mix neural + memory, suppress control tokens, break repetition loops ---
    const probs = new Float32Array(vocabSize);
    const repeatedTwice = n >= 2 && tokens[n - 2] === lastToken;
    const mix = this.neuralMix;

    // Single-character fallback tokens only exist for OOV words; a word-level
    // dialogue model should never spell characters out loud. Real single-letter
    // vocabulary words ("a", "i") stay available.
    const isLetterChar = (v: number) => this.tokenizer.isFallbackCharToken(v);
    const structuralSet = new Set([PAD_ID, UNK_ID, BOS_ID, USER_ID, ASSISTANT_ID, NEWLINE_ID]);

    for (let v = 0; v < vocabSize; v++) {
      let p = mix * neuralProbs[v] + (1 - mix) * memoryProbs[v];

      if (openerProbs) {
        p = openerWeight * openerProbs[v] + (1 - openerWeight) * p;
      }

      if (caseToken !== null) {
        p = v === caseToken ? 0.88 + 0.12 * p : 0.12 * p;
      }

      // Never emit raw control tokens like <pad>, <unk>, <bos>, <user>, <assistant>, \n
      if (structuralSet.has(v)) {
        p = 0;
      }

      // Keep EOS available, but not before the reply has actually started.
      if (v === EOS_ID && this.generatedCount < MIN_REPLY_TOKENS) {
        p = 0;
      }

      // Never spell out OOV words character by character
      if (isLetterChar(v)) {
        p = 0;
      }

      // If the model just repeated itself, strongly discourage a third repeat
      if (repeatedTwice && v === lastToken) {
        p *= 0.02;
      }

      probs[v] = p;
    }

    // Renormalize after suppression (with a safe uniform fallback)
    let pSum = 0;
    for (let v = 0; v < vocabSize; v++) pSum += probs[v];
    if (pSum <= 0) {
      const uniform = 1 / vocabSize;
      for (let v = 0; v < vocabSize; v++) probs[v] = uniform;
    } else if (Math.abs(pSum - 1) > 1e-6) {
      const inv = 1 / pSum;
      for (let v = 0; v < vocabSize; v++) probs[v] *= inv;
    }

    // Temperature sharpening on the blended distribution: keeps the fluent
    // memory-backed tokens dominant while neural-only noise gets squeezed.
    const temp = Math.max(0.05, options.temperature);
    let sharpSum = 0;
    for (let v = 0; v < vocabSize; v++) {
      const sharp = Math.pow(probs[v], 1 / temp);
      probs[v] = sharp;
      sharpSum += sharp;
    }
    if (sharpSum > 0) {
      const inv = 1 / sharpSum;
      for (let v = 0; v < vocabSize; v++) probs[v] *= inv;
    }

    // Sample next token
    const sample = sampleFromDistribution(
      probs,
      options.topK,
      options.topP,
      options.repetitionPenalty,
      tokens.slice(-12)
    );

    const tokenStr = this.tokenizer.getTokenString(sample.chosenId);
    const chosenProb = probs[sample.chosenId];

    // Stay on the learned answer until generation diverges from it
    if (caseToken !== null) {
      if (sample.chosenId === caseToken) this.casePos++;
      else this.caseReply = null;
    }

    this.generatedCount++;

    const topCandidates = sample.candidates.map(c => ({
      id: c.id,
      token: this.tokenizer.getTokenString(c.id),
      prob: c.prob,
    }));

    return {
      token: tokenStr,
      id: sample.chosenId,
      prob: chosenProb,
      topCandidates,
    };
  }

  /**
   * The two-token context that opens a reply: the last word of the user's
   * message followed by a space. This mirrors the context the memory layer uses
   * to predict the next word mid-sentence, so "<user> ... <assistant>" and the
   * dataset's own reply share one conditioned entry point.
   *
   * Returns [] when no user word precedes the assistant tag.
   */
  private isPunctuationOrStructuralToken(tokenId: number): boolean {
    if (
      tokenId === USER_ID ||
      tokenId === ASSISTANT_ID ||
      tokenId === NEWLINE_ID ||
      tokenId === BOS_ID ||
      tokenId === PAD_ID ||
      tokenId === EOS_ID ||
      tokenId === UNK_ID
    ) {
      return true;
    }
    const str = this.tokenizer.getTokenString(tokenId).trim();
    if (str.length === 0) return true;
    return !/[a-zA-Z0-9]/.test(str);
  }

  private replyOpeningContext(tokens: number[], assistantIdx: number): number[] {
    const spaceId = this.tokenizer.idOf(' ');
    if (spaceId === undefined) return [];
    for (let i = assistantIdx - 1; i >= 0; i--) {
      if (!this.isPunctuationOrStructuralToken(tokens[i])) {
        return [tokens[i], spaceId];
      }
    }
    return [];
  }

  /**
   * The content tokens of the user message that precedes the assistant tag.
   * Identical for a training sequence and for the formatted prompt of the same
   * message, which is what makes the dialogue-case lookup exact.
   */
  private userContentTokens(tokens: number[], assistantIdx: number): number[] {
    const start = tokens.lastIndexOf(USER_ID, assistantIdx);
    if (start < 0) return [];
    const out: number[] = [];
    for (let i = start + 1; i < assistantIdx; i++) {
      const t = tokens[i];
      if (t === NEWLINE_ID || t === USER_ID || t === ASSISTANT_ID || t === BOS_ID || t === PAD_ID || t === EOS_ID) break;
      out.push(t);
    }
    return out;
  }

  /**
   * The real words of a user message, used as the dialogue-case key. Spaces and
   * punctuation are dropped so that "how are you ?" and "how are you" match.
   */
  private caseWords(tokens: number[]): number[] {
    return tokens.filter((t) => /[a-z0-9]/i.test(this.tokenizer.getTokenString(t)));
  }

  private encodeForGeneration(prompt: string): number[] {
    const tokens = this.tokenizer.encode(prompt, true, false);

    // Seed the memory with the user's last word so the first generated word is
    // conditioned on what was actually asked. Without this the reply would
    // always open from the same generic "<assistant> <space>" context and
    // wander into unrelated boilerplate.
    this.pendingReplyWord = null;
    this.caseReply = null;
    this.casePos = 0;
    this.generatedCount = 0;
    const asstIdx = tokens.lastIndexOf(ASSISTANT_ID);
    if (asstIdx >= 0) {
      // Replay the answer fine-tuning learned for this user message.
      this.caseReply = this.memory.findCase(this.caseWords(this.userContentTokens(tokens, asstIdx)));

      const opening = this.replyOpeningContext(tokens, asstIdx);
      if (opening.length >= 1) {
        this.pendingReplyWord = opening[0];
      }
    }

    return tokens;
  }

  /**
   * Streaming conversational generator: streams token by token
   */
  public async *generateStream(
    prompt: string,
    options: GenerationOptions,
    useLora = true
  ): AsyncGenerator<GeneratedTokenInfo> {
    const tokens = this.encodeForGeneration(prompt);

    for (let step = 0; step < options.maxNewTokens; step++) {
      const tokenInfo = this.generateNextToken(tokens, options, useLora);
      tokens.push(tokenInfo.id);

      yield tokenInfo;

      if (tokenInfo.id === EOS_ID) break;

      // Small async tick for responsive typing animation
      await new Promise(r => setTimeout(r, 18));
    }
  }

  /**
   * Synchronous completion generator
   */
  public generate(
    prompt: string,
    options: GenerationOptions,
    useLora = true
  ): { text: string; tokens: GeneratedTokenInfo[] } {
    const tokens = this.encodeForGeneration(prompt);
    const generatedInfo: GeneratedTokenInfo[] = [];

    for (let step = 0; step < options.maxNewTokens; step++) {
      const tokenInfo = this.generateNextToken(tokens, options, useLora);
      tokens.push(tokenInfo.id);
      generatedInfo.push(tokenInfo);

      if (tokenInfo.id === EOS_ID) break;
    }

    const text = this.tokenizer.decode(generatedInfo.map(t => t.id), true);
    return { text, tokens: generatedInfo };
  }
}
