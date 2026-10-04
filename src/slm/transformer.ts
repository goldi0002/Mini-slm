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
  layerNormBackwardParam,
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

/**
 * Global-norm threshold for one training step's gradients.
 *
 * Full backpropagation updates ~100k weights from a single short sequence, and
 * a small browser model can produce a very large gradient early in training.
 * Clipping the *global* norm (rather than each element) preserves the direction
 * of the update while keeping the step bounded, which is what makes the same
 * learning rate usable from a random initialization.
 */
export const GRADIENT_CLIP_NORM = 1.0;

/**
 * How strongly the reply-opening distribution learned for the user's last word
 * overrides the blend on the first generated token. These are guardrails, not
 * the model: the network and memory blend still gets a real share of the mass.
 */
const OPENER_LINK_WEIGHT = 0.85;
const OPENER_GENERIC_WEIGHT = 0.65;

/**
 * Extra step size applied to the LoRA adapters only.
 *
 * Adapters are zero-initialized, so their first epochs must build the adapter
 * from nothing; the base network's weights are already trained. Measured on
 * held-out sentences (scripts/test_fixes.ts), a multiplier of 5 turns a barely
 * visible adapter fit into a real one without destabilising the base model.
 */
export const ADAPTER_LR_MULTIPLIER = 5;

/**
 * Share of the mass pinned to a token while replaying a dataset answer the
 * model was fine-tuned on. Kept high because this is an exact-recall path (and
 * now reported as retrieval rather than claimed as generation), but low enough
 * that a confidently wrong network can still break away from it.
 */
const CASE_REPLAY_STRENGTH = 0.82;

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

/**
 * One gradient buffer per base weight, shaped exactly like the weight it
 * belongs to. Allocated on first use, so inference-only models never pay for
 * them, and rebuilt after a vocabulary resize (embeddings and LM head change
 * shape there).
 */
interface WeightGradients {
  wte: Float32Array;
  wpe: Float32Array;
  layers: Array<{
    q_proj: Float32Array;
    k_proj: Float32Array;
    v_proj: Float32Array;
    out_proj: Float32Array;
    fc1: Float32Array;
    fc1_b: Float32Array;
    fc2: Float32Array;
    fc2_b: Float32Array;
    ln1_gamma: Float32Array;
    ln1_beta: Float32Array;
    ln2_gamma: Float32Array;
    ln2_beta: Float32Array;
  }>;
  ln_f_gamma: Float32Array;
  ln_f_beta: Float32Array;
  lm_head: Float32Array;
}

/** How the last generated reply was actually produced. */
export interface GenerationTrace {
  /** Share of the blend that came from the neural network (0..1). */
  neuralMix: number;
  /** Tokens reproduced from a dataset answer learned during fine-tuning. */
  retrievalTokens: number;
  /** Tokens drawn from the neural + memory blend. */
  generatedTokens: number;
  /** True when at least one token came from dataset retrieval. */
  usedRetrieval: boolean;
}

/** What one training step reports about itself. */
export interface TrainStepResult {
  /** Neural cross-entropy: the objective the optimizer actually minimizes. */
  loss: number;
  perplexity: number;
  /** Same as `loss`, named explicitly for charts that show both numbers. */
  neuralLoss: number;
  /** NLL of the neural + memory blend generation samples from (diagnostic). */
  blendedLoss: number;
}

/** Result of fitting the neural/memory blend weight on held-out text. */
export interface NeuralMixCalibration {
  mix: number;
  neuralPerplexity: number;
  memoryPerplexity: number;
  blendedPerplexity: number;
  tokens: number;
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

  // How the current reply was produced: tokens replayed from a learned dataset
  // answer (retrieval) versus tokens drawn from the blended distribution. The
  // studio reports this so dataset recall is never mistaken for the network
  // generating the answer itself.
  private retrievalTokens = 0;
  private blendedTokens = 0;

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

  // Momentum buffers are kept per trainable tensor. A single velocity buffer
  // gives SGD the history it needs to cross the shallow plateaus visible in
  // browser fine-tuning without the 2x extra memory cost of full Adam.
  private optimizerVelocity = new WeakMap<Float32Array, Float32Array>();

  // Gradient buffers for the full backpropagation path (see backwardFull).
  private weightGrads: WeightGradients | null = null;

  // Scratch distributions reused by the training/calibration loops so a step
  // allocates nothing (histories of these were a per-step GC source).
  private scratchNeuralProbs: Float32Array;
  private scratchBlendedProbs: Float32Array;

  // Share of the final sampling distribution that comes from the neural
  // forward pass; the remainder comes from the statistical memory layer.
  //
  // 0.08 is deliberately the *floor*: a randomly initialized network should not
  // be trusted more than the memory table. `calibrateNeuralMix()` fits this on
  // held-out text, so as the network actually learns English its influence
  // rises instead of being capped forever by a hard-coded constant.
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
    this.scratchNeuralProbs = new Float32Array(v);
    this.scratchBlendedProbs = new Float32Array(v);

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

    // Reset optimizer history too: momentum from a previous fine-tune must not
    // leak into a fresh run after the user restores the base checkpoint.
    this.optimizerVelocity = new WeakMap<Float32Array, Float32Array>();

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
    // A full retrain now really does update every base weight (see
    // backwardFull), so the reported trainable count matches the UI's claim;
    // an out-of-date "trainable = lm_head only" number would understate it the
    // same way the old implementation overstated it (ISS-15).
    return {
      total: stats.totalParams,
      trainable: loraMode ? stats.loraParams : stats.totalParams - stats.loraParams,
      loraOnly: stats.loraParams
    };
  }

  /** Current share of the sampling distribution that comes from the network. */
  public getNeuralMix(): number {
    return this.neuralMix;
  }

  /** Set the neural share directly (clamped to [0, 1]). */
  public setNeuralMix(mix: number): void {
    this.neuralMix = Math.min(1, Math.max(0, mix));
  }

  /**
   * How the last generated reply was produced: how many of its tokens were
   * replayed from a learned dataset answer rather than drawn from the blend.
   *
   * Dataset retrieval is what guarantees exact recall of fine-tuned answers, but
   * it also hides whether the network learned anything — so it is reported
   * instead of being silently indistinguishable from generation (ISS-20).
   */
  public getLastGenerationTrace(): GenerationTrace {
    return {
      neuralMix: this.neuralMix,
      retrievalTokens: this.retrievalTokens,
      generatedTokens: this.blendedTokens,
      usedRetrieval: this.retrievalTokens > 0,
    };
  }

  /**
   * Fit the neural/memory blend on text that is not being trained on and keep
   * the best weight.
   *
   * The blend weight used to be the hard-coded constant 0.08, so no matter how
   * much the network learned it could never steer more than 8% of the sampling
   * mass (ISS-16). Here both predictors are scored on real held-out tokens and
   * the blend is chosen by measurement: a network that has learned English gets
   * a share proportional to how much it actually helps, and an untrained one
   * keeps the memory-dominated default.
   */
  public calibrateNeuralMix(
    texts: string[],
    candidates: number[] = [0.08, 0.2, 0.35, 0.5, 0.7],
    useLora = true
  ): NeuralMixCalibration {
    const { vocabSize, maxSeqLen } = this.config;
    const neuralProbs = this.scratchNeuralProbs;
    const memoryProbs = this.scratchBlendedProbs;
    const samples: Array<{ neural: number; memory: number }> = [];
    let neuralSum = 0;
    let memorySum = 0;

    for (const text of texts) {
      const encoded = this.tokenizer.encode(text, true, true);
      const seqTokens = encoded.length > maxSeqLen ? encoded.slice(0, maxSeqLen) : encoded;
      const seqLen = seqTokens.length;
      if (seqLen <= 1) continue;
      const { logits } = this.forward(seqTokens, useLora);
      for (let i = 0; i < seqLen - 1; i++) {
        const target = seqTokens[i + 1];
        if (target < 0 || target >= vocabSize || target === PAD_ID) continue;
        softmax(logits.subarray(i * vocabSize, (i + 1) * vocabSize), neuralProbs, 1.0);
        this.memory.distribution(i >= 1 ? seqTokens[i - 1] : BOS_ID, seqTokens[i], memoryProbs);
        const neural = Math.max(1e-8, neuralProbs[target]);
        const memory = Math.max(1e-8, memoryProbs[target]);
        samples.push({ neural, memory });
        neuralSum += -Math.log(neural);
        memorySum += -Math.log(memory);
      }
    }

    if (samples.length === 0) {
      return {
        mix: this.neuralMix,
        neuralPerplexity: 0,
        memoryPerplexity: 0,
        blendedPerplexity: 0,
        tokens: 0,
      };
    }

    let bestMix = this.neuralMix;
    let bestNll = Infinity;
    for (const mix of candidates) {
      let nll = 0;
      for (const s of samples) {
        nll += -Math.log(Math.max(1e-8, mix * s.neural + (1 - mix) * s.memory));
      }
      nll /= samples.length;
      if (nll < bestNll) {
        bestNll = nll;
        bestMix = mix;
      }
    }

    this.setNeuralMix(bestMix);
    return {
      mix: bestMix,
      neuralPerplexity: Math.exp(Math.min(20, neuralSum / samples.length)),
      memoryPerplexity: Math.exp(Math.min(20, memorySum / samples.length)),
      blendedPerplexity: Math.exp(Math.min(20, bestNll)),
      tokens: samples.length,
    };
  }

  /**
   * Grow the vocabulary in place, preserving every trained weight.
   *
   * Teaching the tokenizer a new word used to rebuild the model from scratch
   * with fresh random weights (and wipe the base snapshot), so training progress
   * could not survive a single dataset edit (ISS-21). Growing the embedding
   * table, the LM head and the memory layer's key space keeps everything that
   * was learned; only the new rows are fresh.
   */
  public resizeVocabulary(newVocabSize: number): boolean {
    const oldVocabSize = this.config.vocabSize;
    if (newVocabSize <= oldVocabSize) return false;
    const { dModel } = this.config;
    const oldVocabFloats = oldVocabSize * dModel;
    const newVocabFloats = newVocabSize * dModel;

    // Embedding rows for the new words start as small random vectors, exactly
    // like every other embedding; the LM head rows mirror that at the output.
    const addRows = (old: Float32Array): Float32Array => {
      const grown = new Float32Array(newVocabFloats);
      grown.set(old);
      grown.set(createRandomNormalMatrix(newVocabSize - oldVocabSize, dModel, 0.03), oldVocabFloats);
      return grown;
    };
    const nextWte = addRows(this.weights.wte);
    const nextLmHead = addRows(this.weights.lm_head);

    // Rebuild the flat base snapshot with the new layout. The middle chunk
    // (wpe, every layer tensor, the final LayerNorm) is vocabulary-independent,
    // so it is copied across untouched; only the two vocab-sized tables change.
    if (this.baseWeightsSnapshot) {
      const oldSnapshot = this.baseWeightsSnapshot;
      const middleLen = oldSnapshot.length - 2 * oldVocabFloats;
      const grown = new Float32Array(oldSnapshot.length + 2 * (newVocabFloats - oldVocabFloats));
      grown.set(oldSnapshot.subarray(0, oldVocabFloats), 0);
      grown.set(nextWte.subarray(oldVocabFloats, newVocabFloats), oldVocabFloats);
      grown.set(oldSnapshot.subarray(oldVocabFloats, oldVocabFloats + middleLen), newVocabFloats);
      const lmOffset = newVocabFloats + middleLen;
      grown.set(oldSnapshot.subarray(oldVocabFloats + middleLen), lmOffset);
      grown.set(nextLmHead.subarray(oldVocabFloats, newVocabFloats), lmOffset + oldVocabFloats);
      this.baseWeightsSnapshot = grown;
    }

    this.weights.wte = nextWte;
    this.weights.lm_head = nextLmHead;
    this.totalWeightFloats += 2 * (newVocabFloats - oldVocabFloats);
    this.config.vocabSize = newVocabSize;

    // Vocab-sized scratch and gradient buffers must be rebuilt, not resized.
    this.scratchLogits = new Float32Array(this.config.maxSeqLen * newVocabSize);
    this.scratchNeuralProbs = new Float32Array(newVocabSize);
    this.scratchBlendedProbs = new Float32Array(newVocabSize);
    this.weightGrads = null;
    if (this.gradScratchReady) {
      this.gradScratchReady = false;
      this.ensureGradScratch();
    }

    // The trigram table keys encode (prev2, prev1) as prev2 * vocabSize + prev1,
    // so the multiplier has to follow the new vocabulary size.
    this.memory.remapVocabSize(newVocabSize);
    return true;
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
    // Full backpropagation holds one gradient buffer per base weight.
    const grads = this.weightGrads;
    if (grads) {
      floats += grads.wte.length + grads.wpe.length + grads.ln_f_gamma.length + grads.ln_f_beta.length + grads.lm_head.length;
      for (const l of grads.layers) {
        floats +=
          l.q_proj.length + l.k_proj.length + l.v_proj.length + l.out_proj.length +
          l.fc1.length + l.fc1_b.length + l.fc2.length + l.fc2_b.length +
          l.ln1_gamma.length + l.ln1_beta.length + l.ln2_gamma.length + l.ln2_beta.length;
      }
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
    const lr = learningRate * ADAPTER_LR_MULTIPLIER;
    const beta = 0.9;
    let velocity = this.optimizerVelocity.get(param);
    if (!velocity) {
      velocity = new Float32Array(param.length);
      this.optimizerVelocity.set(param, velocity);
    }

    // Momentum SGD: average the sequence gradient first, then keep 90% of the
    // previous direction. This is materially faster on the tiny, noisy
    // per-turn gradients produced by browser fine-tuning while retaining a
    // constant one-buffer memory overhead.
    for (let i = 0; i < param.length; i++) {
      const g = grad[i] * invCount;
      velocity[i] = beta * velocity[i] + (1 - beta) * g;
      param[i] -= lr * (velocity[i] + weightDecay * param[i]);
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
   * Allocate one gradient buffer per base weight. Rebuilt from scratch after a
   * vocabulary resize, because the embedding table and LM head change shape.
   */
  private ensureWeightGrads(): WeightGradients {
    if (this.weightGrads) return this.weightGrads;
    const { nLayers, dModel, dFfn, vocabSize, maxSeqLen } = this.config;
    const grads: WeightGradients = {
      wte: new Float32Array(vocabSize * dModel),
      wpe: new Float32Array(maxSeqLen * dModel),
      layers: [],
      ln_f_gamma: new Float32Array(dModel),
      ln_f_beta: new Float32Array(dModel),
      lm_head: new Float32Array(vocabSize * dModel),
    };
    for (let l = 0; l < nLayers; l++) {
      grads.layers.push({
        q_proj: new Float32Array(dModel * dModel),
        k_proj: new Float32Array(dModel * dModel),
        v_proj: new Float32Array(dModel * dModel),
        out_proj: new Float32Array(dModel * dModel),
        fc1: new Float32Array(dFfn * dModel),
        fc1_b: new Float32Array(dFfn),
        fc2: new Float32Array(dModel * dFfn),
        fc2_b: new Float32Array(dModel),
        ln1_gamma: new Float32Array(dModel),
        ln1_beta: new Float32Array(dModel),
        ln2_gamma: new Float32Array(dModel),
        ln2_beta: new Float32Array(dModel),
      });
    }
    this.weightGrads = grads;
    return grads;
  }

  /** Zero every base-weight gradient before a backward pass accumulates into it. */
  private zeroWeightGrads(g: WeightGradients): void {
    g.wte.fill(0);
    g.wpe.fill(0);
    g.ln_f_gamma.fill(0);
    g.ln_f_beta.fill(0);
    g.lm_head.fill(0);
    for (const l of g.layers) {
      l.q_proj.fill(0);
      l.k_proj.fill(0);
      l.v_proj.fill(0);
      l.out_proj.fill(0);
      l.fc1.fill(0);
      l.fc1_b.fill(0);
      l.fc2.fill(0);
      l.fc2_b.fill(0);
      l.ln1_gamma.fill(0);
      l.ln1_beta.fill(0);
      l.ln2_gamma.fill(0);
      l.ln2_beta.fill(0);
    }
  }

  /**
   * Apply one accumulated gradient to a base weight:
   * `w -= lr * (grad / positions * clip + wd * w)`.
   * The gradient is summed over the sequence, so it is divided by the number of
   * positions to keep the step size independent of sequence length.
   */
  private applyFullUpdate(
    param: Float32Array,
    grad: Float32Array,
    invCount: number,
    learningRate: number,
    weightDecay: number,
    clipScale: number
  ): void {
    const beta = 0.9;
    let velocity = this.optimizerVelocity.get(param);
    if (!velocity) {
      velocity = new Float32Array(param.length);
      this.optimizerVelocity.set(param, velocity);
    }
    for (let i = 0; i < param.length; i++) {
      const g = grad[i] * invCount * clipScale;
      velocity[i] = beta * velocity[i] + (1 - beta) * g;
      param[i] -= learningRate * (velocity[i] + weightDecay * param[i]);
    }
  }

  /**
   * Full backpropagation: the gradient of the token cross-entropy w.r.t. every
   * weight in the network — embeddings, all four projections, both feed-forward
   * matrices, every LayerNorm and the LM head — followed by one SGD step.
   *
   * This is what makes "Full Fine-Tuning" true. The previous implementation
   * updated `lm_head` rows and nothing else, so every other tensor stayed at its
   * random initialization forever and the network could never learn English
   * structure no matter how long it trained (ISS-14/ISS-15/ISS-18).
   *
   * Every gradient is accumulated first and applied at the end: an update
   * applied mid-pass would change the weights the later gradients are derived
   * from, so the activations cached by the forward pass would no longer describe
   * the network being differentiated.
   */
  private backwardFull(
    tokens: number[],
    seqLen: number,
    learningRate: number,
    weightDecay: number,
    targetCount: number,
    lossStartIndex = 0
  ): void {
    const cache = this.activationCache;
    if (!cache) return;
    this.ensureGradScratch();
    const grads = this.ensureWeightGrads();
    this.zeroWeightGrads(grads);

    const { dModel, nHeads, nLayers, dFfn, vocabSize } = this.config;
    const headDim = Math.floor(dModel / nHeads);
    const attnScale = 1.0 / Math.sqrt(headDim);
    const tokenArea = seqLen * dModel;
    // Must match forward(): positions are window-relative (see forward()).
    const positionOffset = 0;

    // 1. dL/dlogits = softmax(logits) − onehot(target). Accumulate the LM head
    //    gradient from the final LayerNorm activations and push the gradient
    //    into that LayerNorm's output.
    this.gNorm.fill(0);
    for (let i = lossStartIndex; i < seqLen - 1; i++) {
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
          grads.lm_head[vOffset + d] += grad * this.scratchFinalNorm[rowOffset + d];
          this.gNorm[rowOffset + d] += grad * this.weights.lm_head[vOffset + d];
        }
      }
    }

    // 2. Final LayerNorm: gamma/beta gradients and the gradient w.r.t. the
    //    residual stream the blocks produced.
    for (let i = 0; i < seqLen; i++) {
      const rowOffset = i * dModel;
      layerNormBackward(cache.finalResidual, rowOffset, this.weights.ln_f_gamma, this.gNorm, rowOffset, this.gResidual, rowOffset, dModel);
      layerNormBackwardParam(cache.finalResidual, rowOffset, this.gNorm, rowOffset, grads.ln_f_gamma, grads.ln_f_beta, dModel);
    }

    // 3. Transformer blocks, top down.
    for (let l = nLayers - 1; l >= 0; l--) {
      const layer = this.weights.layers[l];
      const g = grads.layers[l];
      const blockInput = cache.blockInput[l];
      const norm1 = cache.norm1[l];
      const cachedQ = cache.q[l];
      const cachedK = cache.k[l];
      const cachedV = cache.v[l];
      const x1 = cache.x1[l];
      const norm2 = cache.norm2[l];

      // --- Feed-forward: fc2, fc2_b, fc1, fc1_b and LN2 ---
      for (let i = 0; i < seqLen; i++) {
        const rowOffset = i * dModel;
        // Recompute the pre-activation from the cached LN2 output. The weights
        // are still the ones the forward pass used, because every update is
        // applied after this whole pass.
        for (let r = 0; r < dFfn; r++) {
          let pre = layer.fc1_b[r];
          const rOffset = r * dModel;
          for (let c = 0; c < dModel; c++) pre += layer.fc1[rOffset + c] * norm2[rowOffset + c];
          this.gFfnPre[r] = pre;
        }
        // dL/dW2 = dResidual ⊗ gelu(pre)
        for (let r = 0; r < dFfn; r++) {
          const mid = gelu(this.gFfnPre[r]);
          for (let row = 0; row < dModel; row++) {
            g.fc2[row * dFfn + r] += this.gResidual[rowOffset + row] * mid;
          }
        }
        for (let row = 0; row < dModel; row++) g.fc2_b[row] += this.gResidual[rowOffset + row];
        // dL/dpre = (fc2ᵀ dResidual) ⊙ gelu'(pre)
        for (let r = 0; r < dFfn; r++) {
          let dMid = 0;
          for (let row = 0; row < dModel; row++) {
            dMid += this.gResidual[rowOffset + row] * layer.fc2[row * dFfn + r];
          }
          this.gFfnPre[r] = dMid * geluDerivative(this.gFfnPre[r]);
        }
        // dL/dW1 = dPre ⊗ LN2(x1), dL/dB1 = dPre, dL/dLN2 = fc1ᵀ dPre
        for (let c = 0; c < dModel; c++) {
          let sum = 0;
          for (let r = 0; r < dFfn; r++) {
            const dPre = this.gFfnPre[r];
            g.fc1[r * dModel + c] += dPre * norm2[rowOffset + c];
            sum += dPre * layer.fc1[r * dModel + c];
          }
          this.gXNorm2Row[c] = sum;
        }
        for (let r = 0; r < dFfn; r++) g.fc1_b[r] += this.gFfnPre[r];
        layerNormBackward(x1, rowOffset, layer.ln2_gamma, this.gXNorm2Row, 0, this.gX1, rowOffset, dModel);
        layerNormBackwardParam(x1, rowOffset, this.gXNorm2Row, 0, g.ln2_gamma, g.ln2_beta, dModel);
      }
      // The residual path carries the incoming gradient through unchanged.
      for (let idx = 0; idx < tokenArea; idx++) this.gX1[idx] += this.gResidual[idx];

      // --- Attention: out_proj, softmax, then the q/k/v projections ---
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
      // dL/dout_proj = gProjOut ⊗ attnOut
      for (let i = 0; i < seqLen; i++) {
        const rowOffset = i * dModel;
        for (let row = 0; row < dModel; row++) {
          const gRow = this.gProjOut[rowOffset + row];
          if (gRow === 0) continue;
          const rOffset = row * dModel;
          for (let c = 0; c < dModel; c++) {
            g.out_proj[rOffset + c] += gRow * this.scratchAttnOut[rowOffset + c];
          }
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
          for (let j = 0; j <= i; j++) {
            const w = weights[j];
            if (w === 0) continue;
            const vjOffset = j * dModel + headOffset;
            for (let d = 0; d < headDim; d++) {
              this.gV[vjOffset + d] += w * this.gAttnOut[outOffset + d];
            }
          }
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
      // dL/dq_proj = gQ ⊗ LN1(x), likewise k/v; dL/dLN1 = QᵀgQ + KᵀgK + VᵀgV
      for (let i = 0; i < seqLen; i++) {
        const rowOffset = i * dModel;
        for (let row = 0; row < dModel; row++) {
          const gq = this.gQ[rowOffset + row];
          const gk = this.gK[rowOffset + row];
          const gv = this.gV[rowOffset + row];
          if (gq === 0 && gk === 0 && gv === 0) continue;
          const rOffset = row * dModel;
          for (let c = 0; c < dModel; c++) {
            const x = norm1[rowOffset + c];
            g.q_proj[rOffset + c] += gq * x;
            g.k_proj[rOffset + c] += gk * x;
            g.v_proj[rOffset + c] += gv * x;
          }
        }
      }
      for (let i = 0; i < seqLen; i++) {
        const rowOffset = i * dModel;
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
        layerNormBackward(blockInput, rowOffset, layer.ln1_gamma, this.gXNorm1, rowOffset, this.gIn, rowOffset, dModel);
        layerNormBackwardParam(blockInput, rowOffset, this.gXNorm1, rowOffset, g.ln1_gamma, g.ln1_beta, dModel);
      }
      for (let idx = 0; idx < tokenArea; idx++) this.gIn[idx] += this.gX1[idx];

      copyInto(this.gResidual, this.gIn, tokenArea);
    }

    // 4. Token and position embedding gradients: every position contributed to
    //    the residual stream that entered the first block.
    for (let i = 0; i < seqLen; i++) {
      let tokenId = tokens[i];
      if (tokenId < 0 || tokenId >= vocabSize || !Number.isFinite(tokenId)) tokenId = UNK_ID;
      const wteOffset = tokenId * dModel;
      const wpeOffset = (positionOffset + i) * dModel;
      const rowOffset = i * dModel;
      for (let d = 0; d < dModel; d++) {
        const gIn = this.gIn[rowOffset + d];
        grads.wte[wteOffset + d] += gIn;
        grads.wpe[wpeOffset + d] += gIn;
      }
    }

    // 5. Clip the global gradient norm so a single short sequence can never
    //    take an unbounded step, then apply one SGD step to every weight.
    let normSq = 0;
    const accumulateNorm = (buf: Float32Array) => {
      for (let i = 0; i < buf.length; i++) normSq += buf[i] * buf[i];
    };
    accumulateNorm(grads.wte);
    accumulateNorm(grads.wpe);
    accumulateNorm(grads.ln_f_gamma);
    accumulateNorm(grads.ln_f_beta);
    accumulateNorm(grads.lm_head);
    for (const l of grads.layers) {
      accumulateNorm(l.q_proj);
      accumulateNorm(l.k_proj);
      accumulateNorm(l.v_proj);
      accumulateNorm(l.out_proj);
      accumulateNorm(l.fc1);
      accumulateNorm(l.fc1_b);
      accumulateNorm(l.fc2);
      accumulateNorm(l.fc2_b);
      accumulateNorm(l.ln1_gamma);
      accumulateNorm(l.ln1_beta);
      accumulateNorm(l.ln2_gamma);
      accumulateNorm(l.ln2_beta);
    }
    const gradNorm = Math.sqrt(normSq);
    const clipScale = gradNorm > GRADIENT_CLIP_NORM ? GRADIENT_CLIP_NORM / gradNorm : 1.0;

    const invCount = targetCount > 0 ? 1 / targetCount : 0;
    const update = (param: Float32Array, grad: Float32Array) =>
      this.applyFullUpdate(param, grad, invCount, learningRate, weightDecay, clipScale);

    update(this.weights.wte, grads.wte);
    update(this.weights.wpe, grads.wpe);
    update(this.weights.ln_f_gamma, grads.ln_f_gamma);
    update(this.weights.ln_f_beta, grads.ln_f_beta);
    update(this.weights.lm_head, grads.lm_head);
    for (let l = 0; l < nLayers; l++) {
      const layer = this.weights.layers[l];
      const g = grads.layers[l];
      update(layer.q_proj, g.q_proj);
      update(layer.k_proj, g.k_proj);
      update(layer.v_proj, g.v_proj);
      update(layer.out_proj, g.out_proj);
      update(layer.fc1, g.fc1);
      update(layer.fc1_b, g.fc1_b);
      update(layer.fc2, g.fc2);
      update(layer.fc2_b, g.fc2_b);
      update(layer.ln1_gamma, g.ln1_gamma);
      update(layer.ln1_beta, g.ln1_beta);
      update(layer.ln2_gamma, g.ln2_gamma);
      update(layer.ln2_beta, g.ln2_beta);
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
    // Positions are window-relative: the first token of the window the model is
    // shown sits at position 0. This is deliberate and is what ISS-23 called
    // into question. Anchoring positions to the *end* of the window instead
    // would re-number every token the moment one more token is decoded, so a
    // prompt would drift through the position table as its reply grows, and the
    // positions a finished sequence was trained with could never be reproduced.
    // Window-relative positions stay fixed while the reply grows, and they match
    // training exactly (a training turn also starts at position 0), which is
    // what lets the sliding window drop the oldest turns without re-positioning
    // the rest. The genuinely broken half of ISS-23 was that every embedding
    // here was frozen at its random initialization — that is fixed by training
    // `wte`/`wpe` in backwardFull.
    const positionOffset = 0;
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
      const wpeOffset = (positionOffset + i) * dModel;
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
   */  public trainStep(
    tokens: number[],
    learningRate = 0.01,
    loraMode = true,
    weightDecay = 0.005,
    observeMemory = true
  ): TrainStepResult {
    const { vocabSize, loraRank, maxSeqLen } = this.config;
    // Training scores a fixed-length prefix and hands it to forward() as-is, so
    // the activations this step differentiates are exactly the ones the loss
    // was computed from (forward() itself windows the tail for inference, which
    // would misalign the target shift).
    const seqTokens = tokens.length > maxSeqLen ? tokens.slice(0, maxSeqLen) : tokens;
    const seqLen = seqTokens.length;
    if (seqLen <= 1) return { loss: 0, perplexity: 1.0, neuralLoss: 0, blendedLoss: 0 };

    // Both adaptation modes now differentiate real activations: LoRA through
    // the frozen body into the adapters, a full retrain through every weight.
    const loraTraining = loraMode && loraRank > 0;

    // Conversational fine-tuning should optimize the assistant response, not
    // spend most of its capacity relearning the user's prompt and control
    // tokens. Warm-up/plain-language sequences have no ASSISTANT marker and
    // therefore keep the original full-sequence objective.
    const assistantIdx = seqTokens.lastIndexOf(ASSISTANT_ID);
    const lossStartIndex = assistantIdx >= 0 ? assistantIdx + 1 : 0;
     // Forward pass, retaining the activations both backward passes need.
    const { logits } = this.forward(seqTokens, loraMode, true);

    // A full retrain leaves the LoRA adapters at zero, so record the adaptation
    // for isFineTuned() (resetToBase() clears it).
    if (!loraMode) {
      this.fullFineTuneApplied = true;
    }

    let totalLoss = 0;
    let totalBlended = 0;
    let targetCount = 0;

    const neuralProbs = this.scratchNeuralProbs;
    const mixed = this.scratchBlendedProbs;
    const mix = this.neuralMix;
    for (let i = lossStartIndex; i < seqLen - 1; i++) {
      let targetToken = seqTokens[i + 1];
      if (targetToken < 0 || targetToken >= vocabSize || !Number.isFinite(targetToken)) {
        targetToken = UNK_ID;
      }
      if (targetToken === PAD_ID) continue;

      // Neural softmax at position i: the distribution the weights own, so it
      // is what both the reported loss and the gradient below measure.
      const logitRow = logits.subarray(i * vocabSize, (i + 1) * vocabSize);
      softmax(logitRow, neuralProbs, 1.0);
      totalLoss += -Math.log(Math.max(1e-8, neuralProbs[targetToken]));

      // The blended probability generation actually samples from is reported
      // separately. It is a diagnostic, never the objective: the memory tables
      // are constant with respect to the weights, so differentiating the blend
      // mostly teaches the network to imitate the tables (ISS-18/ISS-19).
      const prev1 = seqTokens[i];
      const prev2 = i >= 1 ? seqTokens[i - 1] : BOS_ID;
      this.memory.distribution(prev2, prev1, mixed);
      const blendedTarget = mix * neuralProbs[targetToken] + (1 - mix) * mixed[targetToken];
      totalBlended += -Math.log(Math.max(1e-8, blendedTarget));
      targetCount++;
    }

    // One real backward pass over this sequence: LoRA differentiates through
    // the frozen body into the adapters, a full retrain differentiates every
    // weight in the network.
    if (targetCount > 0) {
      if (loraTraining) {
        this.backwardLora(seqTokens, seqLen, learningRate, weightDecay, targetCount, lossStartIndex);
      } else {
        this.backwardFull(seqTokens, seqLen, learningRate, weightDecay, targetCount, lossStartIndex);
      }
    }

    // Only now may the memory observe the sequence. Observing first (the old
    // order) let the tables memorize the very tokens this step was scored
    // against, so the reported loss measured the table being filled rather than
    // the network learning anything (ISS-17). Plain language-modelling warm-up
    // passes opt out entirely, so the dialogue memory stays conversational
    // instead of being diluted with sentence fragments.
    if (observeMemory) {
      this.observeTrainingSequence(seqTokens, seqLen);
    }

    const avgLoss = targetCount > 0 ? totalLoss / targetCount : 0;
    const avgBlended = targetCount > 0 ? totalBlended / targetCount : 0;
    return {
      loss: avgLoss,
      perplexity: Math.min(9999, Math.exp(Math.min(10, avgLoss))),
      neuralLoss: avgLoss,
      blendedLoss: avgBlended,
    };
  }

  /**
   * Teach the statistical memory layer a sequence the network was just
   * trained on: its n-gram tables, the dialogue link that opens a reply, and
   * the exact question-to-answer case used for dataset recall.
   */
  private observeTrainingSequence(tokens: number[], seqLen: number): void {
    this.memory.observe(tokens.slice(0, seqLen), 2.0);

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
      this.memory.rememberCase(
        this.caseWords(this.userContentTokens(tokens, asstIdx)),
        tokens.slice(asstIdx + 2, seqLen)
      );
    }
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
        openerWeight = OPENER_LINK_WEIGHT;
      } else if (this.memory.openerDistribution(opener)) {
        openerProbs = opener;
        openerWeight = OPENER_GENERIC_WEIGHT;
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
        // Guardrail, not a generator: this keeps a stored dataset answer on
        // track while it is being replayed, and the trace below reports those
        // tokens as retrieval rather than as the network producing them.
        p = v === caseToken
          ? CASE_REPLAY_STRENGTH + (1 - CASE_REPLAY_STRENGTH) * p
          : (1 - CASE_REPLAY_STRENGTH) * p;
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

    // Stay on the learned answer until generation diverges from it, counting
    // how much of the reply came from that stored answer.
    if (caseToken !== null) {
      if (sample.chosenId === caseToken) {
        this.casePos++;
        this.retrievalTokens++;
      } else {
        this.caseReply = null;
        this.blendedTokens++;
      }
    } else {
      this.blendedTokens++;
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
    this.retrievalTokens = 0;
    this.blendedTokens = 0;
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
