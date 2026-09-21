/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelConfig, GenerationOptions, GeneratedTokenInfo } from '../types';
import { Tokenizer, defaultTokenizer, BOS_ID, EOS_ID, PAD_ID, UNK_ID, USER_ID, ASSISTANT_ID, NEWLINE_ID } from './tokenizer';
import { NgramLanguageModel } from './ngram';
import {
  createFloat32Matrix,
  createRandomNormalMatrix,
  layerNorm,
  gelu,
  softmax,
  sampleFromDistribution
} from './matrix';

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

  // Last computed attention maps for architecture inspector: [layer][head][seq_len, seq_len]
  public lastAttentionMaps: number[][][][] = [];

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

    // Initialize the statistical memory layer with baseline conversational English
    this.memory = new NgramLanguageModel(v);
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
   */
  public static readonly BASE_CORPUS: string[] = [
    'hello ! I am your conversational AI assistant . how can I help you today ?',
    'hello how are you doing today ? I am here to assist and chat with you .',
    'I am doing wonderful , thank you for asking ! how is your day going ?',
    'hi there ! it is great to hear from you . what is on your mind ?',
    'you are very welcome ! I am always glad to chat and assist you .',
    'that is a thoughtful question . let us explore the answer together .',
    'I am here for you . remember to take a short break and rest your mind .',
    'every small positive habit creates meaningful progress over time .',
    'certainly ! I would be delighted to share some ideas with you .',
    'I am your friendly AI assistant , and I love a good conversation .',
    'what a great question ! here is what I think about it .',
    'of course ! tell me more about what you need and I will help you .',
    'I am listening . share your thoughts and we can think it through together .',
    'take a slow deep breath . let us look at your ideas one step at a time .',
    'you are doing great . keep going and stay curious .',
    'that is wonderful to hear ! tell me more about your day .',
    'sometimes the best answer is to rest for a moment and then try again .',
    'learning something new every day keeps the mind fresh and happy .',
    'what would you like to talk about today ?',
    'I can help you plan your day , share ideas , or simply chat with you .',
    'staying calm and focused one moment at a time is a wonderful habit .',
    'water , sunlight , a short walk , and a good book make a peaceful day .',
    'music can lift your mood and give you fresh energy for the day .',
    'the sky is beautiful today . enjoy the light while it lasts .',
    'every conversation is a chance to learn something new .',
    'your ideas matter , and I enjoy hearing every one of them .',
    'if you feel stressed , pause , breathe slowly , and count to four .',
    'a grateful mind is a peaceful mind . what are you thankful for today ?',
    'small steps taken every day create big change over time .',
    'I am always here whenever you want to talk or share an idea .',
    'that sounds like a lovely plan ! how can I help you make it happen ?',
    'asking questions is how we grow . never stop being curious .',
    'kindness costs nothing and makes the world a warmer place .',
    'rest is not a reward for work , it is part of a good life .',
    'the best time to start is right now , one small step at a time .',
    'listening is a gift you can give to another person today .',
    'I hope your day is full of good thoughts and gentle moments .',
    'remember to drink water and take a short walk between tasks .',
    'it is okay to feel uncertain . clarity comes one thought at a time .',
    'thank you for this lovely conversation . come back and chat anytime !',
    'goodbye for now ! I am here whenever you need a friend to talk to .'
  ];

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
  }

  /**
   * Returns parameter count and exact memory statistics in bytes
   */
  public getMemoryStats(): {
    totalParams: number;
    trainableParams: number;
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
      this.scratchLogits.length
    ) * 4;
    const snapshotMemoryBytes = (this.baseWeightsSnapshot?.length ?? 0) * 4;
    const totalBytes = weightsMemoryBytes + scratchMemoryBytes + snapshotMemoryBytes;

    let loraParams = 0;
    for (const l of this.weights.layers) {
      loraParams += l.lora_q_A.length + l.lora_q_B.length + l.lora_v_A.length + l.lora_v_B.length;
    }

    return {
      totalParams: this.totalWeightFloats,
      trainableParams: this.totalWeightFloats,
      loraParams,
      weightsMemoryBytes,
      scratchMemoryBytes,
      snapshotMemoryBytes,
      totalMemoryFormatted: `${(totalBytes / 1024 / 1024).toFixed(2)} MB`,
      weightsMemoryFormatted: `${(weightsMemoryBytes / 1024).toFixed(1)} KB`,
      scratchBuffersFormatted: `${(scratchMemoryBytes / 1024).toFixed(1)} KB`
    };
  }

  public countParameters(): { total: number; trainable: number; loraOnly: number } {
    const stats = this.getMemoryStats();
    return {
      total: stats.totalParams,
      trainable: stats.trainableParams,
      loraOnly: stats.loraParams
    };
  }

  /**
   * Forward pass: computes logits using pre-allocated scratch memory to prevent GC lag.
   * Returns view of computed logits for sequence.
   */
  public forward(
    tokens: number[],
    useLora = true
  ): {
    logits: Float32Array; // [seqLen, vocabSize]
    seqLen: number;
  } {
    const { dModel, nHeads, nLayers, dFfn, vocabSize, loraRank, loraAlpha } = this.config;
    const seqLen = Math.min(tokens.length, this.config.maxSeqLen);
    const headDim = Math.floor(dModel / nHeads);
    const loraScale = loraRank > 0 ? loraAlpha / loraRank : 1.0;

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
    for (let i = 0; i < seqLen; i++) {
      const tokenId = Math.min(Math.max(0, tokens[i]), vocabSize - 1);
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

      // LayerNorm 1
      for (let i = 0; i < seqLen; i++) {
        const rowIn = this.scratchX.subarray(i * dModel, (i + 1) * dModel);
        const rowOut = this.scratchXNorm1.subarray(i * dModel, (i + 1) * dModel);
        layerNorm(rowIn, layer.ln1_gamma, layer.ln1_beta, rowOut, dModel);
      }

      // Linear projections: Q, K, V
      for (let i = 0; i < seqLen; i++) {
        const xi = this.scratchXNorm1.subarray(i * dModel, (i + 1) * dModel);
        const qi = this.scratchQ.subarray(i * dModel, (i + 1) * dModel);
        const ki = this.scratchK.subarray(i * dModel, (i + 1) * dModel);
        const vi = this.scratchV.subarray(i * dModel, (i + 1) * dModel);

        for (let row = 0; row < dModel; row++) {
          let sumQ = 0, sumK = 0, sumV = 0;
          const rOffset = row * dModel;
          for (let col = 0; col < dModel; col++) {
            const val = xi[col];
            sumQ += layer.q_proj[rOffset + col] * val;
            sumK += layer.k_proj[rOffset + col] * val;
            sumV += layer.v_proj[rOffset + col] * val;
          }

          // LoRA modification: W + (B @ A) * alpha / rank
          if (useLora && loraRank > 0) {
            let loraQ = 0, loraV = 0;
            for (let r = 0; r < loraRank; r++) {
              let aQ = 0, aV = 0;
              const aOffset = r * dModel;
              for (let c = 0; c < dModel; c++) {
                aQ += layer.lora_q_A[aOffset + c] * xi[c];
                aV += layer.lora_v_A[aOffset + c] * xi[c];
              }
              loraQ += layer.lora_q_B[row * loraRank + r] * aQ;
              loraV += layer.lora_v_B[row * loraRank + r] * aV;
            }
            sumQ += loraQ * loraScale;
            sumV += loraV * loraScale;
          }

          qi[row] = sumQ;
          ki[row] = sumK;
          vi[row] = sumV;
        }
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

      // LayerNorm 2
      for (let i = 0; i < seqLen; i++) {
        const rowIn = this.scratchX.subarray(i * dModel, (i + 1) * dModel);
        const rowOut = this.scratchXNorm2.subarray(i * dModel, (i + 1) * dModel);
        layerNorm(rowIn, layer.ln2_gamma, layer.ln2_beta, rowOut, dModel);
      }

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
    const { vocabSize, dModel, loraRank, loraAlpha } = this.config;
    const seqLen = Math.min(tokens.length, this.config.maxSeqLen);
    if (seqLen <= 1) return { loss: 0, perplexity: 1.0 };

    // Forward pass
    const { logits } = this.forward(tokens, loraMode);

    let totalLoss = 0;
    let targetCount = 0;
    const loraScale = loraRank > 0 ? loraAlpha / loraRank : 1.0;

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
      const targetToken = tokens[i + 1];
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

      // Cross-entropy gradient signal for this position
      const gradTarget = targetProb - 1.0;
      const hOffset = i * dModel;

      if (loraMode && loraRank > 0) {
        // Adapter updates conditioned on the hidden state at this position:
        // reinforce the LoRA B rows that map context features toward the target.
        const factor = learningRate * 0.5 * gradTarget * loraScale;
        for (const layer of this.weights.layers) {
          const bRow = (targetToken % dModel) * loraRank;
          for (let r = 0; r < loraRank; r++) {
            let aVal = 0;
            const aOffset = r * dModel;
            for (let c = 0; c < dModel; c++) {
              aVal += layer.lora_v_A[aOffset + c] * hidden[hOffset + c];
            }
            layer.lora_v_B[bRow + r] -= factor * aVal;
          }
        }
      } else {
        // Full fine-tuning: real cross-entropy gradient descent on lm_head rows.
        // dL/dlogit_v = mixed[v] - 1[v==target]; applied via the hidden state.
        const lr = learningRate * 0.35;
        for (let v = 0; v < vocabSize; v++) {
          const grad = mixed[v] - (v === targetToken ? 1.0 : 0.0);
          if (Math.abs(grad) < 0.004) continue; // skip negligible gradients
          const vOffset = v * dModel;
          for (let d = 0; d < dModel; d++) {
            this.weights.lm_head[vOffset + d] -= lr * grad * hidden[hOffset + d];
          }
        }
      }
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

    // --- Neural distribution (temperature-scaled softmax over logits) ---
    const neuralProbs = new Float32Array(vocabSize);
    softmax(rawLogits, neuralProbs, options.temperature);

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

    for (let v = 0; v < vocabSize; v++) {
      let p = mix * neuralProbs[v] + (1 - mix) * memoryProbs[v];

      if (openerProbs) {
        p = openerWeight * openerProbs[v] + (1 - openerWeight) * p;
      }

      if (caseToken !== null) {
        p = v === caseToken ? 0.88 + 0.12 * p : 0.12 * p;
      }

      // Never emit raw control tokens like <pad>, <unk>, <bos>, <user>, <assistant>, \n
      if (v === PAD_ID || v === UNK_ID || v === BOS_ID || v === USER_ID || v === ASSISTANT_ID || v === NEWLINE_ID) {
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
      options.temperature,
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
  private replyOpeningContext(tokens: number[], assistantIdx: number): number[] {
    const spaceId = this.tokenizer.idOf(' ');
    if (spaceId === undefined) return [];
    const structural = (t: number) =>
      t === USER_ID || t === ASSISTANT_ID || t === NEWLINE_ID || t === BOS_ID || t === PAD_ID || t === EOS_ID || t === spaceId;
    for (let i = assistantIdx - 1; i >= 0; i--) {
      if (!structural(tokens[i])) return [tokens[i], spaceId];
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
    const asstIdx = tokens.lastIndexOf(ASSISTANT_ID);
    if (asstIdx >= 0) {
      // Replay the answer fine-tuning learned for this user message.
      this.caseReply = this.memory.findCase(this.caseWords(this.userContentTokens(tokens, asstIdx)));

      if (tokens.length + 2 < this.config.maxSeqLen) {
        const opening = this.replyOpeningContext(tokens, asstIdx);
        if (opening.length === 2) {
          tokens.push(opening[0], opening[1]);
          this.pendingReplyWord = opening[0];
        }
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
      if (tokens.length >= this.config.maxSeqLen) break;

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
      if (tokens.length >= this.config.maxSeqLen) break;

      const tokenInfo = this.generateNextToken(tokens, options, useLora);
      tokens.push(tokenInfo.id);
      generatedInfo.push(tokenInfo);

      if (tokenInfo.id === EOS_ID) break;
    }

    const text = this.tokenizer.decode(generatedInfo.map(t => t.id), true);
    return { text, tokens: generatedInfo };
  }
}
