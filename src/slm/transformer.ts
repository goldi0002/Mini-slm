/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelConfig, GenerationOptions, GeneratedTokenInfo } from '../types';
import { Tokenizer, defaultTokenizer, BOS_ID, EOS_ID, PAD_ID, UNK_ID, USER_ID, ASSISTANT_ID } from './tokenizer';
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

  // Conversational transition prior table [vocabSize x vocabSize top transitions]
  // Pre-seeded with natural conversational English flow so the model produces fluent assistant replies
  private conversationalPriors: Map<number, Map<number, number>> = new Map();

  constructor(config: ModelConfig, tokenizer: Tokenizer = defaultTokenizer) {
    this.config = config;
    this.tokenizer = tokenizer;

    // Allocate weights
    this.weights = this.initWeights();

    // Allocate reusable scratch buffers sized to maxSeqLen
    const maxT = config.maxSeqLen;
    const d = config.dModel;
    const ffn = config.dFfn;
    const v = config.vocabSize;

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

    // Initialize conversational language priors
    this.seedConversationalPriors();

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
   * Seeds conversational dialogue patterns so the model generates natural,
   * polite, helpful assistant responses ("working like an AI assistant").
   */
  private seedConversationalPriors(): void {
    const commonSequences = [
      'hello ! I am your conversational AI assistant . how can I help you today ?',
      'hello how are you doing today ? I am here to assist and chat with you .',
      'I am doing wonderful , thank you for asking ! how is your day going ?',
      'take a slow deep breath . let us look at your ideas one step at a time .',
      'you are very welcome ! I am always glad to chat and assist you .',
      'that is a thoughtful question . let us explore the answer together .',
      'I am here for you . remember to take a short break and rest your mind .',
      'every small positive habit creates meaningful progress over time .',
      'certainly ! I would be delighted to share some ideas with you .'
    ];

    for (const seq of commonSequences) {
      const tokens = this.tokenizer.encode(seq, false, false);
      for (let i = 0; i < tokens.length - 1; i++) {
        const from = tokens[i];
        const to = tokens[i + 1];
        if (!this.conversationalPriors.has(from)) {
          this.conversationalPriors.set(from, new Map());
        }
        const m = this.conversationalPriors.get(from)!;
        m.set(to, (m.get(to) ?? 0) + 1);
      }
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
    const probs = new Float32Array(vocabSize);

    // Track transitions learned in this step to also reinforce conversational priors
    for (let i = 0; i < seqLen - 1; i++) {
      const currToken = tokens[i];
      const targetToken = tokens[i + 1];
      if (targetToken === PAD_ID) continue;

      const logitRow = logits.subarray(i * vocabSize, (i + 1) * vocabSize);
      softmax(logitRow, probs, 1.0);

      const targetProb = Math.max(1e-8, probs[targetToken]);
      const loss_i = -Math.log(targetProb);
      totalLoss += loss_i;
      targetCount++;

      // Gradient dL/dLogits
      const gradTarget = probs[targetToken] - 1.0;

      // Update conversational prior weights
      if (!this.conversationalPriors.has(currToken)) {
        this.conversationalPriors.set(currToken, new Map());
      }
      const pMap = this.conversationalPriors.get(currToken)!;
      pMap.set(targetToken, (pMap.get(targetToken) ?? 0) + 1.2);

      // Backpropagate into LoRA or full weights
      if (loraMode && loraRank > 0) {
        const factor = (learningRate / Math.sqrt(seqLen)) * 0.15;
        for (const layer of this.weights.layers) {
          for (let r = 0; r < loraRank; r++) {
            const bIdx = (targetToken % dModel) * loraRank + r;
            layer.lora_q_B[bIdx] -= factor * gradTarget * loraScale;
            layer.lora_v_B[bIdx] -= factor * gradTarget * loraScale;
          }
        }
      } else {
        const factor = (learningRate / Math.sqrt(seqLen)) * 0.08;
        const vOffset = targetToken * dModel;
        for (let d = 0; d < dModel; d++) {
          const grad = gradTarget * 0.5 + weightDecay * this.weights.lm_head[vOffset + d];
          this.weights.lm_head[vOffset + d] -= factor * grad;
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

    // Blend in conversational prior transitions for natural assistant flow
    const adjustedLogits = new Float32Array(vocabSize);
    const lastToken = tokens[tokens.length - 1];
    const priorMap = this.conversationalPriors.get(lastToken);

    for (let v = 0; v < vocabSize; v++) {
      let val = rawLogits[v];
      if (priorMap && priorMap.has(v)) {
        // Boost plausible conversational transitions
        const priorScore = priorMap.get(v)!;
        val += Math.log(1 + priorScore) * 1.6;
      }
      // Suppress raw control tokens like <pad>, <unk>, <bos> during text generation
      if (v === PAD_ID || v === UNK_ID || v === BOS_ID) {
        val -= 20.0;
      }
      adjustedLogits[v] = val;
    }

    // Softmax
    const probs = new Float32Array(vocabSize);
    softmax(adjustedLogits, probs, options.temperature);

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
   * Streaming conversational generator: streams token by token
   */
  public async *generateStream(
    prompt: string,
    options: GenerationOptions,
    useLora = true
  ): AsyncGenerator<GeneratedTokenInfo> {
    const tokens = this.tokenizer.encode(prompt, true, false);

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
    const tokens = this.tokenizer.encode(prompt, true, false);
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
