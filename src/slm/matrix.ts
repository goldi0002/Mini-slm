/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Fast Vector and Matrix Math operations implemented with Float32Array
 * specifically designed for Small Language Model inference and backpropagation in TypeScript.
 */

export function createFloat32Matrix(rows: number, cols: number, initVal = 0): Float32Array {
  const arr = new Float32Array(rows * cols);
  if (initVal !== 0) {
    arr.fill(initVal);
  }
  return arr;
}

export function createRandomNormalMatrix(rows: number, cols: number, std = 0.02): Float32Array {
  const arr = new Float32Array(rows * cols);
  for (let i = 0; i < arr.length; i += 2) {
    // Box-Muller transform for normal distribution
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    arr[i] = mag * Math.cos(2.0 * Math.PI * v) * std;
    if (i + 1 < arr.length) {
      arr[i + 1] = mag * Math.sin(2.0 * Math.PI * v) * std;
    }
  }
  return arr;
}

/**
 * GELU activation function (Gaussian Error Linear Unit)
 */
export function gelu(x: number): number {
  return 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
}

/**
 * Layer Normalization
 */
export function layerNorm(
  x: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  out: Float32Array,
  dim: number,
  eps = 1e-5
): { mean: number; invStd: number } {
  let mean = 0;
  for (let i = 0; i < dim; i++) {
    mean += x[i];
  }
  mean /= dim;

  let variance = 0;
  for (let i = 0; i < dim; i++) {
    const diff = x[i] - mean;
    variance += diff * diff;
  }
  variance /= dim;
  const invStd = 1.0 / Math.sqrt(variance + eps);

  for (let i = 0; i < dim; i++) {
    const norm = (x[i] - mean) * invStd;
    out[i] = norm * gamma[i] + beta[i];
  }

  return { mean, invStd };
}

/**
 * Numerically stable Softmax with temperature
 */
export function softmax(logits: Float32Array, out: Float32Array, temp = 1.0): void {
  let maxVal = -Infinity;
  const len = logits.length;
  for (let i = 0; i < len; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }

  let sumExp = 0;
  for (let i = 0; i < len; i++) {
    const exp = Math.exp((logits[i] - maxVal) / Math.max(0.01, temp));
    out[i] = exp;
    sumExp += exp;
  }

  const invSum = 1.0 / Math.max(1e-12, sumExp);
  for (let i = 0; i < len; i++) {
    out[i] *= invSum;
  }
}

/**
 * Sample an index from a probability distribution
 */
export function sampleFromDistribution(
  probs: Float32Array,
  topK = 40,
  topP = 0.9,
  repetitionPenalty = 1.0,
  historyTokens: number[] = []
): { chosenId: number; candidates: Array<{ id: number; prob: number }> } {
  const len = probs.length;

  // Apply repetition penalty
  const adjusted = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    let p = probs[i];
    if (repetitionPenalty !== 1.0 && historyTokens.includes(i)) {
      p = p / repetitionPenalty;
    }
    adjusted[i] = p;
  }

  // Collect candidate pairs [id, prob]
  const pairs: Array<{ id: number; prob: number }> = [];
  for (let i = 0; i < len; i++) {
    if (adjusted[i] > 1e-7) {
      pairs.push({ id: i, prob: adjusted[i] });
    }
  }

  // Sort descending
  pairs.sort((a, b) => b.prob - a.prob);

  // Apply Top-K
  const kFiltered = topK > 0 ? pairs.slice(0, topK) : pairs;

  // Apply Top-P (Nucleus)
  let cumulative = 0;
  const pFiltered: Array<{ id: number; prob: number }> = [];
  for (const pair of kFiltered) {
    pFiltered.push(pair);
    cumulative += pair.prob;
    if (cumulative >= topP) break;
  }

  // Re-normalize probabilities
  let sumP = 0;
  for (const p of pFiltered) sumP += p.prob;
  if (sumP <= 0) {
    return { chosenId: pairs[0]?.id ?? 0, candidates: pairs.slice(0, 5) };
  }

  // Random sample
  const r = Math.random() * sumP;
  let running = 0;
  let chosenId = pFiltered[0].id;

  for (const p of pFiltered) {
    running += p.prob;
    if (running >= r) {
      chosenId = p.id;
      break;
    }
  }

  return {
    chosenId,
    candidates: pairs.slice(0, 5).map(c => ({ id: c.id, prob: c.prob / sumP }))
  };
}
