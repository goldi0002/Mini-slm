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
 * Derivative of the tanh-approximation GELU used above.
 */
export function geluDerivative(x: number): number {
  const s = Math.sqrt(2.0 / Math.PI);
  const inner = x + 0.044715 * x * x * x;
  const tanhVal = Math.tanh(s * inner);
  const sech2 = 1.0 - tanhVal * tanhVal;
  // d/dx [0.5x(1 + tanh(s(x + a x^3)))] where a = 0.044715
  return 0.5 * (1.0 + tanhVal) + 0.5 * x * sech2 * s * (1.0 + 3.0 * 0.044715 * x * x);
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
 * Backward pass of layerNorm: given dy (gradient w.r.t. the normalized output),
 * produce dx (gradient w.r.t. the input). `x` and `dy` are read from the given
 * offsets so callers can pass shared row-major buffers. `dy` and `dx` must not
 * alias.
 */
export function layerNormBackward(
  x: Float32Array,
  xOffset: number,
  gamma: Float32Array,
  dy: Float32Array,
  dyOffset: number,
  dx: Float32Array,
  dxOffset: number,
  dim: number,
  eps = 1e-5
): void {
  let mean = 0;
  for (let i = 0; i < dim; i++) mean += x[xOffset + i];
  mean /= dim;

  let variance = 0;
  for (let i = 0; i < dim; i++) {
    const diff = x[xOffset + i] - mean;
    variance += diff * diff;
  }
  variance /= dim;
  const invStd = 1.0 / Math.sqrt(variance + eps);

  // sumG  = sum_k dy_k * gamma_k
  // sumGX = sum_k dy_k * gamma_k * xhat_k
  let sumG = 0;
  let sumGX = 0;
  for (let i = 0; i < dim; i++) {
    const xhat = (x[xOffset + i] - mean) * invStd;
    const g = dy[dyOffset + i] * gamma[i];
    sumG += g;
    sumGX += g * xhat;
  }

  const invDim = 1.0 / dim;
  for (let i = 0; i < dim; i++) {
    const xhat = (x[xOffset + i] - mean) * invStd;
    const g = dy[dyOffset + i] * gamma[i];
    dx[dxOffset + i] = invStd * (g - sumG * invDim - xhat * sumGX * invDim);
  }
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

export interface SamplingResult {
  chosenId: number;
  /**
   * The most likely candidates with their probability in distribution space.
   * These are the model's own probabilities (after the repetition penalty),
   * normalized by the full distribution mass — NOT by the top-k/top-p subset.
   * Re-normalizing over the filtered subset inflates them whenever the nucleus
   * cuts the tail (top candidates summing past 100%) and contradicts the
   * uninflated probability reported for the sampled token.
   */
  candidates: Array<{ id: number; prob: number }>;
  /** Total mass of the unfiltered distribution the candidates were drawn from. */
  totalMass: number;
}

/**
 * Sample an index from a probability distribution.
 *
 * Top-k and top-p decide where sampling may draw from; the returned candidate
 * list always reports true distribution-space probabilities, so the inspector
 * UI never shows an inflated nucleus.
 */
export function sampleFromDistribution(
  probs: Float32Array,
  topK = 40,
  topP = 0.9,
  repetitionPenalty = 1.0,
  historyTokens: number[] = []
): SamplingResult {
  const len = probs.length;

  // Apply repetition penalty
  const adjusted = new Float32Array(len);
  const historySet = repetitionPenalty !== 1.0 && historyTokens.length > 0 ? new Set(historyTokens) : null;
  for (let i = 0; i < len; i++) {
    let p = probs[i];
    if (historySet !== null && historySet.has(i)) {
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

  // Total mass of the whole (repetition-adjusted) distribution. Every candidate
  // probability handed back to the UI is normalized by this, never by the
  // filtered subset.
  let totalMass = 0;
  for (const p of pairs) totalMass += p.prob;
  const toCandidates = (list: Array<{ id: number; prob: number }>) =>
    totalMass > 0 ? list.map((c) => ({ id: c.id, prob: c.prob / totalMass })) : list;

  // Apply Top-K
  const kFiltered = topK > 0 ? pairs.slice(0, topK) : pairs;

  // Apply Top-P (Nucleus). The filter decides *where* sampling may draw from;
  // it must not inflate the probabilities reported for those candidates, or the
  // inspected nucleus would contradict both the sampled token's own likelihood
  // and the tail mass the modal derives from it.
  let cumulative = 0;
  const pFiltered: Array<{ id: number; prob: number }> = [];
  for (const pair of kFiltered) {
    pFiltered.push(pair);
    cumulative += pair.prob;
    if (cumulative >= topP) break;
  }

  // Only the sampling draw itself re-normalizes over the nucleus.
  let sumP = 0;
  for (const p of pFiltered) sumP += p.prob;
  if (sumP <= 0) {
    return { chosenId: pairs[0]?.id ?? 0, candidates: toCandidates(pairs.slice(0, 5)), totalMass };
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
    candidates: toCandidates(pairs.slice(0, 5)),
    totalMass
  };
}
