/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Trigram language model with backoff, used as a "memory layer" that is
 * blended with the neural transformer logits at generation time.
 *
 * A ~50k-parameter toy transformer cannot learn fluent English from scratch
 * in a browser demo, so the studio pairs it with a statistical memory that
 * stores which tokens realistically follow which contexts. The neural model
 * contributes learned contextual signals; the memory layer guarantees fluent,
 * on-persona sentences. Fine-tuning reinforces both.
 */

export interface NgramSnapshot {
  uni: Map<number, number>; // token -> count
  bi: Map<number, Map<number, number>>; // prev1 -> (next -> count)
  tri: Map<number, Map<number, number>>; // (prev2 * vocabSize + prev1) -> (next -> count)
  total: number;
}

const EMPTY_MAP: Map<number, number> = new Map();

export class NgramLanguageModel {
  public vocabSize: number;
  private tables: NgramSnapshot;

  constructor(vocabSize: number) {
    this.vocabSize = vocabSize;
    this.tables = { uni: new Map(), bi: new Map(), tri: new Map(), total: 0 };
  }

  private key(prev2: number, prev1: number): number {
    return prev2 * this.vocabSize + prev1;
  }

  /** Accumulate token sequence statistics with a learning weight. */
  public observe(tokens: number[], weight = 1): void {
    const t = this.tables;
    for (let i = 1; i < tokens.length; i++) {
      const prev1 = tokens[i - 1];
      const cur = tokens[i];

      t.total += weight;
      t.uni.set(cur, (t.uni.get(cur) ?? 0) + weight);

      let biMap = t.bi.get(prev1);
      if (!biMap) {
        biMap = new Map();
        t.bi.set(prev1, biMap);
      }
      biMap.set(cur, (biMap.get(cur) ?? 0) + weight);

      if (i >= 2) {
        const k = this.key(tokens[i - 2], prev1);
        let triMap = t.tri.get(k);
        if (!triMap) {
          triMap = new Map();
          t.tri.set(k, triMap);
        }
        triMap.set(cur, (triMap.get(cur) ?? 0) + weight);
      }
    }
  }

  private mapSum(m: Map<number, number>): number {
    let s = 0;
    for (const v of m.values()) s += v;
    return s;
  }

  /**
   * Probability of `next` given the two preceding tokens, with interpolation:
   * weighted mix of trigram, bigram, unigram (levels fall back when unseen).
   */
  public prob(next: number, prev2: number, prev1: number, tables?: NgramSnapshot): number {
    const t = tables ?? this.tables;
    let p = 0;
    let weightSum = 0;

    // Unigram always available
    p += 0.15 * ((t.uni.get(next) ?? 0) / Math.max(1, t.total));
    weightSum += 0.15;

    const biMap = t.bi.get(prev1);
    if (biMap) {
      p += 0.35 * ((biMap.get(next) ?? 0) / Math.max(1, this.mapSum(biMap)));
      weightSum += 0.35;
    }

    const triMap = t.tri.get(this.key(prev2, prev1));
    if (triMap) {
      p += 0.5 * ((triMap.get(next) ?? 0) / Math.max(1, this.mapSum(triMap)));
      weightSum += 0.5;
    }

    return p / weightSum;
  }

  /**
   * Compute the full next-token distribution for a context, efficiently.
   * Fills `out` (length >= vocabSize) using stage-wise backoff: when a deeper
   * context (trigram, then bigram) was observed, it dominates the
   * distribution and only the remainder is reserved for the shallower level.
   * This keeps fluent multi-word chains intact instead of derailing into
   * generic unigram soup mid-sentence.
   */
  public distribution(prev2: number, prev1: number, out: Float32Array): void {
    const t = this.tables;
    out.fill(0);

    // Trigram level dominates when this exact context was observed
    const triMap = t.tri.get(this.key(prev2, prev1));
    if (triMap) {
      const invSum = 0.9 / Math.max(1, this.mapSum(triMap));
      for (const [tok, count] of triMap) {
        out[tok] += invSum * count;
      }
    }

    // Bigram level backs off
    const biMap = t.bi.get(prev1);
    if (biMap) {
      const invSum = (triMap ? 0.08 : 0.85) / Math.max(1, this.mapSum(biMap));
      for (const [tok, count] of biMap) {
        out[tok] += invSum * count;
      }
    }

    // Unigram level is the final fallback
    if (t.uni.size > 0) {
      const uniWeight = triMap ? 0.02 : biMap ? 0.15 : 1.0;
      const invTotal = uniWeight / Math.max(1, t.total);
      for (const [tok, count] of t.uni) {
        out[tok] += invTotal * count;
      }
    }

    // Renormalize so the distribution sums to 1
    let sum = 0;
    for (let v = 0; v < out.length; v++) sum += out[v];
    if (sum > 0) {
      const inv = 1 / sum;
      for (let v = 0; v < out.length; v++) out[v] *= inv;
    }
  }

  /** Deepest matching context length (3 = trigram hit, 2 = bigram, 1 = unigram, 0 = none). */
  public contextDepth(prev2: number, prev1: number, tables?: NgramSnapshot): number {
    const t = tables ?? this.tables;
    if (t.tri.has(this.key(prev2, prev1))) return 3;
    if (t.bi.has(prev1)) return 2;
    if (t.uni.size > 0) return 1;
    return 0;
  }

  /** Deep-copy the current tables (used for the base snapshot). */
  public snapshot(): NgramSnapshot {
    const uni = new Map(this.tables.uni);
    const bi = new Map<number, Map<number, number>>();
    for (const [k, m] of this.tables.bi) bi.set(k, new Map(m));
    const tri = new Map<number, Map<number, number>>();
    for (const [k, m] of this.tables.tri) tri.set(k, new Map(m));
    return { uni, bi, tri, total: this.tables.total };
  }

  /** Restore tables from a snapshot (mutates live tables in place). */
  public restore(snap: NgramSnapshot): void {
    this.tables.uni = new Map(snap.uni);
    this.tables.bi = new Map<number, Map<number, number>>();
    for (const [k, m] of snap.bi) this.tables.bi.set(k, new Map(m));
    this.tables.tri = new Map<number, Map<number, number>>();
    for (const [k, m] of snap.tri) this.tables.tri.set(k, new Map(m));
    this.tables.total = snap.total;
  }

  public get size(): number {
    return this.tables.total;
  }

  public static readonly emptyMap: Map<number, number> = EMPTY_MAP;
}
