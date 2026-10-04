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

/**
 * Sentence-boundary stop rule.
 *
 * A token counts as a sentence ender when the corpus has been observed ending
 * an utterance with it — "?", "!" and "." in practice, learned from the data
 * rather than listed here. `MASS` is added on top of a distribution normalized
 * to 1, so EOS ends up with `MASS / (1 + MASS)` of the mass at such a boundary.
 *
 * The point is completion: sampling alone let a reply ramble past the sentence
 * it had just finished and then be cut mid-clause by the token budget, so
 * answers never looked finished.
 */
const SENTENCE_END_STOP_MASS = 6;

export interface NgramSnapshot {
  uni: Map<number, number>; // token -> count
  bi: Map<number, Map<number, number>>; // prev1 -> (next -> count)
  tri: Map<number, Map<number, number>>; // (prev2 * vocabSize + prev1) -> (next -> count)
  total: number;
  // Tokens observed immediately before the EOS of an utterance, plus how many
  // utterance endings were seen in total. This is what lets generation stop at
  // a sentence boundary instead of running into the token limit.
  enders: Map<number, number>;
  endersTotal: number;
  // Dialogue-pair links: last word of a user message -> first word of the
  // matching reply (plus a global count of every reply opening).
  links: Map<number, Map<number, number>>;
  openers: Map<number, number>;
  // Dialogue cases: a user message's content tokens -> the reply it was
  // answered with, so a trained prompt reproduces its dataset answer.
  cases: Map<string, number[]>;
}

export class NgramLanguageModel {
  public vocabSize: number;
  private tables: NgramSnapshot;
  // End-of-utterance id, needed by the sentence-boundary stop rule. -1 means the
  // caller never told us how utterances end, so the rule stays off.
  private eosId: number;

  constructor(vocabSize: number, eosId = -1) {
    this.vocabSize = vocabSize;
    this.eosId = eosId;
    this.tables = {
      uni: new Map(),
      bi: new Map(),
      tri: new Map(),
      total: 0,
      links: new Map(),
      openers: new Map(),
      cases: new Map(),
      enders: new Map(),
      endersTotal: 0,
    };
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

    // Utterance endings: the token right before a trailing EOS. A truncated
    // sequence (no EOS last) records nothing, which is right — it did not end.
    if (this.eosId >= 0 && tokens.length >= 2 && tokens[tokens.length - 1] === this.eosId) {
      const ender = tokens[tokens.length - 2];
      if (ender !== this.eosId) {
        t.enders.set(ender, (t.enders.get(ender) ?? 0) + weight);
        t.endersTotal += weight;
      }
    }
  }

  private mapSum(m: Map<number, number>): number {
    let s = 0;
    for (const v of m.values()) s += v;
    return s;
  }

  /**
   * Stable key for a user message in the dialogue-case memory.
   *
   * A trigram table pools every conversation it has ever seen, so a context like
   * ("your", " ") is genuinely ambiguous between "your friendly assistant" and
   * "your mind". Remembering the answer *per user message* is what lets a
   * fine-tuned model answer its dataset prompts with the dataset answers instead
   * of wandering through the pooled average of every sentence.
   */
  public static caseKey(wordTokens: number[]): string {
    return wordTokens.join(',');
  }

  /** Remember how a user message was answered (keyed by its content words). */
  public rememberCase(wordTokens: number[], replyTokens: number[]): void {
    if (wordTokens.length === 0 || replyTokens.length === 0) return;
    this.tables.cases.set(NgramLanguageModel.caseKey(wordTokens), [...replyTokens]);
  }

  /**
   * The reply learned for this user message: an exact content-word match first,
   * then the closest stored question above `minScore` (word-overlap / union), so
   * rephrasings still find the answer a dataset taught. Returns null when the
   * message has nothing in common with anything the model has been taught.
   */
  public findCase(wordTokens: number[], minScore = 0.65, tables?: NgramSnapshot): number[] | null {
    if (wordTokens.length === 0) return null;
    const cases = (tables ?? this.tables).cases;

    const exact = cases.get(NgramLanguageModel.caseKey(wordTokens));
    if (exact) return exact;

    const want = new Set(wordTokens);
    let best: number[] | null = null;
    let bestScore = 0;
    for (const [key, reply] of cases) {
      if (key.length === 0) continue;
      const have = new Set(key.split(',').map(Number));
      let shared = 0;
      for (const w of want) if (have.has(w)) shared++;
      const score = shared / (want.size + have.size - shared);
      if (score > bestScore) {
        bestScore = score;
        best = reply;
      }
    }

    return bestScore >= minScore ? best : null;
  }

  /**
   * Learn the link between a user message and how the assistant answered it:
   * the last word of the user turn and the first word of the reply.
   *
   * A flat n-gram table cannot express "this word ends the question, the answer
   * starts like this" — the user turn and the reply are adjacent in the stream,
   * so their trigrams genuinely blur together. Recording the pairing separately
   * lets generation open a reply relevant to what was actually asked.
   */
  public observeReplyLink(userWord: number, firstReplyWord: number, weight = 1): void {
    const t = this.tables;
    let m = t.links.get(userWord);
    if (!m) {
      m = new Map();
      t.links.set(userWord, m);
    }
    m.set(firstReplyWord, (m.get(firstReplyWord) ?? 0) + weight);
    t.openers.set(firstReplyWord, (t.openers.get(firstReplyWord) ?? 0) + weight);
  }

  /**
   * Opening distribution learned for this exact user word. Returns false when
   * the word never ended a user message we have seen.
   */
  public linkDistribution(userWord: number, out: Float32Array, tables?: NgramSnapshot): boolean {
    const m = (tables ?? this.tables).links.get(userWord);
    if (!m || m.size === 0) return false;
    this.fillFromCounts(m, out);
    return true;
  }

  /** Opening distribution across every reply we have seen (generic fallback). */
  public openerDistribution(out: Float32Array, tables?: NgramSnapshot): boolean {
    const m = (tables ?? this.tables).openers;
    if (m.size === 0) return false;
    this.fillFromCounts(m, out);
    return true;
  }

  private fillFromCounts(counts: Map<number, number>, out: Float32Array): void {
    out.fill(0);
    let sum = 0;
    for (const [tok, count] of counts) {
      if (tok >= 0 && tok < out.length) {
        out[tok] = count;
        sum += count;
      }
    }
    if (sum > 0) {
      const inv = 1 / sum;
      for (let v = 0; v < out.length; v++) out[v] *= inv;
    }
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

    // Sentence-boundary stop rule. When the previous token is one the corpus
    // uses to end utterances — ".", "!", "?" after even a small amount of
    // training — the reply is allowed to finish here. Without this it rambles
    // past the sentence it just completed and is stopped by the token budget
    // mid-clause, which reads as an unfinished answer.
    if (this.eosId >= 0 && this.eosId < out.length && (t.enders.get(prev1) ?? 0) > 0) {
      out[this.eosId] += SENTENCE_END_STOP_MASS;
    }

    // Renormalize so the distribution sums to 1
    let sum = 0;
    for (let v = 0; v < out.length; v++) sum += out[v];
    if (sum > 0) {
      const inv = 1 / sum;
      for (let v = 0; v < out.length; v++) out[v] *= inv;
    }
  }

  /**
   * Follow a vocabulary that grew underneath the tables.
   *
   * Trigram contexts are keyed by `prev2 * vocabSize + prev1`, so growing the
   * vocabulary silently invalidates every stored key. Rewriting the keys is what
   * lets a trained memory layer survive the addition of new words instead of
   * being discarded with the model (ISS-21).
   */
  public remapVocabSize(newVocabSize: number): void {
    const oldVocabSize = this.vocabSize;
    if (newVocabSize <= oldVocabSize) return;
    const remapped = new Map<number, Map<number, number>>();
    for (const [key, counts] of this.tables.tri) {
      const prev2 = Math.floor(key / oldVocabSize);
      const prev1 = key - prev2 * oldVocabSize;
      remapped.set(prev2 * newVocabSize + prev1, counts);
    }
    this.tables.tri = remapped;
    this.vocabSize = newVocabSize;
  }

  private static copyNested(src: Map<number, Map<number, number>>): Map<number, Map<number, number>> {
    const dst = new Map<number, Map<number, number>>();
    for (const [k, m] of src) dst.set(k, new Map(m));
    return dst;
  }

  /** Deep-copy the current tables (used for the base snapshot). */
  public snapshot(): NgramSnapshot {
    const cases = new Map<string, number[]>();
    for (const [k, seq] of this.tables.cases) cases.set(k, [...seq]);
    return {
      uni: new Map(this.tables.uni),
      bi: NgramLanguageModel.copyNested(this.tables.bi),
      tri: NgramLanguageModel.copyNested(this.tables.tri),
      total: this.tables.total,
      links: NgramLanguageModel.copyNested(this.tables.links),
      openers: new Map(this.tables.openers),
      cases,
      enders: new Map(this.tables.enders),
      endersTotal: this.tables.endersTotal,
    };
  }

  /** Restore tables from a snapshot (mutates live tables in place). */
  public restore(snap: NgramSnapshot): void {
    this.tables.uni = new Map(snap.uni);
    this.tables.bi = NgramLanguageModel.copyNested(snap.bi);
    this.tables.tri = NgramLanguageModel.copyNested(snap.tri);
    this.tables.total = snap.total;
    this.tables.links = NgramLanguageModel.copyNested(snap.links);
    this.tables.openers = new Map(snap.openers);
    const cases = new Map<string, number[]>();
    for (const [k, seq] of snap.cases) cases.set(k, [...seq]);
    this.tables.cases = cases;
    this.tables.enders = new Map(snap.enders);
    this.tables.endersTotal = snap.endersTotal;
  }

  public get size(): number {
    return this.tables.total;
  }
}
