/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Hybrid word-level + BPE tokenizer.
 *
 * The vocabulary is built in four deterministic stages:
 *   1. The conversational word vocabulary (below) — ids are stable forever.
 *   2. The single-character fallback alphabet.
 *   3. Every word of the seed corpus (`BASE_CORPUS`) as a whole-word token, so
 *      the model's fluent baseline is always expressible without subwords.
 *   4. BPE merge symbols learned from the seed corpus + `BPE_SEED_WORDS`.
 *
 * Stages 1–2 keep the same ids this tokenizer has always produced, which is
 * what keeps n-gram memory links, dialogue-case keys, and saved weight
 * snapshots meaningful across the upgrade. Only stages 3–4 append new ids.
 *
 * Encoding: a word found in the vocabulary is a single token (exactly as
 * before, so datasets keep their exact-recall behavior). An unknown word is no
 * longer spelled out character by character — it decomposes into learned
 * subword pieces ("unbelievable" → "un" + "believ" + "able"), each with its own
 * embedding. That is the property that lets the model read and compose words it
 * has never seen instead of choking on them.
 */

import { BASE_CORPUS, BPE_SEED_WORDS, ENGLISH_LEARNING_CORPUS } from './corpus';

/**
 * Every text the tokenizer learns its vocabulary and merges from: the dialogue
 * seed (for fluent conversation) and the plain-English learning corpus (so the
 * network can read and produce real sentences, not just dialogue fragments).
 */
const VOCABULARY_SOURCES: string[] = [...BASE_CORPUS, ...ENGLISH_LEARNING_CORPUS];

// Special control tokens
export const SPECIAL_TOKENS = {
  PAD: '<pad>',
  UNK: '<unk>',
  BOS: '<bos>',
  EOS: '<eos>',
  USER: '<user>',
  ASSISTANT: '<assistant>',
  NEWLINE: '\n',
} as const;

export const PAD_ID = 0;
export const UNK_ID = 1;
export const BOS_ID = 2;
export const EOS_ID = 3;
export const USER_ID = 4;
export const ASSISTANT_ID = 5;
export const NEWLINE_ID = 6;

/** Upper bound on learned BPE merge symbols; keeps the vocabulary (and therefore the parameter count) bounded. */
const MAX_BPE_MERGES = 256;

/** Separator inside merge-rank keys; cannot appear inside a word piece. */
const PAIR_SEP = '\u0000';

// Pure conversational chat vocabulary: everyday dialogue, emotions, questions, and polite conversation
const CONVERSATIONAL_VOCAB = [
  // Special tokens
  SPECIAL_TOKENS.PAD,
  SPECIAL_TOKENS.UNK,
  SPECIAL_TOKENS.BOS,
  SPECIAL_TOKENS.EOS,
  SPECIAL_TOKENS.USER,
  SPECIAL_TOKENS.ASSISTANT,
  SPECIAL_TOKENS.NEWLINE,

  // Punctuation & spacing
  ' ', '!', '?', '.', ',', ':', ';', '-', '_', '(', ')', '"', "'", '`', '/',

  // Numbers
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

  // Common conversational contractions
  "i'm", "you're", "it's", "don't", "can't", "that's", "what's", "we're", "there's", "i've", "i'll", "didn't", "won't",

  // Greetings & daily openers
  'hello', 'hi', 'hey', 'greetings', 'welcome', 'morning', 'afternoon', 'evening', 'night', 'goodbye', 'bye',

  // Pronouns & Articles
  // Capital "I" is a real word token: it opens most assistant replies, and if it
  // fell through to the single-character OOV alphabet it would be suppressed
  // during generation (making "I am ..." impossible to produce).
  'I', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
  'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours',
  'a', 'an', 'the', 'this', 'that', 'these', 'those',

  // Conversational verbs & auxiliary
  'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'have', 'has', 'had', 'do', 'does', 'did', 'doing',
  'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
  'help', 'assist', 'know', 'think', 'feel', 'understand', 'explain', 'share',
  'ask', 'tell', 'listen', 'hear', 'speak', 'talk', 'chat', 'say', 'said',
  'see', 'look', 'watch', 'find', 'make', 'create', 'give', 'take', 'come', 'go',
  'learn', 'read', 'write', 'grow', 'remember', 'forget', 'enjoy', 'like', 'love',
  'need', 'want', 'wish', 'hope', 'try', 'start', 'stop', 'keep', 'stay', 'breathe',
  'reflect', 'relax', 'focus', 'wonder', 'appreciate', 'inspire', 'connect',

  // Question words
  'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose',

  // Politeness, affirmation, and reassurance
  'please', 'thank', 'thanks', 'certainly', 'absolutely', 'definitely',
  'sure', 'yes', 'no', 'of', 'course', 'glad', 'happy', 'delighted', 'pleasure',
  'sorry', 'apologies', 'alright', 'fine', 'okay', 'great', 'good', 'wonderful',
  'excellent', 'fantastic', 'peaceful', 'calm', 'kind', 'friendly', 'gentle',
  'thoughtful', 'meaningful', 'true', 'fascinating', 'inspiring',

  // Daily life & concept nouns
  'ai', 'assistant', 'companion', 'model', 'friend', 'person', 'people', 'human',
  'day', 'today', 'tonight', 'tomorrow', 'time', 'moment', 'life',
  'world', 'mind', 'heart', 'thought', 'idea', 'question', 'answer', 'story',
  'advice', 'habit', 'routine', 'task', 'goal', 'plan', 'work', 'break', 'rest',
  'nature', 'sun', 'sky', 'water', 'lake', 'earth', 'light', 'tree', 'flower',
  'book', 'music', 'sound', 'step', 'breath', 'journey', 'experience', 'wisdom',
  'stress', 'peace', 'joy', 'smile', 'conversation', 'dialogue',

  // Connectors, prepositions, & qualifiers
  'and', 'or', 'but', 'so', 'because', 'although', 'while', 'since', 'if', 'then',
  'as', 'at', 'by', 'for', 'from', 'in', 'into', 'off', 'on', 'onto', 'out',
  'over', 'to', 'up', 'with', 'about', 'against', 'between', 'through', 'during',
  'before', 'after', 'above', 'below', 'under',
  'all', 'some', 'any', 'every', 'each', 'both', 'few', 'more', 'most', 'other',
  'much', 'many', 'very', 'quite', 'really', 'too', 'just', 'only', 'even',
  'well', 'actually', 'honestly', 'perhaps', 'maybe', 'together', 'always', 'sometimes',
  'never', 'often', 'now', 'here', 'there', 'again', 'ever'
];

export class Tokenizer {
  private tokenToId: Map<string, number> = new Map();
  private idToToken: Map<number, string> = new Map();
  /** Next unused token id; keeps the two id maps in lockstep with `vocabSize`. */
  private nextId = 0;
  public vocabSize: number;

  // Ids that exist purely as the single-character OOV fallback alphabet.
  // Real single-letter words ("a", "i") are NOT fallback tokens.
  private fallbackCharIds: Set<number> = new Set();

  // BPE state: ordered merge rules (rank -> "a" + "b" -> "ab") and a memo
  // table so repeated encodes of the same word do not re-run the merge loop.
  private mergeRanks: Map<string, number> = new Map();
  private mergeRules: Array<[string, string]> = [];
  private wordCache: Map<string, number[]> = new Map();

  /**
   * Register a token and return its id, reusing the existing id when the
   * token is already known. The single place ids are ever handed out, so
   * `nextId`, `tokenToId` and `idToToken` can never drift apart.
   */
  private addToken(token: string): number {
    const existing = this.tokenToId.get(token);
    if (existing !== undefined) return existing;
    const id = this.nextId++;
    this.tokenToId.set(token, id);
    this.idToToken.set(id, token);
    return id;
  }

  constructor() {
    for (const token of CONVERSATIONAL_VOCAB) {
      this.addToken(token);
    }

    // Add fallback alphabet
    for (let c = 97; c <= 122; c++) {
      const char = String.fromCharCode(c);
      if (!this.tokenToId.has(char)) {
        this.fallbackCharIds.add(this.addToken(char));
      }
    }
    for (let c = 65; c <= 90; c++) {
      const char = String.fromCharCode(c);
      // Never create an uppercase fallback for a letter that already exists as a
      // lowercase word ("A" -> 'a', "I" -> 'I'): the uppercase entry would
      // shadow the real word and then be suppressed during generation.
      if (this.tokenToId.has(char.toLowerCase())) continue;
      if (!this.tokenToId.has(char)) {
        this.fallbackCharIds.add(this.addToken(char));
      }
    }

    // Teach the seed corpus as whole words BEFORE learning merges, so corpus
    // fluency never depends on subword composition and merge results that
    // equal a corpus word reuse the existing id instead of duplicating it.
    const corpusWordRegex = /[a-zA-Z0-9]+(?:'[a-zA-Z]+)?/g;
    for (const text of VOCABULARY_SOURCES) {
      for (const match of text.matchAll(corpusWordRegex)) {
        const word = match[0].toLowerCase();
        if (/^[a-z0-9]+(?:'[a-z]+)?$/.test(word)) this.addToken(word);
      }
    }

    // Learn BPE merges from every vocabulary source plus the affix-rich seed
    // words, so an unseen word composes from pieces this corpus has taught.
    this.learnBpeMerges([...VOCABULARY_SOURCES, ...BPE_SEED_WORDS]);

    this.vocabSize = this.nextId;
  }

  /**
   * Learn byte-pair-encoding merges: repeatedly find the most frequent adjacent
   * symbol pair across the (lowercased) corpus words and freeze it as a merge
   * rule. Deterministic — ties break on the lexicographically smallest pair —
   * so the vocabulary is identical on every load.
   */
  private learnBpeMerges(texts: string[]): void {
    const wordRegex = /[a-zA-Z0-9]+(?:'[a-zA-Z]+)?/g;
    const wordCounts = new Map<string, number>();
    for (const text of texts) {
      for (const match of text.matchAll(wordRegex)) {
        const word = match[0].toLowerCase();
        wordCounts.set(word, (wordCounts.get(word) ?? 0) + 1);
      }
    }

    const words: Array<{ syms: string[]; count: number }> = [];
    for (const [word, count] of wordCounts) {
      words.push({ syms: [...word], count });
    }

    for (let step = 0; step < MAX_BPE_MERGES; step++) {
      // Count adjacent symbol pairs, weighted by word frequency.
      const pairCounts = new Map<string, number>();
      for (const { syms, count } of words) {
        for (let i = 0; i + 1 < syms.length; i++) {
          const key = syms[i] + PAIR_SEP + syms[i + 1];
          pairCounts.set(key, (pairCounts.get(key) ?? 0) + count);
        }
      }

      let bestKey = '';
      let bestCount = 1; // pairs seen once are noise; require frequency >= 2
      for (const [key, count] of pairCounts) {
        if (count > bestCount || (count === bestCount && bestKey !== '' && key < bestKey)) {
          bestCount = count;
          bestKey = key;
        }
      }
      if (bestKey === '') break;

      const sepIdx = bestKey.indexOf(PAIR_SEP);
      const a = bestKey.slice(0, sepIdx);
      const b = bestKey.slice(sepIdx + 1);
      const merged = a + b;

      // Apply the merge everywhere.
      for (const w of words) {
        const syms = w.syms;
        if (syms.length < 2) continue;
        const next: string[] = [];
        for (let i = 0; i < syms.length; ) {
          if (i + 1 < syms.length && syms[i] === a && syms[i + 1] === b) {
            next.push(merged);
            i += 2;
          } else {
            next.push(syms[i]);
            i++;
          }
        }
        w.syms = next;
      }

      // Register the merge rule and, when the result is a new symbol, a token.
      this.mergeRanks.set(bestKey, this.mergeRules.length);
      this.mergeRules.push([a, b]);
      this.addToken(merged);
    }
  }

  /**
   * Decompose a word into BPE pieces using the learned merges: repeatedly apply
   * the lowest-ranked (earliest-learned, most frequent) merge present in the
   * symbol sequence until no known pair remains. Deterministic and vocabulary-
   * independent of casing (callers pass a lowercased word).
   */
  private bpeSplit(word: string): string[] {
    let syms = [...word];
    while (syms.length > 1) {
      let bestRank = Infinity;
      for (let i = 0; i + 1 < syms.length; i++) {
        const rank = this.mergeRanks.get(syms[i] + PAIR_SEP + syms[i + 1]);
        if (rank !== undefined && rank < bestRank) bestRank = rank;
      }
      if (bestRank === Infinity) break;

      // One pass applies every occurrence of that single merge.
      const [a, b] = this.mergeRules[bestRank];
      const next: string[] = [];
      for (let i = 0; i < syms.length; ) {
        if (i + 1 < syms.length && syms[i] === a && syms[i + 1] === b) {
          next.push(a + b);
          i += 2;
        } else {
          next.push(syms[i]);
          i++;
        }
      }
      syms = next;
    }
    return syms;
  }

  /**
   * Token ids for a single word: the whole-word token when known, otherwise its
   * BPE decomposition. Memoized; the cache is invalidated by addWord() because
   * a newly learned whole word supersedes a cached subword split.
   */
  private wordToIds(word: string): number[] {
    const cached = this.wordCache.get(word);
    if (cached) return cached;

    let ids: number[];
    const whole = this.tokenToId.get(word);
    if (whole !== undefined) {
      ids = [whole];
    } else {
      ids = [];
      for (const piece of this.bpeSplit(word)) {
        const pieceId = this.tokenToId.get(piece);
        if (pieceId !== undefined) {
          ids.push(pieceId);
        } else {
          // Unreachable while every merge symbol and every letter/digit is
          // registered, but never emit a bogus id if that ever changes.
          for (const ch of piece) ids.push(this.tokenToId.get(ch) ?? UNK_ID);
        }
      }
    }

    if (this.wordCache.size > 8192) this.wordCache.clear();
    this.wordCache.set(word, ids);
    return ids;
  }

  /** Number of learned BPE merge rules (each contributes at most one new token). */
  public get bpeMergeCount(): number {
    return this.mergeRanks.size;
  }

  /** True when the token is a single-character OOV fallback (never a real word). */
  public isFallbackCharToken(id: number): boolean {
    return this.fallbackCharIds.has(id);
  }

  /**
   * Add a word token (lowercased) if it is not already in the vocabulary.
   * Used to teach the tokenizer the vocabulary of built-in corpora and
   * datasets before models are constructed, so familiar words get a single
   * whole-word token even though BPE could spell them out.
   */
  public addWord(raw: string): boolean {
    const word = raw.toLowerCase().trim();
    if (word.length < 2) return false;
    if (!/^[a-z0-9]+(?:'[a-z]+)?$/.test(word)) return false;
    if (this.tokenToId.has(word)) return false;
    const id = this.addToken(word);
    this.vocabSize = this.nextId;
    // A cached BPE split of this word is now stale: the whole word wins.
    this.wordCache.delete(word);
    return id >= 0;
  }

  /** Teach the tokenizer every word appearing in the given texts. Returns added count. */
  public learnWords(texts: string[]): number {
    let added = 0;
    const wordRegex = /[a-zA-Z0-9]+(?:'[a-zA-Z]+)?/g;
    for (const text of texts) {
      for (const match of text.matchAll(wordRegex)) {
        if (this.addWord(match[0])) added++;
      }
    }
    return added;
  }

  /**
   * Split string into words, punctuation, whitespace, and special tags
   */
  private splitIntoTokens(text: string): string[] {
    const rawTokens: string[] = [];
    const regex = /<user>|<assistant>|<bos>|<eos>|<pad>|<unk>|\n|[a-zA-Z]+'[a-zA-Z]+|[a-zA-Z0-9]+|[^\s\w]|\s+/g;
    let match;

    while ((match = regex.exec(text)) !== null) {
      rawTokens.push(match[0]);
    }

    return rawTokens;
  }

  /** Characters that make up a word-like piece routable through the BPE path. */
  private static readonly WORD_CHARS = /^[a-zA-Z0-9']+$/;

  /**
   * Encode a text string into token IDs
   */
  public encode(text: string, addBos = true, addEos = false): number[] {
    const tokens: number[] = [];
    if (addBos) tokens.push(BOS_ID);

    const pieces = this.splitIntoTokens(text);

    for (const piece of pieces) {
      if (this.tokenToId.has(piece)) {
        tokens.push(this.tokenToId.get(piece)!);
        continue;
      }

      const lower = piece.toLowerCase();
      if (this.tokenToId.has(lower)) {
        tokens.push(this.tokenToId.get(lower)!);
        continue;
      }

      // If whitespace, break into single spaces
      if (/^\s+$/.test(piece)) {
        for (let i = 0; i < piece.length; i++) {
          const char = piece[i];
          if (char === '\n') {
            tokens.push(NEWLINE_ID);
          } else {
            tokens.push(this.tokenToId.get(' ') ?? UNK_ID);
          }
        }
        continue;
      }

      // Word-like pieces decompose through the learned BPE merges instead of
      // being spelled out character by character. Every resulting piece carries
      // its own embedding, so unseen words are composed from learned parts.
      if (Tokenizer.WORD_CHARS.test(piece)) {
        for (const id of this.wordToIds(lower)) tokens.push(id);
        continue;
      }

      // Sub-character fallback (punctuation and other symbols)
      for (const char of piece) {
        if (this.tokenToId.has(char)) {
          tokens.push(this.tokenToId.get(char)!);
        } else if (this.tokenToId.has(char.toLowerCase())) {
          tokens.push(this.tokenToId.get(char.toLowerCase())!);
        } else {
          tokens.push(UNK_ID);
        }
      }
    }

    if (addEos) tokens.push(EOS_ID);
    return tokens;
  }

  /**
   * Decode token IDs back to string
   */
  public decode(tokenIds: number[], skipSpecial = true): string {
    let result = '';
    for (const id of tokenIds) {
      if (skipSpecial) {
        if (id === PAD_ID || id === BOS_ID || id === EOS_ID) continue;
        if (id === USER_ID) {
          result += 'User: ';
          continue;
        }
        if (id === ASSISTANT_ID) {
          result += 'Assistant: ';
          continue;
        }
      }

      const token = this.idToToken.get(id);
      if (token !== undefined) {
        result += token;
      } else {
        result += ' ';
      }
    }

    // Clean up formatting
    return result
      .replace(/ {2,}/g, ' ')
      .replace(/\s+([!?,.:;])/g, '$1')
      .replace(/([!?,.:;])([a-zA-Z])/g, '$1 $2')
      .trim();
  }

  public getTokenString(id: number): string {
    return this.idToToken.get(id) ?? `<id:${id}>`;
  }

  /** Look up the id of an exact token string, or undefined when unknown. */
  public idOf(token: string): number | undefined {
    return this.tokenToId.get(token);
  }

  public formatConversationPrompt(userMessage: string, history: Array<{ role: string; content: string }> = []): string {
    let prompt = '';
    for (const turn of history) {
      if (turn.role === 'user') {
        prompt += `${SPECIAL_TOKENS.USER} ${turn.content} ${SPECIAL_TOKENS.NEWLINE}`;
      } else if (turn.role === 'assistant') {
        prompt += `${SPECIAL_TOKENS.ASSISTANT} ${turn.content} ${SPECIAL_TOKENS.NEWLINE}`;
      }
    }
    prompt += `${SPECIAL_TOKENS.USER} ${userMessage} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} `;
    return prompt;
  }
}

export const defaultTokenizer = new Tokenizer();
