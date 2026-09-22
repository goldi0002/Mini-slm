/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

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
  public vocabSize: number;

  // Ids that exist purely as the single-character OOV fallback alphabet.
  // Real single-letter words ("a", "i") are NOT fallback tokens.
  private fallbackCharIds: Set<number> = new Set();

  constructor() {
    const seen = new Set<string>();
    let id = 0;

    for (const token of CONVERSATIONAL_VOCAB) {
      if (!seen.has(token)) {
        seen.add(token);
        this.tokenToId.set(token, id);
        this.idToToken.set(id, token);
        id++;
      }
    }

    // Add fallback alphabet
    for (let c = 97; c <= 122; c++) {
      const char = String.fromCharCode(c);
      if (!seen.has(char)) {
        seen.add(char);
        this.tokenToId.set(char, id);
        this.idToToken.set(id, char);
        this.fallbackCharIds.add(id);
        id++;
      }
    }
    for (let c = 65; c <= 90; c++) {
      const char = String.fromCharCode(c);
      // Never create an uppercase fallback for a letter that already exists as a
      // lowercase word ("A" -> 'a', "I" -> 'I'): the uppercase entry would
      // shadow the real word and then be suppressed during generation.
      if (this.tokenToId.has(char.toLowerCase())) continue;
      if (!seen.has(char)) {
        seen.add(char);
        this.tokenToId.set(char, id);
        this.idToToken.set(id, char);
        this.fallbackCharIds.add(id);
        id++;
      }
    }

    this.vocabSize = id;
  }

  /** True when the token is a single-character OOV fallback (never a real word). */
  public isFallbackCharToken(id: number): boolean {
    return this.fallbackCharIds.has(id);
  }

  /**
   * Add a word token (lowercased) if it is not already in the vocabulary.
   * Used to teach the tokenizer the vocabulary of built-in corpora and
   * datasets before models are constructed, so no word is ever reduced to
   * single-character fallback tokens that generation suppresses.
   */
  public addWord(raw: string): boolean {
    const word = raw.toLowerCase().trim();
    if (word.length < 2) return false;
    if (!/^[a-z0-9]+(?:'[a-z]+)?$/.test(word)) return false;
    if (this.tokenToId.has(word)) return false;
    const id = this.vocabSize;
    this.tokenToId.set(word, id);
    this.idToToken.set(id, word);
    this.vocabSize = id + 1;
    return true;
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
    const regex = /<user>|<assistant>|<bos>|<eos>|<pad>|\n|[a-zA-Z]+'[a-zA-Z]+|[a-zA-Z0-9]+|[^\s\w]|\s+/g;
    let match;

    while ((match = regex.exec(text)) !== null) {
      rawTokens.push(match[0]);
    }

    return rawTokens;
  }

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

      // Sub-character fallback
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
