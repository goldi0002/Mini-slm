/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Dependency-free seed corpus for the tokenizer and the base model.
 *
 * This module must not import anything: `src/slm/tokenizer.ts` learns its BPE
 * merges from these texts at construction time, and the transformer imports the
 * tokenizer, so importing the transformer here would create a circular module
 * dependency (tokenizer -> transformer -> tokenizer).
 */

/**
 * Baseline conversational corpus for the statistical memory layer, written
 * against the tokenizer vocabulary so the base model is already fluent.
 */
export const BASE_CORPUS: string[] = [
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
  'goodbye for now ! I am here whenever you need a friend to talk to .',
];

/**
 * Additional English words used only to seed the BPE merge table.
 *
 * `BASE_CORPUS` alone teaches merges about the words it contains; this list
 * widens coverage to frequent English affixes and stems ("re", "ing", "ed",
 * "tion", "less", "ful", ...) so an out-of-vocabulary word like
 * "unbelievable" decomposes into "un", "believ", "able" rather than being
 * spelled out character by character.
 *
 * Higher frequency in the corpus means earlier merges, which is why each word
 * is repeated in proportion to how common it is in everyday English.
 */
export const BPE_SEED_WORDS: string[] = [
  // Function words seen in every English sentence (high frequency)
  'the', 'the', 'the', 'the', 'and', 'and', 'and', 'a', 'a', 'a', 'to', 'to', 'to',
  'of', 'of', 'in', 'in', 'is', 'is', 'it', 'it', 'you', 'you', 'that', 'that',
  'was', 'for', 'are', 'with', 'as', 'his', 'they', 'at', 'be', 'this', 'have',
  'from', 'or', 'had', 'but', 'not', 'what', 'all', 'were', 'we', 'when',
  // Common nouns & verbs (frequent, so their merges come early)
  'thing', 'thing', 'something', 'anything', 'everything', 'nothing',
  'people', 'person', 'time', 'day', 'way', 'world', 'life', 'work', 'word',
  'make', 'made', 'take', 'took', 'give', 'gave', 'come', 'came', 'want',
  'know', 'knew', 'think', 'thought', 'find', 'found', 'tell', 'told',
  'feel', 'felt', 'become', 'became', 'leave', 'left', 'keep', 'kept',
  'help', 'helped', 'talk', 'talked', 'turn', 'turned', 'start', 'started',
  'show', 'showed', 'hear', 'heard', 'play', 'played', 'move', 'moved',
  'like', 'liked', 'live', 'lived', 'believe', 'believed', 'hold', 'held',
  'bring', 'brought', 'happen', 'happened', 'write', 'written', 'provide',
  'sit', 'stand', 'lose', 'lost', 'pay', 'meet', 'include', 'continue',
  'learn', 'learning', 'learned', 'understand', 'understood',
  // Adjectives & adverbs
  'good', 'new', 'first', 'last', 'long', 'great', 'little', 'other',
  'old', 'right', 'big', 'high', 'small', 'large', 'next', 'early',
  'young', 'important', 'few', 'public', 'bad', 'same', 'able',
  // Affix-rich words: these drive the suffix/prefix merges that matter
  'care', 'careful', 'carefully', 'careless',
  'use', 'useful', 'useless', 'used', 'using',
  'help', 'helpful', 'helpless', 'helping',
  'hope', 'hopeful', 'hopeless', 'hoping',
  'think', 'thinking', 'thinker',
  'read', 'reader', 'reading',
  'kind', 'kindness', 'kindly',
  'quiet', 'quietly', 'quietness',
  'sad', 'sadly', 'sadness',
  'happy', 'happily', 'happiness', 'unhappy',
  'beauty', 'beautiful', 'beautifully',
  'wonder', 'wonderful', 'wondering',
  'create', 'creative', 'creation', 'creating',
  'relate', 'relation', 'relations',
  'educate', 'education',
  'communicate', 'communication',
  'inform', 'information',
  'converse', 'conversation', 'conversations',
  'connect', 'connection', 'connecting',
  'possible', 'impossible', 'possibility',
  'believe', 'believable', 'unbelievable', 'believing',
  'comfort', 'comfortable', 'uncomfortable', 'comforting',
  'understand', 'understanding', 'misunderstand', 'understands',
  // Numbers as words
  'one', 'two', 'three', 'four', 'five', 'ten', 'hundred', 'thousand',
];
