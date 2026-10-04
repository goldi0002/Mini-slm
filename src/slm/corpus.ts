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
 * Simple, grammatical English sentences used to pre-train the network.
 *
 * `BASE_CORPUS` above is a fluent *dialogue* seed for the memory layer; this
 * corpus is the language-modelling data the transformer itself learns from. It
 * covers the grammar a learner meets first — "to be", present simple and
 * continuous, past and future, questions, negation, articles, plurals,
 * prepositions, possession and comparison — in short sentences whose structure
 * repeats often enough for a small model to pick it up.
 *
 * The tokenizer learns these words as whole-word tokens (so nothing here can
 * ever be spelled out letter by letter), and the pre-training warm-up runs real
 * gradient descent over the sentences. Add more sentences here — or import a
 * dataset through the Dataset Manager — to keep teaching it English.
 *
 * Format note for dataset authors: a pure next-word-prediction corpus lives in
 * plain sentences (no `<user>` / `<assistant>` markers). Dataset turns that are
 * meant to be spoken by the assistant instead follow the `ConversationTurn`
 * shape used by the UI — a `user` prompt and an `assistant` reply — and stay
 * dialogue-oriented.
 */
export const ENGLISH_LEARNING_CORPUS: string[] = [
  // "to be" — present
  'I am a student .',
  'you are my friend .',
  'he is a teacher .',
  'she is at home today .',
  'it is a beautiful morning .',
  'we are ready to begin .',
  'they are in the garden .',
  'the sky is blue and clear .',
  'this book is very interesting .',
  'my brother is a doctor .',
  'the children are happy and calm .',
  'I am not tired today .',
  'she is not angry with you .',
  'we are not late for the lesson .',
  // Questions with "to be"
  'are you ready to start ?',
  'is she your sister ?',
  'where are my keys ?',
  'how old is your brother ?',
  'why is the door open ?',
  'who is that young man ?',
  // Present simple
  'I like warm tea in the morning .',
  'you speak English very well .',
  'we study together every evening .',
  'they live near the river .',
  'she works in a small office .',
  'he reads a book before bed .',
  'the dog runs across the field .',
  'my father drives to work every day .',
  'the train leaves at seven in the morning .',
  'I do not drink coffee at night .',
  'he does not like cold weather .',
  'we do not go out on rainy days .',
  // Questions in the present
  'do you like this song ?',
  'does she speak French ?',
  'what do you do on weekends ?',
  'where does your family live ?',
  'how do you make this soup ?',
  'when does the class begin ?',
  // Present continuous
  'I am reading a story about the sea .',
  'she is cooking dinner for her family .',
  'they are playing football in the park .',
  'we are learning English together .',
  'the baby is sleeping in the next room .',
  'it is raining outside right now .',
  'my friends are waiting for me .',
  'he is not listening to the radio .',
  // Past simple
  'I walked to the market yesterday .',
  'she visited her grandmother last week .',
  'we watched a film together last night .',
  'they played cards after dinner .',
  'he finished his work before noon .',
  'the bus arrived ten minutes late .',
  'I did not see the message .',
  'she did not come to the party .',
  'where did you find this book ?',
  'when did they move to the city ?',
  // Future
  'I will call you tomorrow morning .',
  'we will meet at the station .',
  'she will help you with your homework .',
  'it will rain again in the evening .',
  'they will arrive before sunset .',
  'I am going to study tonight .',
  'we are going to travel next summer .',
  // Everyday needs and polite speech
  'I would like a glass of water , please .',
  'could you help me with this bag ?',
  'may I ask you a question ?',
  'would you like some tea or coffee ?',
  'thank you for your kind help .',
  'I am sorry for the mistake .',
  'excuse me , where is the station ?',
  'please open the window a little .',
  'can you speak more slowly , please ?',
  'I do not understand this word yet .',
  'what does this sentence mean ?',
  'how do you say this in English ?',
  'could you repeat that one more time ?',
  // Contractions as they are actually written
  "I'm learning English every day .",
  "you're my best friend .",
  "it's a lovely evening .",
  "we're going to the market .",
  "that's a good idea .",
  "I don't have any questions .",
  "she can't come tonight .",
  // Articles, plurals, quantities
  'there is a lamp on the table .',
  'there are two chairs near the door .',
  'I bought an apple and some bread .',
  'the cats are sleeping on the sofa .',
  'these shoes are too small for me .',
  'those flowers are beautiful in spring .',
  'I need a little more time .',
  'we have very few eggs left .',
  'she has many friends in the city .',
  // Prepositions of place and time
  'the keys are under the cushion .',
  'the book is between the lamp and the cup .',
  'we walked through the old town .',
  'she put the letter on the shelf .',
  'I will see you at nine in the morning .',
  'he was born in a small village .',
  // Possession and description
  "my sister's room is very bright .",
  "the teacher's desk stands near the window .",
  'this is our favourite place to rest .',
  'that old house has a red roof .',
  'her voice sounds calm and kind .',
  // Comparison
  'this road is longer than the other one .',
  'today is colder than yesterday .',
  'she runs faster than her brother .',
  'the blue shirt is the cheapest one .',
  'this is the best tea in the shop .',
  // Because, when, if, so
  'I stayed home because it was raining .',
  'we will start when everyone arrives .',
  'if you practise daily , you will improve .',
  'she was tired , so she went to bed early .',
  'although it was late , we kept talking .',
  // Feelings and everyday talk
  'I feel happy when I hear this music .',
  'he is worried about the exam .',
  'we are excited about the trip .',
  'she looks tired after a long day .',
  'that story made me smile .',
  'I hope you have a peaceful evening .',
  'learning a language takes time and patience .',
  'practice a little every day and you will improve .',
  'mistakes are a normal part of learning .',
  'reading aloud helps your pronunciation .',
  'speaking slowly makes your words clearer .',
  'I write new words in a small notebook .',
  'we review the lesson before the class .',
  'she listens to English songs every morning .',
  // Simple conversational answers
  'yes , I would like that very much .',
  'no , thank you , maybe next time .',
  'that sounds like a wonderful plan .',
  'I agree with you completely .',
  'I am not sure about that answer .',
  'let me think about it for a moment .',
  'of course , I am happy to help you .',
  'you are right , I forgot about that .',
  'it does not matter , we can try again .',
  'see you tomorrow , take care !',
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
