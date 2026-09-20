/**
 * Temporary diagnostic script: checks what the SLM actually generates
 * before/after fine-tuning. Run with: bun scripts/diagnose-slm.ts
 */
import { PREDEFINED_MODELS, initializePretrainedModel } from '../src/slm/predefinedModels';
import { PREDEFINED_DATASETS } from '../src/slm/datasets';
import { SPECIAL_TOKENS } from '../src/slm/tokenizer';
import { GenerationOptions } from '../src/types';

const options: GenerationOptions = {
  temperature: 0.7,
  topK: 25,
  topP: 0.85,
  repetitionPenalty: 1.15,
  maxNewTokens: 26,
};

function chat(model: ReturnType<typeof initializePretrainedModel>, user: string) {
  const prompt = model.tokenizer.formatConversationPrompt(user);
  const out = model.generate(prompt, options, true);
  console.log(`USER:  ${user}\nMODEL: ${out.text}\n`);
}

console.log('=== BASE MODEL (pre-warmup) ===');
const model = initializePretrainedModel(PREDEFINED_MODELS[0]);
chat(model, 'hello who are you');
chat(model, 'how are you doing today');
chat(model, 'can you help me stay focused');
chat(model, 'what is the weather like');
chat(model, 'tell me something interesting');

console.log('=== AFTER FINE-TUNING on "Helpful Daily Assistant" (10 epochs) ===');
const preset = PREDEFINED_DATASETS[0];
for (let epoch = 0; epoch < 10; epoch++) {
  let sum = 0;
  for (const turn of preset.turns) {
    const text = `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
    const tokens = model.tokenizer.encode(text, true, true);
    const r = model.trainStep(tokens, 0.015, true, 0.005);
    sum += r.loss;
  }
  if (epoch === 0 || epoch === 4 || epoch === 9) {
    console.log(`epoch ${epoch + 1}: avg loss ${(sum / preset.turns.length).toFixed(3)}`);
  }
}
chat(model, 'hello who are you');
chat(model, 'how are you doing today');
chat(model, 'can you help me stay focused');
chat(model, 'what is the weather like');
chat(model, 'tell me something interesting');
