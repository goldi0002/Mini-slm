/** Debug: inspect memory vs neural distributions for key contexts. */
import { PREDEFINED_MODELS, initializePretrainedModel } from '../src/slm/predefinedModels';

const model = initializePretrainedModel(PREDEFINED_MODELS[0]);
const tok = model.tokenizer;
const V = model.config.vocabSize;

function showContext(label: string, prev2str: string, prev1str: string) {
  const prev2 = tok.encode(prev2str, false, false).pop()!;
  const prev1 = tok.encode(prev1str, false, false).pop()!;
  const mem = new Float32Array(V);
  model.memory.distribution(prev2, prev1, mem);
  const memTop = [...mem.keys()]
    .sort((a, b) => mem[b] - mem[a])
    .slice(0, 6)
    .map((v) => `${JSON.stringify(tok.getTokenString(v))}:${mem[v].toFixed(3)}`);
  console.log(`${label} [p2=${JSON.stringify(tok.getTokenString(prev2))} p1=${JSON.stringify(tok.getTokenString(prev1))}]`);
  console.log(`  memory top: ${memTop.join('  ')}`);
}

showContext('after <assistant><space>', '<assistant>', ' ');
showContext('after "do"', 'what can you', 'do');
showContext('after "I"', 'and', 'I');
showContext('after "."', 'you', '.');
console.log('memory total size:', model.memory.size);
