/**
 * Test script to evaluate the SLM against trained data across all datasets.
 */
import { PREDEFINED_MODELS, initializePretrainedModel } from '../src/slm/predefinedModels';
import { PREDEFINED_DATASETS } from '../src/slm/datasets';
import { SPECIAL_TOKENS } from '../src/slm/tokenizer';

async function runTest() {
  console.log('=== TESTING SLM AGAINST TRAINED DATA ===\n');

  for (const dataset of PREDEFINED_DATASETS) {
    console.log(`\n========================================`);
    console.log(`DATASET: ${dataset.name} (${dataset.turns.length} turns)`);
    console.log(`========================================`);

    // Fresh model instance
    const model = initializePretrainedModel(PREDEFINED_MODELS[0]);

    // Test BEFORE fine-tuning
    console.log('\n--- BASE MODEL RESPONSES (BEFORE FINE-TUNING) ---');
    for (let i = 0; i < Math.min(3, dataset.turns.length); i++) {
      const turn = dataset.turns[i];
      const prompt = model.tokenizer.formatConversationPrompt(turn.user);
      const res = model.generate(prompt, {
        temperature: 0.2,
        topK: 10,
        topP: 0.9,
        repetitionPenalty: 1.1,
        maxNewTokens: 35,
      });
      console.log(`[Q]: ${turn.user}`);
      console.log(`[TARGET]: ${turn.assistant}`);
      console.log(`[PRED]:   ${res.text}\n`);
    }

    // Now Fine-Tune on the dataset for 10 epochs
    console.log('--- FINE-TUNING (10 epochs) ---');
    const totalEpochs = 10;
    for (let epoch = 1; epoch <= totalEpochs; epoch++) {
      let epochLoss = 0;
      for (const turn of dataset.turns) {
        const formatted = `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
        const tokens = model.tokenizer.encode(formatted, true, true);
        const { loss } = model.trainStep(tokens, 0.015, true, 0.005);
        epochLoss += loss;
      }
      if (epoch === 1 || epoch === 5 || epoch === 10) {
        console.log(`  Epoch ${epoch}: avg loss = ${(epochLoss / dataset.turns.length).toFixed(4)}`);
      }
    }

    // Test AFTER fine-tuning on ALL dataset turns
    console.log('\n--- EVALUATION ON ALL TRAINED TURNS (AFTER FINE-TUNING) ---');
    let totalOverlap = 0;
    let successfulTurns = 0;

    for (let i = 0; i < dataset.turns.length; i++) {
      const turn = dataset.turns[i];
      const prompt = model.tokenizer.formatConversationPrompt(turn.user);
      const res = model.generate(prompt, {
        temperature: 0.2,
        topK: 10,
        topP: 0.9,
        repetitionPenalty: 1.1,
        maxNewTokens: 35,
      });

      // Measure target word overlap
      const targetWords = turn.assistant.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(Boolean);
      const predWords = res.text.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(Boolean);
      const targetSet = new Set(targetWords);
      let matchCount = 0;
      for (const w of predWords) {
        if (targetSet.has(w)) matchCount++;
      }
      const overlap = targetWords.length > 0 ? (matchCount / targetWords.length) : 0;
      totalOverlap += overlap;

      const isPass = overlap >= 0.4;
      if (isPass) successfulTurns++;

      console.log(`[Turn ${i + 1}] (${isPass ? 'PASS' : 'WARN'}) Overlap: ${(overlap * 100).toFixed(1)}%`);
      console.log(`  Q:      ${turn.user}`);
      console.log(`  TARGET: ${turn.assistant.slice(0, 70)}...`);
      console.log(`  MODEL:  ${res.text.slice(0, 70)}...`);
    }

    const avgOverlap = (totalOverlap / dataset.turns.length) * 100;
    console.log(`\nDataset Summary: ${successfulTurns}/${dataset.turns.length} turns matched >=40%, Avg Overlap: ${avgOverlap.toFixed(1)}%`);
  }
}

runTest().catch(console.error);
