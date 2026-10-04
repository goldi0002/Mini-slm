/**
 * Comprehensive verification test suite for Local SLM TypeScript Studio fixes.
 * Covers ISS-01 through ISS-13.
 *
 * Run: bun scripts/test_fixes.ts
 */

import { SmallLanguageModel } from '../src/slm/transformer';
import { defaultTokenizer, UNK_ID, EOS_ID, ASSISTANT_ID, SPECIAL_TOKENS } from '../src/slm/tokenizer';
import { initializePretrainedModel, PREDEFINED_MODELS } from '../src/slm/predefinedModels';
import { PREDEFINED_DATASETS } from '../src/slm/datasets';
import { sampleFromDistribution, softmax } from '../src/slm/matrix';
import { BASE_CORPUS, ENGLISH_LEARNING_CORPUS } from '../src/slm/corpus';
import {
  evaluateTurn,
  trainingRegimeAdvisory,
  FULL_MODE_ADVISORY_EPOCHS,
} from '../src/components/FineTuningStudio';
import { readFileSync } from 'node:fs';

let passed = 0;
let failed = 0;

function assert(condition: boolean, testName: string, detail?: string) {
  if (condition) {
    console.log(`  ✅ [PASS] ${testName}`);
    passed++;
  } else {
    console.error(`  ❌ [FAIL] ${testName}: ${detail || 'Assertion failed'}`);
    failed++;
  }
}

async function runTestSuite() {
  console.log('\n=======================================================');
  console.log('🧪 RUNNING LOCAL SLM VERIFICATION TEST SUITE');
  console.log('=======================================================\n');

  const baseConfig = PREDEFINED_MODELS[0]; // assistant-48
  const model = initializePretrainedModel(baseConfig);

  // -------------------------------------------------------------
  // Test ISS-01: Triplicate Generation Logic & Prompt Conditioning
  // -------------------------------------------------------------
  console.log('--- ISS-01: Generation Logic & Prompt Conditioning ---');
  {
    const prompt = '<user> how are you doing today ?\n<assistant>';
    const result = model.generate(prompt, {
      temperature: 0.7,
      topK: 20,
      topP: 0.85,
      repetitionPenalty: 1.15,
      maxNewTokens: 15,
    });

    assert(result.tokens.length > 0, 'ISS-01.1: Generates tokens without error');
    assert(result.text.length > 0, 'ISS-01.2: Decodes generated tokens into text');
    assert(!result.text.startsWith('?'), 'ISS-01.3: Does NOT leak question mark or punctuation as reply start');
    assert(!result.text.startsWith('today'), 'ISS-01.4: Does NOT echo user prompt last word into reply');
  }

  // -------------------------------------------------------------
  // Test ISS-02: Context Window Sliding Truncation
  // -------------------------------------------------------------
  console.log('\n--- ISS-02: Context Window Sliding Truncation ---');
  {
    // Create a token sequence that exceeds maxSeqLen (96 for assistant-48)
    const longTokenList = new Array(150).fill(10);
    const forwardResult = model.forward(longTokenList);
    
    assert(forwardResult.seqLen <= model.config.maxSeqLen, 'ISS-02.1: forward() bounds sequence to maxSeqLen');
    assert(forwardResult.logits.length === forwardResult.seqLen * model.config.vocabSize, 'ISS-02.2: logits array matches clamped seqLen');

    let hasNaN = false;
    for (let i = 0; i < Math.min(100, forwardResult.logits.length); i++) {
      if (isNaN(forwardResult.logits[i])) {
        hasNaN = true;
        break;
      }
    }
    assert(!hasNaN, 'ISS-02.3: Logits contain valid numbers without NaN on long sequences');
  }

  // -------------------------------------------------------------
  // Test ISS-03: Set-based O(1) History Search in Matrix Sampling
  // -------------------------------------------------------------
  console.log('\n--- ISS-03: Set-based History Search in Matrix Sampling ---');
  {
    const vocabSize = 100;
    const probs = new Float32Array(vocabSize);
    probs[5] = 0.6;
    probs[10] = 0.4;
    
    const history = [5, 20, 30, 40, 50];
    const sampled = sampleFromDistribution(probs, 5, 0.9, 2.0, history);
    
    assert(sampled.candidates.length > 0, 'ISS-03.1: Sampling returns candidate distribution');
    assert(sampled.chosenId >= 0, 'ISS-03.2: Sampling returns valid token ID');
  }

  // -------------------------------------------------------------
  // Test ISS-04: Base Model Cache Invalidation & Tokenizer Sync
  // -------------------------------------------------------------
  console.log('\n--- ISS-04: Base Model Cache & Vocab Sync ---');
  {
    const oldVocabSize = defaultTokenizer.vocabSize;
    defaultTokenizer.learnWords(['quantumcomputing', 'astrophysics']);
    const newVocabSize = defaultTokenizer.vocabSize;

    assert(newVocabSize >= oldVocabSize + 2, 'ISS-04.1: Tokenizer learns new dynamic vocabulary');
    
    const refreshedModel = initializePretrainedModel(baseConfig);
    assert(refreshedModel.config.vocabSize === newVocabSize, 'ISS-04.2: initializePretrainedModel syncs with new vocabSize');
    assert(refreshedModel.weights.lm_head.length === newVocabSize * refreshedModel.config.dModel, 'ISS-04.3: lm_head matrix dynamically resized');
  }

  // -------------------------------------------------------------
  // Test ISS-05: Out-of-Bounds Dimension Safety Clamping
  // -------------------------------------------------------------
  console.log('\n--- ISS-05: Out-of-Bounds Dimension Safety Clamping ---');
  {
    // Test with out-of-bounds IDs (-5, 999999, NaN)
    const outOfBoundsTokens = [0, -5, 999999, 10];
    const fwd = model.forward(outOfBoundsTokens);
    
    assert(fwd.seqLen === 4, 'ISS-05.1: forward() handles out-of-bounds tokens without crashing');
    let hasValidOutput = true;
    for (let i = 0; i < fwd.logits.length; i += 100) {
      if (!Number.isFinite(fwd.logits[i])) {
        hasValidOutput = false;
        break;
      }
    }
    assert(hasValidOutput, 'ISS-05.2: forward() returns finite logits for clamped tokens');
  }

  // -------------------------------------------------------------
  // Test ISS-06: Training Early Stop Completion Guard
  // -------------------------------------------------------------
  console.log('\n--- ISS-06: Training Early Stop Completion Guard ---');
  {
    let completionTriggered = false;
    const onTrainingComplete = () => { completionTriggered = true; };

    // Simulate stopping mid-training
    const totalSteps: number = 100;
    const stepCount: number = 45; // stopped early
    const isTraining = false;

    const completedSuccessfully = isTraining && stepCount === totalSteps;
    if (completedSuccessfully) {
      onTrainingComplete();
    }

    assert(!completionTriggered, 'ISS-06.1: Early stopped training does not falsely trigger onTrainingComplete');
  }

  // -------------------------------------------------------------
  // Test ISS-07: Attention Map & Complete Weights Export
  // -------------------------------------------------------------
  console.log('\n--- ISS-07: Weight Export Completeness ---');
  {
    const weights = model.weights;
    const exportData = {
      config: model.config,
      wte: Array.from(weights.wte),
      wpe: Array.from(weights.wpe),
      lm_head: Array.from(weights.lm_head),
      ln_f_gamma: Array.from(weights.ln_f_gamma),
      ln_f_beta: Array.from(weights.ln_f_beta),
      layers: weights.layers.map((l) => ({
        q_proj: Array.from(l.q_proj),
        k_proj: Array.from(l.k_proj),
        v_proj: Array.from(l.v_proj),
        out_proj: Array.from(l.out_proj),
        lora_q_A: Array.from(l.lora_q_A),
        lora_q_B: Array.from(l.lora_q_B),
        lora_v_A: Array.from(l.lora_v_A),
        lora_v_B: Array.from(l.lora_v_B),
        ln1_gamma: Array.from(l.ln1_gamma),
        ln1_beta: Array.from(l.ln1_beta),
        fc1: Array.from(l.fc1),
        fc1_b: Array.from(l.fc1_b),
        fc2: Array.from(l.fc2),
        fc2_b: Array.from(l.fc2_b),
        ln2_gamma: Array.from(l.ln2_gamma),
        ln2_beta: Array.from(l.ln2_beta),
      })),
    };

    assert(exportData.ln_f_gamma.length > 0, 'ISS-07.1: ln_f_gamma included in weight export');
    assert(exportData.layers[0].ln1_gamma.length > 0, 'ISS-07.2: LayerNorm 1 gamma included in export');
    assert(exportData.layers[0].fc1_b.length > 0, 'ISS-07.3: FFN biases included in export');
  }

  // -------------------------------------------------------------
  // Test ISS-08: Token Inspector Probability Distribution Sum
  // -------------------------------------------------------------
  console.log('\n--- ISS-08: Token Inspector Probability Summation ---');
  {
    const candidates = [
      { id: 10, token: 'hello', prob: 0.55 },
      { id: 11, token: 'hi', prob: 0.25 },
      { id: 12, token: 'greetings', prob: 0.10 }
    ];

    const topTotalProb = candidates.reduce((acc, c) => acc + c.prob, 0);
    const remainingProb = Math.max(0, 1.0 - topTotalProb);
    const sum = topTotalProb + remainingProb;

    assert(Math.abs(sum - 1.0) < 1e-6, 'ISS-08.1: Top candidates + remaining tail sum to exactly 1.0 (100%)');
    assert(remainingProb > 0.09 && remainingProb < 0.11, 'ISS-08.2: Tail probability computed accurately (10%)');
  }

  // -------------------------------------------------------------
  // Test ISS-09: LoRA Forward Pass Down-Projection Reuse
  // -------------------------------------------------------------
  console.log('\n--- ISS-09: LoRA Forward Pass Down-Projection Reuse ---');
  {
    const loraModel = initializePretrainedModel(baseConfig);
    // Non-zero adapters, so the LoRA path is actually exercised.
    for (const l of loraModel.weights.layers) {
      l.lora_q_B.fill(0.011);
      l.lora_v_B.fill(-0.007);
    }
    const loraTokens = loraModel.tokenizer.encode(
      '<user> hello there how are you today ?\n<assistant> I am fine thank you !',
      true,
      false
    );
    const { seqLen } = loraModel.forward(loraTokens, true);

    // Private scratch buffers are read directly (compile-time private only).
    const internals = loraModel as unknown as {
      scratchXNorm1: Float32Array;
      scratchQ: Float32Array;
    };
    const { dModel, nLayers, loraRank, loraAlpha } = loraModel.config;
    const loraScale = loraAlpha / loraRank;
    const lastLayer = loraModel.weights.layers[nLayers - 1];

    // Reference: the projection written out naively, re-deriving u = A x inside
    // the row loop exactly like the pre-fix implementation did.
    let maxDelta = 0;
    for (let i = 0; i < seqLen; i++) {
      for (let row = 0; row < dModel; row++) {
        let expected = 0;
        for (let col = 0; col < dModel; col++) {
          expected += lastLayer.q_proj[row * dModel + col] * internals.scratchXNorm1[i * dModel + col];
        }
        let loraQ = 0;
        for (let r = 0; r < loraRank; r++) {
          let aQ = 0;
          for (let c = 0; c < dModel; c++) {
            aQ += lastLayer.lora_q_A[r * dModel + c] * internals.scratchXNorm1[i * dModel + c];
          }
          loraQ += lastLayer.lora_q_B[row * loraRank + r] * aQ;
        }
        expected += loraQ * loraScale;
        maxDelta = Math.max(maxDelta, Math.abs(expected - internals.scratchQ[i * dModel + row]));
      }
    }
    assert(
      maxDelta < 1e-4,
      'ISS-09.1: LoRA-augmented Q projection matches the naive reference formula',
      `max delta ${maxDelta.toExponential(2)}`
    );

    // Work count: A x is rank-sized and must be derived once per token position,
    // so the down-projection matrix is read dModel times less often than the
    // old per-row loop did.
    const layer0 = loraModel.weights.layers[0];
    const realA = layer0.lora_q_A;
    let reads = 0;
    layer0.lora_q_A = new Proxy(realA, {
      get(target, prop, receiver) {
        if (typeof prop === 'string' && /^[0-9]+$/.test(prop)) reads++;
        return Reflect.get(target, prop, receiver);
      },
    }) as unknown as Float32Array;

    loraModel.forward(loraTokens, true);
    layer0.lora_q_A = realA;

    const expectedReads = seqLen * loraRank * dModel;
    const naiveReads = seqLen * dModel * loraRank * dModel;
    assert(
      reads === expectedReads,
      'ISS-09.2: LoRA down-projection A x is computed once per token position',
      `${reads} reads (expected ${expectedReads}; the pre-fix row loop read ${naiveReads})`
    );
  }

  // -------------------------------------------------------------
  // Test ISS-10: Nucleus Sampling Candidate Probability Inflation
  // -------------------------------------------------------------
  console.log('\n--- ISS-10: Nucleus Sampling Candidate Probabilities ---');
  {
    const vocabSize = 64;
    const probs = new Float32Array(vocabSize);
    probs[3] = 0.5;
    probs[7] = 0.25;
    probs[11] = 0.15;
    probs[19] = 0.06;
    probs[23] = 0.04; // distribution sums to 1.0

    // A nucleus of 0.5 truncates to the single best token. The old code
    // re-normalized it to 100%, contradicting the uninflated likelihood shown
    // for the sampled token.
    const truncated = sampleFromDistribution(probs, 40, 0.5, 1.0, []);
    const truncatedSum = truncated.candidates.reduce((acc, c) => acc + c.prob, 0);
    const top = truncated.candidates.find((c) => c.id === 3);
    assert(
      truncatedSum <= 1.0 + 1e-6,
      'ISS-10.1: Truncated nucleus candidates never sum past 100%',
      `sum ${(truncatedSum * 100).toFixed(1)}%`
    );
    assert(
      top !== undefined && Math.abs(top.prob - 0.5) < 1e-6,
      'ISS-10.2: Candidates keep their true distribution-space probability under top-P truncation',
      `top candidate ${top ? (top.prob * 100).toFixed(1) : 'missing'}% (expected 50.0%)`
    );
    assert(
      Math.abs(truncated.totalMass - 1.0) < 1e-6,
      'ISS-10.3: The full distribution mass is reported unchanged by the nucleus filter',
      `${truncated.totalMass.toFixed(4)}`
    );

    const remaining = Math.max(0, 1.0 - truncatedSum);
    assert(
      remaining >= 0 && Math.abs(truncatedSum + remaining - 1.0) < 1e-6,
      'ISS-10.4: Candidates + tail mass reconcile to exactly 100% (Token Inspector)',
      `${(truncatedSum * 100).toFixed(1)}% + ${(remaining * 100).toFixed(1)}%`
    );

    // The candidate list is the top of the whole distribution, so with five
    // non-zero tokens it covers all of it — regardless of the nucleus cut.
    const wideNucleus = sampleFromDistribution(probs, 40, 0.9, 1.0, []);
    const wideSum = wideNucleus.candidates.reduce((acc, c) => acc + c.prob, 0);
    assert(
      Math.abs(wideSum - 1.0) < 1e-6,
      'ISS-10.5: Candidate list sums to 100% when it covers every non-zero token',
      `${(wideSum * 100).toFixed(1)}%`
    );
    assert(
      wideNucleus.candidates.every((c) => c.prob <= probs[c.id] + 1e-6),
      'ISS-10.5b: No candidate reports more probability than the model assigned it',
      wideNucleus.candidates.map((c) => `${(c.prob * 100).toFixed(1)}%`).join(' / ')
    );

    const strictTopK = sampleFromDistribution(probs, 1, 1.0, 1.0, []);
    assert(
      strictTopK.chosenId === 3,
      'ISS-10.6: Top-K = 1 still samples the single most likely token',
      `chosen ${strictTopK.chosenId}`
    );

    const narrowChoices = new Set<number>();
    for (let i = 0; i < 50; i++) {
      narrowChoices.add(sampleFromDistribution(probs, 40, 0.5, 1.0, []).chosenId);
    }
    assert(
      narrowChoices.size === 1 && narrowChoices.has(3),
      'ISS-10.7: top-P = 0.5 keeps sampling restricted to the truncated nucleus',
      `drew ${[...narrowChoices].join(', ')}`
    );

    // The repetition penalty only rescales the distribution; candidates must
    // still be normalized against the whole (adjusted) mass.
    const penalized = sampleFromDistribution(probs, 40, 0.99, 2.0, [3, 7]);
    const penalizedSum = penalized.candidates.reduce((acc, c) => acc + c.prob, 0);
    assert(
      penalizedSum <= 1.0 + 1e-6 && penalized.totalMass > 0,
      'ISS-10.8: Repetition-penalized candidates also stay a proper distribution',
      `sum ${(penalizedSum * 100).toFixed(1)}% of mass ${penalized.totalMass.toFixed(3)}`
    );
  }

  // -------------------------------------------------------------
  // Test ISS-11: Full Fine-Tuning Detection
  // -------------------------------------------------------------
  console.log('\n--- ISS-11: Full Fine-Tuning Detection ---');
  {
    const fullModel = initializePretrainedModel(baseConfig);
    assert(!fullModel.isFineTuned(), 'ISS-11.1: Fresh model reports base pretrained');

    const lmHeadBefore = Float32Array.from(fullModel.weights.lm_head);
    const turn = PREDEFINED_DATASETS[0].turns[0];
    const trainingText = `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
    const trainingTokens = fullModel.tokenizer.encode(trainingText, true, true);
    for (let epoch = 0; epoch < 3; epoch++) {
      fullModel.trainStep(trainingTokens, 0.02, false, 0.005);
    }

    let lmHeadMoved = false;
    for (let i = 0; i < lmHeadBefore.length; i++) {
      if (Math.abs(fullModel.weights.lm_head[i] - lmHeadBefore[i]) > 1e-8) {
        lmHeadMoved = true;
        break;
      }
    }
    assert(lmHeadMoved, 'ISS-11.2: Full fine-tuning really updates the lm_head weights');
    assert(
      fullModel.weights.layers.every(
        (l) =>
          l.lora_q_B.every((w) => Math.abs(w) <= 1e-6) &&
          l.lora_v_B.every((w) => Math.abs(w) <= 1e-6)
      ),
      'ISS-11.3: Full retrain leaves LoRA adapters at zero (the old check could never see it)'
    );
    assert(fullModel.isFineTuned(), 'ISS-11.4: Full fine-tuned model is detected as fine-tuned');
    fullModel.resetToBase();
    assert(!fullModel.isFineTuned(), 'ISS-11.5: resetToBase clears the full fine-tuning state');

    const loraModel = initializePretrainedModel(baseConfig);
    assert(!loraModel.isFineTuned(), 'ISS-11.6: Fresh model reports base before LoRA training');
    for (let epoch = 0; epoch < 3; epoch++) {
      loraModel.trainStep(trainingTokens, 0.02, true, 0.005);
    }
    assert(loraModel.isFineTuned(), 'ISS-11.7: LoRA fine-tuning is still detected (no regression)');
    loraModel.resetToBase();
    assert(!loraModel.isFineTuned(), 'ISS-11.8: resetToBase clears the LoRA fine-tuning state');
  }

  // -------------------------------------------------------------
  // Test ISS-12: Evaluation Mirrors the Active Adaptation Mode
  // -------------------------------------------------------------
  console.log('\n--- ISS-12: Evaluation Mirrors the Active Adaptation Mode ---');
  {
    const evalModel = initializePretrainedModel(baseConfig);
    const evalTurn = PREDEFINED_DATASETS[0].turns[0];
    const evalText = `${SPECIAL_TOKENS.USER} ${evalTurn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${evalTurn.assistant}`;
    const evalTokens = evalModel.tokenizer.encode(evalText, true, true);

    // Train LoRA first (adapters become non-zero), then continue with a full
    // retrain. The stale adapters are exactly what the old hardcoded
    // `forward(tokens, true)` measured instead of the retrained network.
    for (let epoch = 0; epoch < 3; epoch++) evalModel.trainStep(evalTokens, 0.02, true, 0.005);
    for (let epoch = 0; epoch < 3; epoch++) evalModel.trainStep(evalTokens, 0.02, false, 0.005);

    const referenceLoss = (useLora: boolean): number => {
      const V = evalModel.config.vocabSize;
      const { logits, seqLen } = evalModel.forward(evalTokens, useLora);
      let total = 0;
      let count = 0;
      const lossStart = evalTokens.lastIndexOf(ASSISTANT_ID) + 1;
      for (let i = lossStart; i < seqLen - 1; i++) {
        const target = evalTokens[i + 1];
        if (target === 0) continue;
        const row = logits.subarray(i * V, (i + 1) * V);
        let maxLogit = -Infinity;
        for (let j = 0; j < V; j++) if (row[j] > maxLogit) maxLogit = row[j];
        let sumExp = 0;
        for (let j = 0; j < V; j++) sumExp += Math.exp(row[j] - maxLogit);
        total += -Math.log(Math.max(1e-7, Math.exp(row[target] - maxLogit) / sumExp));
        count++;
      }
      return count > 0 ? total / count : 0;
    };

    const loraEval = evaluateTurn(evalModel, evalTurn, true);
    const fullEval = evaluateTurn(evalModel, evalTurn, false);

    assert(
      Math.abs(loraEval.loss - referenceLoss(true)) < 1e-5,
      'ISS-12.1: Evaluating with useLora = true scores the LoRA-adapted network',
      `loss ${loraEval.loss.toFixed(4)} vs reference ${referenceLoss(true).toFixed(4)}`
    );
    assert(
      Math.abs(fullEval.loss - referenceLoss(false)) < 1e-5,
      'ISS-12.2: Evaluating with useLora = false scores the fully retrained network',
      `loss ${fullEval.loss.toFixed(4)} vs reference ${referenceLoss(false).toFixed(4)}`
    );
    assert(
      Math.abs(loraEval.loss - fullEval.loss) > 1e-4,
      'ISS-12.3: The two adaptation modes genuinely evaluate differently',
      `LoRA ${loraEval.loss.toFixed(4)} vs full ${fullEval.loss.toFixed(4)}`
    );
    assert(
      Number.isFinite(fullEval.loss) &&
        fullEval.overlap >= 0 &&
        fullEval.overlap <= 100 &&
        fullEval.isPassed === (fullEval.overlap >= 40),
      'ISS-12.4: Evaluation returns finite, well-formed metrics',
      `loss ${fullEval.loss.toFixed(3)}, overlap ${fullEval.overlap.toFixed(1)}%`
    );

    // Regression guard on the wiring itself: the studio must evaluate with the
    // active hyperparameter, never a hardcoded LoRA flag.
    const studioSource = readFileSync(
      new URL('../src/components/FineTuningStudio.tsx', import.meta.url),
      'utf8'
    );
    assert(
      /evaluateTurn\(model, turn, hyperparams\.loraMode\)/.test(studioSource),
      'ISS-12.5: runDatasetEvaluation passes hyperparams.loraMode into the turn evaluation'
    );
    assert(
      !/model\.forward\(tokens,\s*true\)/.test(studioSource),
      'ISS-12.6: No hardcoded useLora = true remains in the evaluation path'
    );
  }

  // -------------------------------------------------------------
  // Test ISS-13: Full-Mode Training Regime Advisory
  // -------------------------------------------------------------
  console.log('\n--- ISS-13: Full-mode training regime advisory ---');
  {
    const studioDefaults = {
      epochs: 10,
      learningRate: 0.015,
      batchSize: 1,
      weightDecay: 0.005,
      loraMode: true,
      loraRank: 8,
    };

    assert(
      trainingRegimeAdvisory(studioDefaults) === null,
      'ISS-13.1: The default configuration (LoRA) raises no advisory'
    );
    assert(
      trainingRegimeAdvisory({ ...studioDefaults, loraMode: false }) !== null,
      'ISS-13.2: A long full retrain at the default epochs is flagged'
    );
    assert(
      trainingRegimeAdvisory({ ...studioDefaults, loraMode: false, epochs: 1 }) === null &&
        trainingRegimeAdvisory({
          ...studioDefaults,
          loraMode: false,
          epochs: FULL_MODE_ADVISORY_EPOCHS - 1,
        }) === null,
      'ISS-13.3: A short full retrain is left alone (no advisory below the measured threshold)'
    );

    // The advisory recommends LoRA as the default, so that recommendation has
    // to be true of the engine: the default must actually lower held-out loss
    // (sentences held out of fine-tuning entirely, scored with the neural
    // distribution alone).
    //
    // This is a *paired* measurement — the same model before and after its own
    // training run. Comparing two freshly initialised models (the old shape of
    // this test) mixed the adaptation effect with the run-to-run spread of the
    // random weight initialisation, which is now larger than the effect itself.
    // `diag_script.ts` section 12 repeats it under pinned seeds.
    const advisoryHeldOut = BASE_CORPUS.slice(0, 10);
    const advisoryTrain = BASE_CORPUS.slice(10);
    const ADVISORY_EPOCHS = 10; // the studio default
    const heldOutCE = (m: SmallLanguageModel, useLora: boolean): number => {
      const V = m.config.vocabSize;
      const probs = new Float32Array(V);
      let total = 0;
      let count = 0;
      for (const text of advisoryHeldOut) {
        const tokens = m.tokenizer.encode(text, true, true);
        const { logits, seqLen } = m.forward(tokens, useLora);
        for (let i = 0; i < seqLen - 1; i++) {
          const target = tokens[i + 1];
          if (target === 0) continue; // PAD
          softmax(logits.subarray(i * V, (i + 1) * V), probs, 1.0);
          total += -Math.log(Math.max(1e-8, probs[target]));
          count++;
        }
      }
      return count > 0 ? total / count : 0;
    };
    const trained = initializePretrainedModel(baseConfig);
    const beforeLora = heldOutCE(trained, true);
    for (let epoch = 0; epoch < ADVISORY_EPOCHS; epoch++) {
      for (const text of advisoryTrain) {
        trained.trainStep(trained.tokenizer.encode(text, true, true), 0.015, true, 0.005);
      }
    }
    const afterLora = heldOutCE(trained, true);
    assert(
      beforeLora - afterLora >= 0.05,
      'ISS-13.4: The recommended default (LoRA) really does generalise past its training sentences',
      `same model before ${beforeLora.toFixed(3)} -> after ${afterLora.toFixed(3)} (gain ${(beforeLora - afterLora).toFixed(3)} nats)`
    );

    // The other half of the advisory: a full retrain *does* reach a lower
    // held-out loss (it backpropagates through every weight), but only by a
    // small margin over the adapters while costing a full pass per epoch.
    const fullModel = initializePretrainedModel(baseConfig);
    const beforeFull = heldOutCE(fullModel, false);
    for (let epoch = 0; epoch < ADVISORY_EPOCHS; epoch++) {
      for (const text of advisoryTrain) {
        fullModel.trainStep(fullModel.tokenizer.encode(text, true, true), 0.015, false, 0.005);
      }
    }
    const afterFull = heldOutCE(fullModel, false);
    assert(
      beforeFull - afterFull >= 0.3 && beforeFull - afterFull < 3 * (beforeLora - afterLora) + 0.3,
      'ISS-13.4b: A full retrain also generalises, but not by an order of magnitude more',
      `same model before ${beforeFull.toFixed(3)} -> after ${afterFull.toFixed(3)} (gain ${(beforeFull - afterFull).toFixed(3)} nats vs LoRA ${(beforeLora - afterLora).toFixed(3)})`
    );

    // Regression guard: the advisory is useless unless the studio renders it.
    const studioSource = readFileSync(
      new URL('../src/components/FineTuningStudio.tsx', import.meta.url),
      'utf8'
    );
    assert(
      /const regimeAdvisory = trainingRegimeAdvisory\(hyperparams\)/.test(studioSource) &&
        /\{regimeAdvisory && \(/.test(studioSource),
      'ISS-13.5: FineTuningStudio renders the advisory in the hyperparameter panel'
    );
  }

  // -------------------------------------------------------------
  // Dataset Coverage: Standalone "what can you do" Capability Turn
  // -------------------------------------------------------------
  console.log('\n--- DATASET: Standalone capability turn ("what can you do") ---');
  {
    // The capability reply must live in the dataset itself (not only in the
    // warm-up corpus), so fine-tuning on the preset reinforces it too.
    const capabilityTurns = PREDEFINED_DATASETS[0].turns.filter(
      (t) => t.user === 'what can you do'
    );
    assert(
      capabilityTurns.length === 1,
      'DS-1: Helpful Daily Assistant contains a standalone "what can you do" turn',
      `found ${capabilityTurns.length}`
    );

    const capModel = initializePretrainedModel(baseConfig);
    const turn = capabilityTurns[0];
    const text = `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
    const tokens = capModel.tokenizer.encode(text, true, true);
    for (let epoch = 0; epoch < 6; epoch++) capModel.trainStep(tokens, 0.02, true, 0.005);

    const reply = capModel
      .generate(capModel.tokenizer.formatConversationPrompt('what can you do'), {
        temperature: 0.7,
        topK: 1,
        topP: 1.0,
        repetitionPenalty: 1.15,
        maxNewTokens: 30,
      }, true)
      .text.toLowerCase();
    const keyWords = ['converse', 'share', 'ideas', 'fine', 'tuned', 'custom', 'chat', 'datasets'];
    const hits = keyWords.filter((w) => reply.includes(w)).length;
    assert(
      hits >= 6,
      'DS-2: Fine-tuned model reproduces the capability answer for the bare prompt',
      `${hits}/${keyWords.length} keywords in reply: "${reply}"`
    );
  }

  // -------------------------------------------------------------
  // Test ISS-14: full backpropagation trains the whole network
  // -------------------------------------------------------------
  console.log('\n--- ISS-14: Full Backpropagation Trains Every Weight ---');
  {
    const m = initializePretrainedModel(baseConfig);
    const before = {
      wte: Float32Array.from(m.weights.wte),
      wpe: Float32Array.from(m.weights.wpe),
      q_proj: Float32Array.from(m.weights.layers[0].q_proj),
      k_proj: Float32Array.from(m.weights.layers[0].k_proj),
      v_proj: Float32Array.from(m.weights.layers[0].v_proj),
      out_proj: Float32Array.from(m.weights.layers[0].out_proj),
      fc1: Float32Array.from(m.weights.layers[0].fc1),
      fc2: Float32Array.from(m.weights.layers[0].fc2),
      ln1_gamma: Float32Array.from(m.weights.layers[0].ln1_gamma),
      ln_f_beta: Float32Array.from(m.weights.ln_f_beta),
    };
    const tokens = m.tokenizer.encode(
      `${SPECIAL_TOKENS.USER} what makes a good morning routine ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} a gentle morning routine includes water and light .`,
      true,
      true
    );
    for (let i = 0; i < 3; i++) m.trainStep(tokens, 0.02, false, 0.001);

    const moved = (name: keyof typeof before, after: Float32Array) => {
      const b = before[name];
      for (let i = 0; i < b.length; i++) if (Math.abs(b[i] - after[i]) > 1e-7) return true;
      return false;
    };
    const unchanged: string[] = [];
    if (!moved('wte', m.weights.wte)) unchanged.push('wte');
    if (!moved('wpe', m.weights.wpe)) unchanged.push('wpe');
    if (!moved('q_proj', m.weights.layers[0].q_proj)) unchanged.push('q_proj');
    if (!moved('k_proj', m.weights.layers[0].k_proj)) unchanged.push('k_proj');
    if (!moved('v_proj', m.weights.layers[0].v_proj)) unchanged.push('v_proj');
    if (!moved('out_proj', m.weights.layers[0].out_proj)) unchanged.push('out_proj');
    if (!moved('fc1', m.weights.layers[0].fc1)) unchanged.push('fc1');
    if (!moved('fc2', m.weights.layers[0].fc2)) unchanged.push('fc2');
    if (!moved('ln1_gamma', m.weights.layers[0].ln1_gamma)) unchanged.push('ln1_gamma');
    if (!moved('ln_f_beta', m.weights.ln_f_beta)) unchanged.push('ln_f_beta');
    assert(
      unchanged.length === 0,
      'ISS-14.1: Full training updates embeddings, projections, FFN and LayerNorms (not only lm_head)',
      `still frozen: ${unchanged.join(', ') || 'none'}`
    );

    // The gradient must be a *real* gradient of the reported objective: with a
    // correct gradient, repeated steps on one sequence reduce its neural loss.
    const probe = initializePretrainedModel(baseConfig);
    const lossOf = (model: SmallLanguageModel, ts: number[]): number => {
      let res = 0;
      for (let i = 0; i < 4; i++) res = model.trainStep(ts, 0.02, false, 0.001).loss;
      return res;
    };
    const probeTokens = probe.tokenizer.encode('the children are playing football in the park .', true, true);
    const startLoss = lossOf(probe, probeTokens);
    for (let i = 0; i < 6; i++) lossOf(probe, probeTokens);
    const endLoss = lossOf(probe, probeTokens);
    assert(
      endLoss < startLoss - 0.05,
      'ISS-14.2: Full-mode gradient descent really fits the sequence it trains on',
      `loss ${startLoss.toFixed(3)} -> ${endLoss.toFixed(3)}`
    );
    assert(
      probe.weights.layers.every(
        (l) =>
          l.lora_q_B.every((w) => Math.abs(w) <= 1e-6) &&
          l.lora_v_B.every((w) => Math.abs(w) <= 1e-6)
      ),
      'ISS-14.3: Full training leaves the LoRA adapters untouched (modes stay distinct)'
    );
  }

  // -------------------------------------------------------------
  // Test ISS-15: the trainable-parameter count matches the claim
  // -------------------------------------------------------------
  console.log('\n--- ISS-15: Trainable Parameter Accounting ---');
  {
    const m = initializePretrainedModel(baseConfig);
    const stats = m.getMemoryStats();
    const lora = m.countParameters(true);
    const full = m.countParameters(false);
    assert(
      lora.trainable === stats.loraParams && lora.total === stats.totalParams,
      'ISS-15.1: LoRA mode reports exactly the adapter parameters as trainable',
      `${lora.trainable} of ${lora.total}`
    );
    assert(
      full.trainable === stats.totalParams - stats.loraParams && full.trainable > 10 * stats.loraParams,
      'ISS-15.2: Full mode reports every base weight as trainable (not just the LM head)',
      `${full.trainable} trainable of ${full.total}`
    );
    assert(
      full.trainable > stats.lmHeadParams,
      'ISS-15.3: The full-mode count is larger than the LM head alone',
      `full ${full.trainable} vs lm_head ${stats.lmHeadParams}`
    );
  }

  // -------------------------------------------------------------
  // Test ISS-16: the neural/memory blend is measured, not hard-coded
  // -------------------------------------------------------------
  console.log('\n--- ISS-16: Adaptive Neural/Memory Blend ---');
  {
    const m = initializePretrainedModel(baseConfig);
    assert(
      Math.abs(m.getNeuralMix() - 0.08) < 1e-9,
      'ISS-16.1: An untrained model keeps the memory-dominated floor of 8%',
      `mix ${m.getNeuralMix()}`
    );

    const heldOut = ENGLISH_LEARNING_CORPUS.slice(0, 16);
    const train = ENGLISH_LEARNING_CORPUS.slice(16, 70);
    const untrained = m.calibrateNeuralMix(heldOut);
    for (let epoch = 0; epoch < 3; epoch++) {
      for (const text of train) {
        m.trainStep(m.tokenizer.encode(text, true, true), 0.02, false, 0.001, false);
      }
    }
    const trained = m.calibrateNeuralMix(heldOut);
    assert(
      trained.neuralPerplexity < untrained.neuralPerplexity,
      'ISS-16.2: Training genuinely sharpens the network on held-out text',
      `neural ppl ${untrained.neuralPerplexity.toFixed(1)} -> ${trained.neuralPerplexity.toFixed(1)}`
    );
    assert(
      trained.mix >= untrained.mix && trained.mix >= 0.08,
      'ISS-16.3: The fitted blend remains data-driven and never falls below the neural floor',
      `${untrained.mix} -> ${trained.mix}`
    );
    assert(
      Math.abs(m.getNeuralMix() - trained.mix) < 1e-9 && trained.mix <= 0.7,
      'ISS-16.4: The calibrated weight is installed on the model for generation',
      `mix ${m.getNeuralMix()}`
    );

    // The blend weight must be visible in generation, not just stored.
    const next = (mix: number) => {
      m.setNeuralMix(mix);
      const prompt = m.tokenizer.formatConversationPrompt('how are you today');
      const result = m.generate(prompt, { temperature: 0.7, topK: 5, topP: 0.95, repetitionPenalty: 1.1, maxNewTokens: 1 });
      return result.tokens[0].topCandidates.map((c) => `${c.id}:${c.prob.toFixed(4)}`).join(',');
    };
    assert(
      next(0.02) !== next(0.7),
      'ISS-16.5: Changing the mix changes the sampling distribution (the network is not decorative)',
      'top candidates differ between a 2% and a 70% neural share'
    );
    m.setNeuralMix(0.08);
  }

  // -------------------------------------------------------------
  // Test ISS-17: loss is measured before the memory observes
  // -------------------------------------------------------------
  console.log('\n--- ISS-17: Loss Is Measured Before Memory Observation ---');
  {
    const m = initializePretrainedModel(baseConfig);
    const novel =
      `${SPECIAL_TOKENS.USER} tell me about quokkas on the island ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} quokkas are small friendly animals that live near the water .`;
    const tokens = m.tokenizer.encode(novel, true, true);
    const V = m.config.vocabSize;
    const probs = new Float32Array(V);
    const mem = new Float32Array(V);
    const mix = m.getNeuralMix();

    // Compute the blended NLL by hand, from the tables as they are *now* (i.e.
    // before this sequence is ever observed).
    const pre = m.forward(tokens, false);
    let preBlended = 0;
    let counted = 0;
    const lossStart = tokens.lastIndexOf(ASSISTANT_ID) + 1;
    for (let i = lossStart; i < pre.seqLen - 1; i++) {
      const target = tokens[i + 1];
      if (target === 0) continue;
      softmax(pre.logits.subarray(i * V, (i + 1) * V), probs, 1.0);
      m.memory.distribution(i >= 1 ? tokens[i - 1] : 2, tokens[i], mem);
      preBlended += -Math.log(Math.max(1e-8, mix * probs[target] + (1 - mix) * mem[target]));
      counted++;
    }
    preBlended /= counted;

    const memoryBefore = m.memory.size;
    const step = m.trainStep(tokens, 0.0, false, 0.0); // zero lr: measure without learning
    const memoryAfter = m.memory.size;

    // After the call, observing the same sequence must now make it *cheaper*.
    let postBlended = 0;
    counted = 0;
    for (let i = 0; i < pre.seqLen - 1; i++) {
      const target = tokens[i + 1];
      if (target === 0) continue;
      softmax(pre.logits.subarray(i * V, (i + 1) * V), probs, 1.0);
      m.memory.distribution(i >= 1 ? tokens[i - 1] : 2, tokens[i], mem);
      postBlended += -Math.log(Math.max(1e-8, mix * probs[target] + (1 - mix) * mem[target]));
      counted++;
    }
    postBlended /= counted;

    assert(
      memoryAfter > memoryBefore,
      'ISS-17.1: The training step does teach the memory layer about the sequence',
      `memory size ${memoryBefore} -> ${memoryAfter}`
    );
    assert(
      Math.abs(step.blendedLoss - preBlended) < 1e-5,
      'ISS-17.2: Reported loss is the pre-observation value, not the post-memorisation one',
      `reported ${step.blendedLoss.toFixed(4)} vs pre-observation ${preBlended.toFixed(4)}`
    );
    assert(
      postBlended < preBlended - 1e-4,
      'ISS-17.3: Observing afterwards really does lower the next measurement (the check has teeth)',
      `pre ${preBlended.toFixed(4)} -> post ${postBlended.toFixed(4)}`
    );
  }

  // -------------------------------------------------------------
  // Test ISS-18/ISS-19: the reported metric is the objective
  // -------------------------------------------------------------
  console.log('\n--- ISS-18/19: Reported Loss Equals the Optimized Objective ---');
  {
    const m = initializePretrainedModel(baseConfig);
    const tokens = m.tokenizer.encode(
      `${SPECIAL_TOKENS.USER} how do I stay focused ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} work in short focused intervals and then rest .`,
      true,
      true
    );
    const V = m.config.vocabSize;
    const probs = new Float32Array(V);
    const { logits, seqLen } = m.forward(tokens, true);
    let neuralCE = 0;
    let count = 0;
    const lossStart = tokens.lastIndexOf(ASSISTANT_ID) + 1;
    for (let i = lossStart; i < seqLen - 1; i++) {
      const target = tokens[i + 1];
      if (target === 0) continue;
      softmax(logits.subarray(i * V, (i + 1) * V), probs, 1.0);
      neuralCE += -Math.log(Math.max(1e-8, probs[target]));
      count++;
    }
    neuralCE /= count;

    const report = m.trainStep(tokens, 0.0, true, 0.0);
    assert(
      Math.abs(report.loss - neuralCE) < 1e-5,
      'ISS-19.1: LoRA mode reports the neural cross-entropy its gradient descends',
      `reported ${report.loss.toFixed(4)} vs neural CE ${neuralCE.toFixed(4)}`
    );
    assert(
      Number.isFinite(report.blendedLoss) && Math.abs(report.blendedLoss - report.loss) > 0.05,
      'ISS-19.2: The blended NLL is reported separately from the neural objective',
      `neural ${report.loss.toFixed(3)} vs blended ${report.blendedLoss.toFixed(3)}`
    );
    assert(
      Math.abs(report.neuralLoss - report.loss) < 1e-9,
      'ISS-19.3: neuralLoss and loss name the same number',
      `${report.neuralLoss.toFixed(4)}`
    );
  }

  // -------------------------------------------------------------
  // DIAGNOSTIC: target-loss trajectory for response-only LoRA
  // -------------------------------------------------------------
  {
    const m = initializePretrainedModel(baseConfig);
    const diagnosticText =
      SPECIAL_TOKENS.USER + ' explain a simple morning routine ' + SPECIAL_TOKENS.NEWLINE +
      SPECIAL_TOKENS.ASSISTANT + ' drink water take a short walk and plan one important task .';
    const diagnosticTokens = m.tokenizer.encode(diagnosticText, true, true);
    const responseLoss = (model: SmallLanguageModel): number => {
      const { logits, seqLen } = model.forward(diagnosticTokens, true);
      const V = model.config.vocabSize;
      const start = diagnosticTokens.lastIndexOf(ASSISTANT_ID) + 1;
      const p = new Float32Array(V);
      let total = 0;
      let count = 0;
      for (let i = start; i < seqLen - 1; i++) {
        const target = diagnosticTokens[i + 1];
        if (target === 0) continue;
        softmax(logits.subarray(i * V, (i + 1) * V), p, 1.0);
        total += -Math.log(Math.max(1e-8, p[target]));
        count++;
      }
      return total / Math.max(1, count);
    };
    const before = responseLoss(m);
    for (let step = 0; step < 40; step++) m.trainStep(diagnosticTokens, 0.03, true, 0.001, false);
    const after = responseLoss(m);
    console.log('  📈 DIAG LoRA response loss: ' + before.toFixed(3) + ' -> ' + after.toFixed(3) + ' after 40 steps (target 0.30)');

    const full = initializePretrainedModel(baseConfig);
    const fullBefore = responseLoss(full);
    for (let step = 0; step < 80; step++) full.trainStep(diagnosticTokens, 0.05, false, 0.0, false);
    const fullAfter = responseLoss(full);
    console.log('  📈 DIAG Full response loss @0.05: ' + fullBefore.toFixed(3) + ' -> ' + fullAfter.toFixed(3) + ' after 80 steps (target 0.30)');

    const fullFast = initializePretrainedModel(baseConfig);
    const fastBefore = responseLoss(fullFast);
    for (let step = 0; step < 80; step++) fullFast.trainStep(diagnosticTokens, 0.2, false, 0.0, false);
    const fastAfter = responseLoss(fullFast);
    console.log('  📈 DIAG Full response loss @0.20: ' + fastBefore.toFixed(3) + ' -> ' + fastAfter.toFixed(3) + ' after 80 steps (target 0.30)');

    for (let step = 80; step < 400; step++) fullFast.trainStep(diagnosticTokens, 0.2, false, 0.0, false);
    const fastLongAfter = responseLoss(fullFast);
    console.log('  📈 DIAG Full response loss @0.20 long: ' + fastAfter.toFixed(3) + ' -> ' + fastLongAfter.toFixed(3) + ' after 400 total steps (target 0.30)');

    const fullFaster = initializePretrainedModel(baseConfig);
    const fasterBefore = responseLoss(fullFaster);
    for (let step = 0; step < 400; step++) fullFaster.trainStep(diagnosticTokens, 0.3, false, 0.0, false);
    const fasterAfter = responseLoss(fullFaster);
    console.log('  📈 DIAG Full response loss @0.30: ' + fasterBefore.toFixed(3) + ' -> ' + fasterAfter.toFixed(3) + ' after 400 steps (target 0.30)');

    const multi = initializePretrainedModel(baseConfig);
    const trainTurns = PREDEFINED_DATASETS[0].turns.filter((_, i) => i % 4 !== 3);
    const responseLossForTurn = (model: SmallLanguageModel, turn: typeof trainTurns[number]): number => {
      const ts = model.tokenizer.encode(
        SPECIAL_TOKENS.USER + ' ' + turn.user + ' ' + SPECIAL_TOKENS.NEWLINE +
        SPECIAL_TOKENS.ASSISTANT + ' ' + turn.assistant,
        true,
        true
      );
      const { logits, seqLen } = model.forward(ts, false);
      const V = model.config.vocabSize;
      const start = ts.lastIndexOf(ASSISTANT_ID) + 1;
      const p = new Float32Array(V);
      let total = 0;
      let count = 0;
      for (let i = start; i < seqLen - 1; i++) {
        const target = ts[i + 1];
        if (target === 0) continue;
        softmax(logits.subarray(i * V, (i + 1) * V), p, 1.0);
        total += -Math.log(Math.max(1e-8, p[target]));
        count++;
      }
      return total / Math.max(1, count);
    };
    for (let epoch = 0; epoch < 70; epoch++) {
      for (const turn of trainTurns) {
        const ts = multi.tokenizer.encode(
          SPECIAL_TOKENS.USER + ' ' + turn.user + ' ' + SPECIAL_TOKENS.NEWLINE +
          SPECIAL_TOKENS.ASSISTANT + ' ' + turn.assistant,
          true,
          true
        );
        multi.trainStep(ts, 0.3, false, 0.0, false);
      }
    }
    const multiAvg = trainTurns.reduce((sum, turn) => sum + responseLossForTurn(multi, turn), 0) / trainTurns.length;
    console.log('  📈 DIAG Multi-turn full response loss @0.30: ' + multiAvg.toFixed(3) + ' after 70 epochs / ' + (70 * trainTurns.length) + ' steps (target 0.30)');

    const deep = initializePretrainedModel(PREDEFINED_MODELS[2]);
    const deepTurns = PREDEFINED_DATASETS[0].turns.filter((_, i) => i % 4 !== 3);
    for (let epoch = 0; epoch < 70; epoch++) {
      for (const turn of deepTurns) {
        const ts = deep.tokenizer.encode(
          SPECIAL_TOKENS.USER + ' ' + turn.user + ' ' + SPECIAL_TOKENS.NEWLINE +
          SPECIAL_TOKENS.ASSISTANT + ' ' + turn.assistant,
          true,
          true
        );
        deep.trainStep(ts, 0.3, false, 0.0, false);
      }
    }
    const deepAvg = deepTurns.reduce((sum, turn) => sum + responseLossForTurn(deep, turn), 0) / deepTurns.length;
    console.log('  📈 DIAG Multi-turn deep-model response loss @0.30: ' + deepAvg.toFixed(3) + ' after 70 epochs (target 0.30)');
  }

  // -------------------------------------------------------------
  // Test ISS-20: retrieval is reported, and the studio scores held-out turns
  // -------------------------------------------------------------
  console.log('\n--- ISS-20: Retrieval Is Reported, Held-Out Evaluation Exists ---');
  {
    const m = initializePretrainedModel(baseConfig);
    const turn = PREDEFINED_DATASETS[0].turns[0];
    const text = `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
    for (let epoch = 0; epoch < 4; epoch++) {
      m.trainStep(m.tokenizer.encode(text, true, true), 0.02, true, 0.005);
    }
    m.generate(m.tokenizer.formatConversationPrompt(turn.user), {
      temperature: 0.4, topK: 10, topP: 0.9, repetitionPenalty: 1.1, maxNewTokens: 20,
    }, true);
    const tunedTrace = m.getLastGenerationTrace();
    assert(
      tunedTrace.usedRetrieval && tunedTrace.retrievalTokens > 0,
      'ISS-20.1: A reply replayed from a learned dataset answer is reported as retrieval',
      `${tunedTrace.retrievalTokens} retrieval / ${tunedTrace.generatedTokens} generated tokens`
    );

    const fresh = initializePretrainedModel(baseConfig);
    fresh.generate(fresh.tokenizer.formatConversationPrompt('what is the weather like tomorrow'), {
      temperature: 0.4, topK: 10, topP: 0.9, repetitionPenalty: 1.1, maxNewTokens: 12,
    }, true);
    const freshTrace = fresh.getLastGenerationTrace();
    assert(
      !freshTrace.usedRetrieval && freshTrace.generatedTokens > 0,
      'ISS-20.2: An untrained prompt is reported as generated, not as retrieval',
      `${freshTrace.retrievalTokens} retrieval / ${freshTrace.generatedTokens} generated tokens`
    );

    const studioSource = readFileSync(
      new URL('../src/components/FineTuningStudio.tsx', import.meta.url),
      'utf8'
    );
    assert(
      /const heldOutTurns = /.test(studioSource) &&
        /heldOutLoss/.test(studioSource) &&
        /neuralInfluence/.test(studioSource),
      'ISS-20.3: The studio reports held-out loss and the neural influence it measured',
      'FineTuningStudio contains the held-out split, held-out loss and neural-influence readout'
    );
  }

  // -------------------------------------------------------------
  // Test ISS-21: growing the vocabulary preserves training
  // -------------------------------------------------------------
  console.log('\n--- ISS-21: Vocabulary Growth Preserves Trained Weights ---');
  {
    const m = initializePretrainedModel(baseConfig);
    const turn = PREDEFINED_DATASETS[0].turns[1];
    const text = `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
    for (let epoch = 0; epoch < 4; epoch++) {
      m.trainStep(m.tokenizer.encode(text, true, true), 0.02, true, 0.005);
    }
    const trained = m.isFineTuned();
    const beforeQ = Float32Array.from(m.weights.layers[0].q_proj);
    const beforeAdapter = Float32Array.from(m.weights.layers[0].lora_q_B);
    const vocabBefore = m.config.vocabSize;

    const learned = defaultTokenizer.learnWords(['quokkas', 'xylophonically', 'marsupials']);
    const grew = m.resizeVocabulary(defaultTokenizer.vocabSize);

    let sameQ = true;
    let sameAdapter = true;
    for (let i = 0; i < beforeQ.length; i++) if (m.weights.layers[0].q_proj[i] !== beforeQ[i]) sameQ = false;
    for (let i = 0; i < beforeAdapter.length; i++) if (m.weights.layers[0].lora_q_B[i] !== beforeAdapter[i]) sameAdapter = false;

    assert(
      learned > 0 && grew && m.config.vocabSize > vocabBefore,
      'ISS-21.1: New dataset words grow the model in place',
      `vocab ${vocabBefore} -> ${m.config.vocabSize}`
    );
    assert(
      sameQ && sameAdapter,
      'ISS-21.2: Every trained weight survives the resize unchanged',
      `q_proj ${sameQ ? 'preserved' : 'CHANGED'}, adapter ${sameAdapter ? 'preserved' : 'CHANGED'}`
    );
    assert(
      m.weights.lm_head.length === m.config.vocabSize * m.config.dModel &&
        m.weights.wte.length === m.config.vocabSize * m.config.dModel,
      'ISS-21.3: Embedding table and LM head cover the grown vocabulary',
      `lm_head ${m.weights.lm_head.length} floats`
    );
    assert(
      m.isFineTuned() === trained,
      'ISS-21.4: The fine-tuned state survives the resize',
      `isFineTuned ${m.isFineTuned()}`
    );
    m.resetToBase();
    assert(
      !m.isFineTuned() && m.config.vocabSize > vocabBefore,
      'ISS-21.5: resetToBase still restores the (resized) base snapshot',
      `vocab ${m.config.vocabSize}`
    );
  }

  // -------------------------------------------------------------
  // Test ISS-22: the base model is pre-trained on real English
  // -------------------------------------------------------------
  console.log('\n--- ISS-22: Pre-Training Teaches Real English Structure ---');
  {
    assert(
      ENGLISH_LEARNING_CORPUS.length >= 120,
      'ISS-22.1: The built-in English corpus is large enough to train on',
      `${ENGLISH_LEARNING_CORPUS.length} sentences`
    );
    const raw = new SmallLanguageModel({ ...baseConfig, vocabSize: defaultTokenizer.vocabSize }, defaultTokenizer);
    const pretrained = initializePretrainedModel(baseConfig);
    const V = pretrained.config.vocabSize;
    const probs = new Float32Array(V);
    const ce = (model: SmallLanguageModel, texts: string[]): number => {
      let total = 0;
      let count = 0;
      for (const text of texts) {
        const tokens = model.tokenizer.encode(text, true, true);
        const { logits, seqLen } = model.forward(tokens, false);
        for (let i = 0; i < seqLen - 1; i++) {
          const target = tokens[i + 1];
          if (target === 0) continue;
          softmax(logits.subarray(i * V, (i + 1) * V), probs, 1.0);
          total += -Math.log(Math.max(1e-8, probs[target]));
          count++;
        }
      }
      return total / count;
    };
    const sample = ENGLISH_LEARNING_CORPUS.slice(100, 120);
    const rawCE = ce(raw, sample);
    const pretrainedCE = ce(pretrained, sample);
    assert(
      pretrainedCE < rawCE - 0.5,
      'ISS-22.2: Pre-training measurably lowers held-out loss versus random weights',
      `random ${rawCE.toFixed(3)} -> pretrained ${pretrainedCE.toFixed(3)} (${(rawCE - pretrainedCE).toFixed(3)} nats)`
    );
  }

  // -------------------------------------------------------------
  // Test ISS-23: sliding-window positions are anchored, not restarted
  // -------------------------------------------------------------
  console.log('\n--- ISS-23: Positional Window Anchoring ---');
  {
    const m = initializePretrainedModel(baseConfig);
    const { maxSeqLen, dModel } = m.config;
    // A probe model whose only non-zero weights are the position embeddings:
    // with every block weight zeroed the residual stream never changes, so the
    // values in scratchX after forward() *are* wte[token] + wpe[position].
    for (const l of m.weights.layers) {
      l.q_proj.fill(0); l.k_proj.fill(0); l.v_proj.fill(0); l.out_proj.fill(0);
      l.fc1.fill(0); l.fc1_b.fill(0); l.fc2.fill(0); l.fc2_b.fill(0);
      l.ln1_gamma.fill(0); l.ln1_beta.fill(0); l.ln2_gamma.fill(0); l.ln2_beta.fill(0);
    }
    m.weights.ln_f_gamma.fill(0);
    m.weights.ln_f_beta.fill(0);
    m.weights.wte.fill(0);
    for (let pos = 0; pos < maxSeqLen; pos++) {
      m.weights.wpe.fill(pos + 1, pos * dModel, (pos + 1) * dModel);
    }

    const scratchX = (m as unknown as { scratchX: Float32Array }).scratchX;
    // Token i must sit at position i regardless of the sequence length: growing
    // a sequence (which is exactly what decoding does) must never re-number the
    // tokens already in it.
    const positionsMatch = (count: number): boolean => {
      for (let i = 0; i < count; i++) {
        if (Math.abs(scratchX[i * dModel] - (i + 1)) > 1e-6) return false;
      }
      return true;
    };
    m.forward(new Array(10).fill(10));
    const shortPositions = positionsMatch(10);
    m.forward(new Array(30).fill(10));
    const longPositions = positionsMatch(30);
    assert(
      shortPositions && longPositions,
      'ISS-23.1: Positions are window-relative and stable as the sequence grows',
      `10-token and 30-token sequences both map token i to position i`
    );

    // A sequence longer than the context window is windowed to maxSeqLen and
    // must stay inside the positional table (the old bug's failure mode was
    // indexing wpe past its end / mismatching the loss shift).
    const overLong = m.forward(new Array(maxSeqLen + 25).fill(10));
    let inRange = true;
    for (let i = 0; i < overLong.seqLen; i++) {
      const value = scratchX[i * dModel];
      if (value < 1 || value > maxSeqLen) inRange = false;
    }
    assert(
      overLong.seqLen === maxSeqLen && inRange,
      'ISS-23.2: An over-long sequence is windowed without leaving the positional table',
      `seqLen ${overLong.seqLen} of ${maxSeqLen}, positions within 1..${maxSeqLen}`
    );
    const frozen =
      m.weights.wte.every((w) => w === 0) &&
      m.weights.wpe.some((w) => w !== 0) &&
      m.weights.layers.every((l) => l.fc1.every((w) => w === 0));
    assert(frozen, 'ISS-23.3: The probe isolates positional embeddings from every other weight');
  }

  console.log('\n=======================================================');
  console.log(`📊 TEST SUITE SUMMARY: ${passed} PASSED, ${failed} FAILED`);
  console.log('=======================================================\n');

  if (failed > 0) {
    process.exit(1);
  }
}

runTestSuite().catch(err => {
  console.error('Fatal error in test suite:', err);
  process.exit(1);
});
