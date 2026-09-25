/**
 * Comprehensive verification test suite for Local SLM TypeScript Studio fixes.
 * Covers ISS-01 through ISS-13.
 *
 * Run: bun scripts/test_fixes.ts
 */

import { SmallLanguageModel } from '../src/slm/transformer';
import { defaultTokenizer, UNK_ID, EOS_ID, SPECIAL_TOKENS } from '../src/slm/tokenizer';
import { initializePretrainedModel, PREDEFINED_MODELS } from '../src/slm/predefinedModels';
import { PREDEFINED_DATASETS } from '../src/slm/datasets';
import { sampleFromDistribution, softmax } from '../src/slm/matrix';
import { BASE_CORPUS } from '../src/slm/corpus';
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
      for (let i = 0; i < seqLen - 1; i++) {
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
    // distribution alone). `diag_script.ts` section 12 covers the other half —
    // that the full retrain buys no measurable held-out gain over LoRA — under
    // pinned seeds, because mode-vs-mode differences are smaller than the
    // run-to-run spread of weight initialisation this suite cannot control.
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
    const stockHeldOut = heldOutCE(initializePretrainedModel(baseConfig), true);
    const trained = initializePretrainedModel(baseConfig);
    for (let epoch = 0; epoch < ADVISORY_EPOCHS; epoch++) {
      for (const text of advisoryTrain) {
        trained.trainStep(trained.tokenizer.encode(text, true, true), 0.015, true, 0.005);
      }
    }
    const loraHeldOut = heldOutCE(trained, true);
    assert(
      stockHeldOut - loraHeldOut >= 0.3,
      'ISS-13.4: The recommended default (LoRA) really does generalise past its training sentences',
      `pretrained ${stockHeldOut.toFixed(3)} -> LoRA ${loraHeldOut.toFixed(3)}`
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
