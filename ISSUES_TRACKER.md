# Local SLM TypeScript Studio — Issues Tracker

This document tracks all identified engine, state, and UI issues, their fix implementation status, and corresponding automated tests.

---

## Issue Status Overview

| ID | Issue Description | Component / File | Status | Test Coverage |
|---|---|---|---|---|
| **ISS-01** | Triplicate Generation Logic & Hyperparameter Drift | `src/slm/transformer.ts` |  Completed | `scripts/test_fixes.ts` (ISS-01.1–4) |
| **ISS-02** | Fixed Static Context Window Truncation without Graceful Sliding | `src/slm/transformer.ts` |  Completed | `scripts/test_fixes.ts` (ISS-02.1–3) |
| **ISS-03** | Linear Search in High-Frequency Generation Loops | `src/slm/transformer.ts`, `matrix.ts` |  Completed | `scripts/test_fixes.ts` (ISS-03.1–2) |
| **ISS-04** | Base Model Comparison Cache Invalidation in ChatPlayground | `src/components/ChatPlayground.tsx` |  Completed | `scripts/test_fixes.ts` (ISS-04.1–3) |
| **ISS-05** | Vocabulary Growth & Out-of-Bounds Dimension Safety | `src/slm/transformer.ts`, `predefinedModels.ts` |  Completed | `scripts/test_fixes.ts` (ISS-05.1–2) |
| **ISS-06** | Training Loop Responsiveness & Micro-Batching in FineTuningStudio | `src/components/FineTuningStudio.tsx` |  Completed | `scripts/test_fixes.ts` (ISS-06.1) |
| **ISS-07** | Attention Map Visualization Resolution on Mobile Devices | `src/components/ArchitectureInspector.tsx` |  Completed | `scripts/test_fixes.ts` (ISS-07.1–3) |
| **ISS-08** | Token Inspector Modal Probabilities Summation Rounding | `src/components/TokenInspectorModal.tsx` |  Completed | `scripts/test_fixes.ts` (ISS-08.1–2) |
| **ISS-09** | LoRA Forward Pass Quadratic Redundant Matrix Multiplications | `src/slm/transformer.ts` | ✅ Completed | `scripts/test_fixes.ts` (ISS-09.1–2) |
| **ISS-10** | Sampling Candidate Probability Normalization & Nucleus Inflation Contradiction | `src/slm/matrix.ts`, `src/slm/transformer.ts` | ✅ Completed | `scripts/test_fixes.ts` (ISS-10.1–8) |
| **ISS-11** | Full Fine-Tuning Detection Failure in `isFineTuned()` Status Check | `src/slm/transformer.ts` | ✅ Completed | `scripts/test_fixes.ts` (ISS-11.1–8) |
| **ISS-12** | Fine-Tuning Evaluation Mode & LoRA Flag Mismatch | `src/components/FineTuningStudio.tsx` | ✅ Completed | `scripts/test_fixes.ts` (ISS-12.1–6) |
| **ISS-13** | Unstated Full-Retrain Cost: No Guardrail on the Training Regime | `src/components/FineTuningStudio.tsx` | ✅ Completed | `scripts/test_fixes.ts` (ISS-13.1–5), `scripts/diag_script.ts` §12 |
| **ISS-14** | Neural Body Never Trained: Only `lm_head` & LoRA Q/V Adapters Receive Gradients | `src/slm/transformer.ts`, `src/slm/predefinedModels.ts` | 🔍 Open / Identified | Registered — needs test & fix |
| **ISS-15** | "Full Fine-Tuning" Updates Only `lm_head` While UI Claims It Rewrites Every Weight | `src/slm/transformer.ts`, `src/components/FineTuningStudio.tsx` | 🔍 Open / Identified | Registered — needs test & fix |
| **ISS-16** | Hard-Coded `neuralMix = 0.08` Caps Neural Influence; Memory Overrides Dominate Generation | `src/slm/transformer.ts`, `src/slm/ngram.ts` | 🔍 Open / Identified | Registered — needs test & fix |
| **ISS-17** | Memory Observes Training Tokens Before the Loss Loop: Reported Loss Measures N-Gram Memorization | `src/slm/transformer.ts` (`trainStep`) | 🔍 Open / Identified | Registered — needs test & fix |
| **ISS-18** | Full-Mode Gradient Is Not the Gradient of the Blended Loss (`mixed − onehot`) | `src/slm/transformer.ts` (`trainStep`) | 🔍 Open / Identified | Registered — needs test & fix |
| **ISS-19** | LoRA Mode Reports Blended Loss but Optimizes Pure Neural CE (Metric ≠ Objective) | `src/slm/transformer.ts` | 🔍 Open / Identified | Registered — needs test & fix |
| **ISS-20** | Retrieval Shortcuts Mask Learning; Studio Reports Only In-Sample Loss | `src/slm/transformer.ts`, `src/slm/ngram.ts`, `src/components/FineTuningStudio.tsx` | 🔍 Open / Identified | Registered — needs test & fix |
| **ISS-21** | Vocabulary Growth Rebuilds the Model With Fresh Random Weights, Discarding Training Progress | `src/App.tsx` | 🔍 Open / Identified | Registered — needs test & fix |
| **ISS-22** | Training Data Scale Too Small for Grammar Induction; No In-App Held-Out Evaluation | `src/slm/corpus.ts`, `src/slm/datasets.ts` | 🔍 Open / Identified | Registered — needs test & fix |
| **ISS-23** | Frozen Random Representations: Untrained OOV Embeddings & Sliding-Window Position Reset | `src/slm/transformer.ts`, `src/slm/tokenizer.ts` | 🔍 Open / Identified | Registered — needs test & fix |

> **ISS-14 – ISS-23** were registered from a code review of the learning pipeline. They are engine-level defects that currently **prevent or mask** the neural transformer from learning English. They are tracked here only — **no fixes or tests have been implemented yet**.

---

## Detailed Issue Tracking

### ISS-01: Triplicate Generation Logic & Hyperparameter Drift
- **Status**:  Completed
- **Location**: `src/slm/transformer.ts` (`generate`, `generateStream`, `generateChatStream`, `encodeForGeneration`, `replyOpeningContext`)
- **Root Cause**: Sampling logic and prompt conditioning were pushing user prompt tokens into the assistant turn, causing punctuation leakage (`?`) and prompt echoing. Additionally, three generation methods had disparate loop bounds.
- **Resolution**:
  - Filtered punctuation tokens from `replyOpeningContext`.
  - Removed token pushing into sequence in `encodeForGeneration`, properly conditioning memory solely via `pendingReplyWord`.
  - Unified loop progression across `generate`, `generateStream`, and `generateChatStream` using `generateNextToken`.
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests ISS-01.1 through ISS-01.4) and `scripts/diagnose-slm.ts`.

---

### ISS-02: Fixed Static Context Window Truncation without Graceful Sliding
- **Status**:  Completed
- **Location**: `src/slm/transformer.ts` (`forward`, `generateStream`, `generateChatStream`, `generate`)
- **Root Cause**: Sequences exceeding `maxSeqLen` caused hard stops or potential positional embedding buffer overflow.
- **Resolution**:
  - Implemented automatic FIFO sliding context window in `forward`: evaluates on the most recent `maxSeqLen` tokens when sequence length exceeds `maxSeqLen`.
  - Removed arbitrary `tokens.length >= maxSeqLen` break clauses from generation streams so long conversations generate gracefully with sliding context.
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests ISS-02.1 through ISS-02.3).

---

### ISS-03: Linear Search in High-Frequency Generation Loops
- **Status**:  Completed
- **Location**: `src/slm/matrix.ts` (`sampleFromDistribution`), `src/slm/transformer.ts` (`generateNextToken`)
- **Root Cause**: Repeated `Array.prototype.includes` lookups on history arrays and structural token checks during per-token loops.
- **Resolution**:
  - Converted `historyTokens` check in `sampleFromDistribution` to $O(1)$ `Set.has()` lookup.
  - Used `Set` lookups for structural tokens (`PAD_ID`, `UNK_ID`, `BOS_ID`, `USER_ID`, `ASSISTANT_ID`, `NEWLINE_ID`) in `generateNextToken`.
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests ISS-03.1 and ISS-03.2).

---

### ISS-04: Base Model Comparison Cache Invalidation in ChatPlayground
- **Status**:  Completed
- **Location**: `src/components/ChatPlayground.tsx` (`getBaseModel`), `src/slm/predefinedModels.ts`
- **Root Cause**: Base model cache key was static `${model.config.id}:${model.config.vocabSize}` and did not invalidate when dynamic vocabulary grew via `defaultTokenizer.learnWords`.
- **Resolution**:
  - Appended `defaultTokenizer.vocabSize` to `getBaseModel` cache key.
  - Ensured `initializePretrainedModel` always creates configurations synced with `defaultTokenizer.vocabSize`.
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests ISS-04.1 through ISS-04.3).

---

### ISS-05: Vocabulary Growth & Out-of-Bounds Dimension Safety
- **Status**:  Completed
- **Location**: `src/slm/transformer.ts` (`forward`, `backwardLora`, `trainStep`), `src/slm/tokenizer.ts`
- **Root Cause**: Token IDs outside $[0, \text{vocabSize} - 1]$ or $< 0$ caused invalid offset calculations in typed arrays. Also `<unk>` was missing from the tokenizer splitting regex.
- **Resolution**:
  - Added `<unk>` to the tokenizer regex in `tokenizer.ts`.
  - Added safety clamping in `forward`, `backwardLora`, and `trainStep` mapping any invalid, negative, or $\ge \text{vocabSize}$ tokens to `UNK_ID`.
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests ISS-05.1 and ISS-05.2).

---

### ISS-06: Training Loop Responsiveness & Micro-Batching in FineTuningStudio
- **Status**:  Completed
- **Location**: `src/components/FineTuningStudio.tsx`
- **Root Cause**: Cancelling or stopping training early still triggered `onTrainingComplete(selectedPreset.name)`. In addition, long epoch steps blocked main thread animations.
- **Resolution**:
  - Added `completedSuccessfully = isTrainingRef.current && stepCount === totalSteps` guard before firing `onTrainingComplete`.
  - Standardized micro-batch yields every 2 steps to ensure UI responsiveness.
- **Verification**: Verified via `scripts/test_fixes.ts` (Test ISS-06.1).

---

### ISS-07: Attention Map Visualization Resolution on Mobile Devices
- **Status**:  Completed
- **Location**: `src/components/ArchitectureInspector.tsx`
- **Root Cause**: Attention heatmap cells lacked touch-scrolling constraints on mobile viewports, and weight export was omitting layer normalization weights and biases.
- **Resolution**:
  - Added `touch-pan-x`, `min-w-max`, and responsive cell sizing (`w-8 sm:w-10 h-7 sm:h-8`) to heatmap cells.
  - Included `ln_f_gamma`, `ln_f_beta`, `ln1_gamma`, `ln1_beta`, `fc1_b`, `fc2_b`, `ln2_gamma`, `ln2_beta` in `handleExportWeights`.
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests ISS-07.1 through ISS-07.3).

---

### ISS-08: Token Inspector Modal Probabilities Summation Rounding
- **Status**:  Completed
- **Location**: `src/components/TokenInspectorModal.tsx`
- **Root Cause**: Top candidate probabilities didn't show the remaining distribution mass, confusing users when top candidates summed to $< 100\%$.
- **Resolution**:
  - Added automatic calculation and visual display of remaining tail distribution mass (`1.0 - sum(topCandidates)`).
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests ISS-08.1 and ISS-08.2).

---

### ISS-09: LoRA Forward Pass Quadratic Redundant Matrix Multiplications
- **Status**: ✅ Completed
- **Location**: `src/slm/transformer.ts` (`forward` method, lines 1076–1090)
- **Root Cause**: Inside `forward()`, the projection loop computes $u = A \cdot x$ inside the `row` loop across all $dModel$ rows. Because $A \cdot x$ depends only on the rank $r$ and input $x$, recomputing it for each row results in $O(\text{seqLen} \cdot d^2 \cdot r)$ operations instead of $O(\text{seqLen} \cdot d \cdot r)$, creating a 32x redundant computational overhead per token during chat generation and training.
- **Resolution**:
  - Added rank-sized scratch buffers (`scratchLoraQ`, `scratchLoraV`) to `SmallLanguageModel`, sized from `loraRank` and reported in the memory stats.
  - `forward()` now computes the LoRA down-projection $u_Q = A_Q x$ / $u_V = A_V x$ once per token position, then evaluates $B u$ inside the output-row loop, dropping the down-projection work from $O(\text{seqLen} \cdot d^2 \cdot r)$ to $O(\text{seqLen} \cdot d \cdot r)$.
  - Arithmetic and accumulation order are unchanged, so LoRA-augmented logits stay numerically identical to the previous implementation.
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests ISS-09.1 and ISS-09.2). The second test instruments `lora_q_A` with a read-counting proxy: exactly `seqLen · r · dModel` element reads instead of the previous `seqLen · dModel · r · dModel` (a 48x reduction per layer for Assistant-48).

---

### ISS-10: Sampling Candidate Probability Normalization & Nucleus Inflation Contradiction
- **Status**: ✅ Completed
- **Location**: `src/slm/matrix.ts` (`sampleFromDistribution`), `src/slm/transformer.ts` (`generateNextToken`)
- **Root Cause**: In `sampleFromDistribution`, candidate probabilities returned to the UI inspector are mapped with `c.prob / sumP`, where `sumP` is the cumulative sum of the top-P filtered subset. When top-P is < 1.0, this inflates candidate probabilities above their actual distribution value, causing top candidates to sum to > 100% and directly contradicting the uninflated `chosenProb` displayed in the Token Inspector modal.
- **Resolution**:
  - `sampleFromDistribution` now returns a `SamplingResult` carrying `totalMass`: the mass of the whole repetition-adjusted distribution.
  - Candidate probabilities are normalized by `totalMass` instead of the top-P-filtered `sumP`. The nucleus filter now only decides *where* sampling may draw from and can no longer inflate the reported probabilities; the zero-mass early return reports the same distribution-space candidates.
  - `tokenInfo.prob` (the sampled token's own likelihood) and the modal's tail-mass indicator now reconcile with the candidate list.
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests ISS-10.1 through ISS-10.8). Before the fix, a 0.5 nucleus over a five-token distribution reported candidates summing to 200% with the best token at 100%; it now reports 100% total with that token at its true 50%.

---

### ISS-11: Full Fine-Tuning Detection Failure in `isFineTuned()` Status Check
- **Status**: ✅ Completed
- **Location**: `src/slm/transformer.ts` (`isFineTuned`, `resetToBase`, `trainStep`)
- **Root Cause**: `model.isFineTuned()` only checks whether adapter weights in `lora_q_B` or `lora_v_B` have non-zero values. When a user runs full fine-tuning (`loraMode: false`), gradient descent updates `lm_head` while leaving LoRA matrices at zero. Consequently, `isFineTuned()` incorrectly returns `false`, preventing UI indicators, comparison modes, and architecture views from recognizing that the model has been fine-tuned.
- **Resolution**:
  - Added an explicit `fullFineTuneApplied` flag to `SmallLanguageModel`, set by `trainStep` whenever training runs in full mode (`loraMode: false`).
  - `isFineTuned()` now returns `true` when the flag is set **or** any LoRA `lora_q_B` / `lora_v_B` weight moved off zero, so both adaptation modes are detected.
  - `saveBaseSnapshot()` clears the flag (the snapshot *is* the base model, so the pre-training warm-up counts as base rather than as user fine-tuning) and `resetToBase()` clears it too, keeping the header badge, comparison mode and architecture view consistent after a reset.
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests ISS-11.1 through ISS-11.8): full-mode training moves `lm_head` while leaving the adapters at zero and is now detected; reset clears both modes; LoRA detection is unchanged; a freshly pre-trained model still reports base pretrained.

---

### ISS-12: Fine-Tuning Evaluation Mode & LoRA Flag Mismatch
- **Status**: ✅ Completed
- **Location**: `src/components/FineTuningStudio.tsx` (`runDatasetEvaluation`)
- **Root Cause**: In `runDatasetEvaluation`, `model.forward(tokens, true)` hardcodes `useLora = true`. When the user trains with `hyperparams.loraMode = false` (full fine-tuning), training steps execute with `useLora = false`, but dataset evaluation evaluates with `useLora = true`.
- **Resolution**:
  - Extracted the per-turn scoring into an exported `evaluateTurn(model, turn, useLora)` helper in `FineTuningStudio.tsx`, which threads `useLora` through both `model.generate(...)` and `model.forward(tokens, useLora)`.
  - `runDatasetEvaluation` now calls `evaluateTurn(model, turn, hyperparams.loraMode)` instead of hardcoding `model.forward(tokens, true)`, so evaluation strictly mirrors the active training mode.
  - The per-epoch sample completions and the quick single-prompt tester also generate with `hyperparams.loraMode`, so every readout in the studio reflects the adaptation mode that is currently active.
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests ISS-12.1 through ISS-12.6), including a check that the evaluated loss matches an independently computed cross-entropy in each mode, that the two modes genuinely differ once stale adapters are present, and a source-level guard that no hardcoded `model.forward(tokens, true)` remains in the evaluation path.

---

### ISS-13: Unstated Full-Retrain Cost — No Guardrail on the Training Regime
- **Status**: ✅ Completed
- **Location**: `src/components/FineTuningStudio.tsx` (`trainingRegimeAdvisory`, `FULL_MODE_ADVISORY_EPOCHS`, hyperparameter panel), `scripts/diag_script.ts` section 12
- **Root Cause**: The studio offers two adaptation modes and defaults to neither loudly: `loraMode` defaults to `true`, but nothing in the UI says why, and the epochs slider invites a 10–25 epoch *full retrain* — a backprop pass over every weight in the network. The two modes had never been compared on data held out of training, so a user pushing epochs in full mode had no way to know they were buying nothing: the run only reported in-sample loss, which improves in both modes.
- **Resolution**:
  - Measured both regimes for the studio's own hyperparameters on `BASE_CORPUS` sentences held out of fine-tuning entirely, under multiple pinned seeds (`diag_script.ts` section 12). Both modes cut held-out cross-entropy by well over a nat, but the *mode* does not decide the outcome: the per-seed winner flips (per-seed LoRA-vs-full differences span -0.29 to +0.64 nats), and the two means land within a tenth of a nat of each other at the studio default of 10 epochs. Only the average supports a claim here, which is why the section measures more than one seed.
  - Added `FULL_MODE_ADVISORY_EPOCHS = 4` and an exported `trainingRegimeAdvisory(hyperparams)` that returns an advisory string for a long full retrain (`!loraMode && epochs >= 4`) and `null` otherwise. It is returned as data, not rendered text, so it is unit-testable.
  - The hyperparameter panel now renders that advisory as an amber inline note under the adaptation-mode selector, so the cost of a long full retrain is visible before training starts rather than only as a slower run.
- **Verification**: `scripts/test_fixes.ts` (Tests ISS-13.1–13.5): the default LoRA configuration raises no advisory, a default-length full retrain is flagged, a 1–3 epoch full retrain is not, the recommended default genuinely lowers held-out loss (10 epochs, held-out corpus, threshold 0.3 nats above the pretrained model), and `FineTuningStudio` actually renders the advisory. `scripts/diag_script.ts` section 12 adds the seeded multi-seed measurement behind the threshold. Suite totals: 52 passed, 0 failed; `diag_script.ts`: 46 passed, 0 failed; `bun tsc -b --noEmit`: clean.

---

### ISS-14: Neural Body Never Trained — Embeddings, Projections & FFN Stay at Random Initialization
- **Status**: 🔍 Open / Identified (registered — not yet fixed)
- **Location**: `src/slm/transformer.ts` (`trainStep`, `backwardLora`, `applyAdapterUpdate`), `src/slm/predefinedModels.ts` (`initializePretrainedModel`)
- **Root Cause**: The only gradient updates in the entire engine are the four LoRA adapter matrices (`transformer.ts` lines 960–963) and, in full mode, `lm_head` (line 1330). `wte`, `wpe`, `q_proj`/`k_proj`/`v_proj`/`out_proj`, `fc1`/`fc2`, all LayerNorms and biases are **never updated anywhere** — they remain at their random `std = 0.03` initialization forever. Worse, the pre-training warm-up runs `model.trainStep(tokens, 0.08, false, 0.001)` (`predefinedModels.ts:128`), i.e. `loraMode = false`, so even "pre-training" only trains `lm_head`. The base network is therefore a random feature extractor with a tuned output layer, and no later fine-tuning can induce grammar through it.
- **Impact**: The neural pathway cannot learn English structure — representations feeding the LM head are noise, which is the root enabler of the memory-dominant design (ISS-16).
- **Proposed Fix Direction**: Backpropagate through the full network (embeddings, projections, FFN, norms) in full mode, and warm up with that path (or LoRA over all projections) instead of `lm_head`-only.
- **Verification**: Not covered by `scripts/test_fixes.ts` or `scripts/diag_script.ts` today; a fix should assert that base weights change during warm-up/full training.

---

### ISS-15: "Full Fine-Tuning" Updates Only `lm_head` While UI Claims It Rewrites Every Weight
- **Status**: 🔍 Open / Identified (registered — not yet fixed)
- **Location**: `src/slm/transformer.ts` (`trainStep`, lines 1319–1330), `src/components/FineTuningStudio.tsx` (`trainingRegimeAdvisory`, line 140, adaptation-mode selector)
- **Root Cause**: The mode is labelled "Full Fine-Tuning" and the amber advisory literally says a full retrain "rewrites every weight in the network" — but the full-mode branch updates only `lm_head` rows (`vocabSize × dModel`), never the transformer body. `countParameters(loraMode = false)` correctly reports `trainable = lmHeadParams`, contradicting the copy next to it.
- **Impact**: Users choosing full mode believe they are retraining the whole model; measured "no gain over LoRA" (ISS-13) is partially an artifact of there being almost nothing to train.
- **Proposed Fix Direction**: Either implement true full backprop (see ISS-14) or rename/correct the mode and advisory to state that only the LM head is trained.
- **Verification**: Not yet covered; a fix should assert which weight tensors move per mode and that UI copy matches measured behavior.

---

### ISS-16: Hard-Coded `neuralMix = 0.08` Caps Neural Influence; Memory Overrides Dominate Generation
- **Status**: 🔍 Open / Identified (registered — not yet fixed)
- **Location**: `src/slm/transformer.ts` (`neuralMix`, line 181; mix sites lines 1307 and 1409; opener override lines 1386–1395; case forcing line 1425), `src/slm/ngram.ts` (`SENTENCE_END_STOP_MASS = 6`)
- **Root Cause**: Every generated token is `0.08 × neural + 0.92 × memory`. On top of that: the reply's first token is overridden 90% by the memory reply-link distribution (70% for the generic opener), trained prompts are forced to their case token with `p = 0.88 + 0.12·p`, and after any observed sentence ender the memory adds 6 mass units to EOS (~86% stop probability). The neural pathway can steer at most ~8% of sampling mass, and less at reply openings and during case replay.
- **Impact**: Whatever the network learns is nearly invisible in output; all observed fluency comes from the trigram table. The model structurally cannot "demonstrate" learning English.
- **Proposed Fix Direction**: Make the mix configurable/observable, ramp `neuralMix` up as training progresses (or distill memory into weights), and reduce hard overrides to guardrails only.
- **Verification**: Not yet covered; a test could train the network until it disagrees with the memory and assert its influence on the final distribution.

---

### ISS-17: Memory Observes Training Tokens Before the Loss Loop — Reported Loss Measures N-Gram Memorization
- **Status**: 🔍 Open / Identified (registered — not yet fixed)
- **Location**: `src/slm/transformer.ts` (`trainStep`: `this.memory.observe(tokens.slice(0, seqLen), 2.0)` at line 1267, executed before the loss loop around lines 1276–1335)
- **Root Cause**: Each training step first feeds the exact sequence to the n-gram memory at weight 2.0, *then* computes loss on the neural/memory blend for that same sequence. The memory has just memorized the trigram contexts, so the target probability (and therefore the reported loss) is already low before any weight update runs. In full mode it additionally collapses the gradient, because `mixed[target]` is already large and `grad = mixed[v] − onehot` shrinks toward zero.
- **Impact**: The Fine-Tuning Studio's loss/perplexity curves chart the n-gram table being filled in, not gradient descent — "the model is learning" is an artifact of measurement order.
- **Proposed Fix Direction**: Compute the loss/gradient from the pre-observation distribution (or from neural CE only, see ISS-19), and observe the memory after the update; report blended loss separately as a memory metric.
- **Verification**: Not yet covered; a test could assert first-step loss is computed before `memory.size` grows for that sequence.

---

### ISS-18: Full-Mode Gradient Is Not the Gradient of the Blended Loss
- **Status**: 🔍 Open / Identified (registered — not yet fixed)
- **Location**: `src/slm/transformer.ts` (`trainStep`, lines 1319–1331: `const grad = mixed[v] - (v === targetToken ? 1.0 : 0.0)` and the `Math.abs(grad) < 0.004` skip at line 1325)
- **Root Cause**: The loss is $-\log P_{\text{target}}$ where $P = \text{mix}\cdot p_{\text{neural}} + (1-\text{mix})\cdot m$ and $m$ (memory) is constant w.r.t. the weights. The true logit gradient is $\frac{\text{mix}\cdot p_t}{P_t}(p_v - \delta_{tv})$, but the code uses $P_v - \delta_{tv}$ — treating the blended distribution as if it were the model's own softmax. With 92% of the blend being constant memory mass, the update mostly follows the memory's shape rather than the data. The `|grad| < 0.004` threshold additionally discards small but meaningful gradients.
- **Impact**: "Full" fine-tuning moves `lm_head` in a biased direction and under-learns; combined with ISS-17 the effective signal is near zero on anything the memory already covers.
- **Proposed Fix Direction**: Derive `dL/dz` from the neural softmax scaled by `mix·p/P_target`, and drop or justify the gradient threshold.
- **Verification**: Not yet covered; a numerical-gradient check (finite differences) on a tiny config would catch this.

---

### ISS-19: LoRA Mode Reports Blended Loss but Optimizes Pure Neural CE (Metric ≠ Objective)
- **Status**: 🔍 Open / Identified (registered — not yet fixed)
- **Location**: `src/slm/transformer.ts` (`trainStep` loss loop lines 1307–1315 vs. `backwardLora` cross-entropy lines 746–760)
- **Root Cause**: The loss/perplexity returned by `trainStep` is computed on the **blended** distribution, while `backwardLora` differentiates the **pure neural** cross-entropy of `softmax(logits)`. The two numbers can move in opposite directions: the reported loss can fall because the memory improved while the adapters got worse, and vice versa.
- **Impact**: Training curves, early stopping decisions and user judgments of "did it learn English?" are decoupled from what the optimizer actually minimizes.
- **Proposed Fix Direction**: Report both numbers (neural CE and blended NLL) with labels, and chart the one that matches the active mode's objective.
- **Verification**: Not yet covered; a test could assert the reported LoRA-mode loss equals an independently computed neural CE (it would currently fail).

---

### ISS-20: Retrieval Shortcuts Mask Whether the Network Learned; Studio Reports Only In-Sample Loss
- **Status**: 🔍 Open / Identified (registered — not yet fixed)
- **Location**: `src/slm/transformer.ts` (case replay line 1425, opener links lines 1386–1395), `src/slm/ngram.ts` (`findCase`, line 147, fuzzy `minScore = 0.65`), `src/components/FineTuningStudio.tsx` (`runDatasetEvaluation`)
- **Root Cause**: Trained prompts are answered from lookup tables: `findCase` returns an exact or ≥65%-overlap stored reply, generation then pins it with 0.88 mass, and the opener link pins the first token with 0.9 mass. The studio's evaluation is **in-sample** (the same turns just trained on, scored on the blended distribution); the only held-out measurement in the repo lives in `scripts/diag_script.ts` §12.
- **Impact**: Benchmark checks (overlap ≥ 50%) pass whether or not any weight learned anything; grammar generalization to unseen sentences is never measured in the app.
- **Proposed Fix Direction**: Add held-out evaluation (split the dataset, score unseen turns), and report case-replay usage so users can see when an answer came from memory rather than the network.
- **Verification**: Not yet covered in-app; `diag_script.ts` §8/§12 show the pattern to reuse.

---

### ISS-21: Vocabulary Growth Rebuilds the Model With Fresh Random Weights, Discarding Training Progress
- **Status**: 🔍 Open / Identified (registered — not yet fixed)
- **Location**: `src/App.tsx` (`handleDatasetsChange`, lines 64–79), `src/slm/predefinedModels.ts` (`initializePretrainedModel`)
- **Root Cause**: When a dataset edit/import teaches the tokenizer at least one new word, `handleDatasetsChange` calls `initializePretrainedModel` — constructing a brand-new model with **unseeded `Math.random` weights** — and `syncResetState()` clears the fine-tuned badge. All previously trained adapters, `lm_head` updates and the base snapshot are silently lost, and the rebuilt model differs run-to-run.
- **Impact**: Fine-tuning cannot survive adding a single turn containing an unseen word; repeated edits produce a different "base" model each time, so learning never accumulates.
- **Proposed Fix Direction**: Resize/pad `wte`/`lm_head` in place while preserving existing weights (and the snapshot), or warn before rebuilding.
- **Verification**: Not yet covered; a test could train, add a new-word turn, and assert the adapters/`lm_head` survive.

---

### ISS-22: Training Data Scale Too Small for Grammar Induction; No In-App Held-Out Evaluation
- **Status**: 🔍 Open / Identified (registered — not yet fixed)
- **Location**: `src/slm/corpus.ts` (`BASE_CORPUS`, 41 sentences), `src/slm/predefinedModels.ts` (`PRETRAIN_CORPUS`, 7 dialogues), `src/slm/datasets.ts` (standard presets ~6–9 turns each; `generateExpandedChatCorpus`), `src/components/FineTuningStudio.tsx` (`activeTurns`)
- **Root Cause**: The network (even once ISS-14 is fixed) would see a few hundred short template sentences in total; the "Expanded"/"Large" scales are mechanical recombinations of the same turns rather than new English. Nothing in the app measures grammar generalization — only in-sample loss — so the project cannot distinguish memorization from learning.
- **Impact**: Open-domain English cannot be induced at this scale; the memory layer's exact recall (ISS-20) hides the shortfall.
- **Proposed Fix Direction**: Import a larger real corpus, add a held-out split with per-epoch held-out perplexity, and treat generalization — not in-sample loss — as the learning signal.
- **Verification**: Not yet covered in-app; reuse the held-out methodology of `diag_script.ts` §12.

---

### ISS-23: Frozen Random Representations — Untrained OOV Embeddings & Sliding-Window Position Reset
- **Status**: 🔍 Open / Identified (registered — not yet fixed)
- **Location**: `src/slm/transformer.ts` (`forward`: tail-slice + `wpe` indexed from `i = 0`), `src/slm/tokenizer.ts` (`bpeSplit`/`wordToIds`), `src/slm/transformer.ts` (`initWeights`, `std = 0.03`)
- **Root Cause**: (a) BPE lets unseen words decompose into subword tokens, but their embedding rows live in `wte`, which never trains (ISS-14) — so "reading" an unseen word amounts to feeding frozen random vectors into a frozen random body. (b) When a sequence exceeds `maxSeqLen`, `forward` keeps the tail but indexes `wpe` from 0, restarting absolute positions — every sliding-window context is mis-positioned, and `wpe` itself is never trained either.
- **Impact**: Subword compositionality (a headline tokenizer feature) is unusable by the neural path, and long multi-turn chats are conditioned on misaligned positions.
- **Proposed Fix Direction**: Train `wte`/`wpe` (follows from ISS-14); offset `wpe` by the number of dropped tokens (or use relative positions).
- **Verification**: Not yet covered; a test could assert `wpe` responds to a token's true index after truncation, and that subword embeddings move during training.

---

### DS-01: Standalone Capability Turn ("what can you do") Missing from Predefined Dataset
- **Status**: ✅ Completed
- **Location**: `src/slm/datasets.ts` (Helpful Daily Assistant preset), `scripts/test_fixes.ts`
- **Observation**: The capability prompt `what can you do` was only present in the pre-training warm-up corpus (`src/slm/predefinedModels.ts`), so the base model answered it, but fine-tuning on any predefined dataset did not reinforce the reply — after a reset-to-base the exact-recall path relied solely on warm-up memory.
- **Resolution**:
  - Added a dedicated turn `ha-9` to the Helpful Daily Assistant preset: user `what can you do` → assistant `I can converse with you , share ideas , and be fine tuned on custom chat datasets .` (matching the warm-up reply verbatim, so warm-up memory and fine-tuned weights reinforce each other).
  - All four open issues plus this dataset addition remain fully covered by the test suite.
- **Verification**: Verified via `scripts/test_fixes.ts` (Tests DS-1 and DS-2): DS-1 asserts the standalone turn exists exactly once in the preset; DS-2 fine-tunes a fresh model on that turn (6 LoRA epochs) and confirms greedy generation for the bare prompt reproduces the capability answer (≥ 6 of 8 key phrases). Full suite: 47 passed, 0 failed; `bun scripts/diag_script.ts`: 33 passed, 0 failed; `bun tsc -b --noEmit`: clean.

