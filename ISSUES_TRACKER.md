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
| **ISS-09** | LoRA Forward Pass Quadratic Redundant Matrix Multiplications | `src/slm/transformer.ts` | 🔍 Open / Identified | Needs Test & Fix |
| **ISS-10** | Sampling Candidate Probability Normalization & Nucleus Inflation Contradiction | `src/slm/matrix.ts`, `src/slm/transformer.ts` | 🔍 Open / Identified | Needs Test & Fix |
| **ISS-11** | Full Fine-Tuning Detection Failure in `isFineTuned()` Status Check | `src/slm/transformer.ts` | 🔍 Open / Identified | Needs Test & Fix |
| **ISS-12** | Fine-Tuning Evaluation Mode & LoRA Flag Mismatch | `src/components/FineTuningStudio.tsx` | 🔍 Open / Identified | Needs Test & Fix |

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
- **Status**: 🔍 Open / Identified
- **Location**: `src/slm/transformer.ts` (`forward` method, lines 1076–1090)
- **Root Cause**: Inside `forward()`, the projection loop computes $u = A \cdot x$ inside the `row` loop across all $dModel$ rows. Because $A \cdot x$ depends only on the rank $r$ and input $x$, recomputing it for each row results in $O(\text{seqLen} \cdot d^2 \cdot r)$ operations instead of $O(\text{seqLen} \cdot d \cdot r)$, creating a 32x redundant computational overhead per token during chat generation and training.
- **Proposed Fix**: Precompute $u_Q = A_Q \cdot x$ and $u_V = A_V \cdot x$ into temporary rank-sized vectors once per token position, and compute $B \cdot u$ inside the row loop or as a dedicated GEMV.

---

### ISS-10: Sampling Candidate Probability Normalization & Nucleus Inflation Contradiction
- **Status**: 🔍 Open / Identified
- **Location**: `src/slm/matrix.ts` (`sampleFromDistribution`), `src/slm/transformer.ts` (`generateNextToken`)
- **Root Cause**: In `sampleFromDistribution`, candidate probabilities returned to the UI inspector are mapped with `c.prob / sumP`, where `sumP` is the cumulative sum of the top-P filtered subset. When top-P is < 1.0, this inflates candidate probabilities above their actual distribution value, causing top candidates to sum to > 100% and directly contradicting the uninflated `chosenProb` displayed in the Token Inspector modal.
- **Proposed Fix**: Return the candidate probabilities normalized against the full distribution mass (or preserving distribution-space probabilities), ensuring consistency between `tokenInfo.prob`, candidate values, and the tail mass indicator.

---

### ISS-11: Full Fine-Tuning Detection Failure in `isFineTuned()` Status Check
- **Status**: 🔍 Open / Identified
- **Location**: `src/slm/transformer.ts` (`isFineTuned`, `resetToBase`, `trainStep`)
- **Root Cause**: `model.isFineTuned()` only checks whether adapter weights in `lora_q_B` or `lora_v_B` have non-zero values. When a user runs full fine-tuning (`loraMode: false`), gradient descent updates `lm_head` while leaving LoRA matrices at zero. Consequently, `isFineTuned()` incorrectly returns `false`, preventing UI indicators, comparison modes, and architecture views from recognizing that the model has been fine-tuned.
- **Proposed Fix**: Track explicit fine-tuned state or inspect both LoRA matrices and `lm_head` against the base snapshot so both LoRA and full fine-tuning modes are accurately reflected.

---

### ISS-12: Fine-Tuning Evaluation Mode & LoRA Flag Mismatch
- **Status**: 🔍 Open / Identified
- **Location**: `src/components/FineTuningStudio.tsx` (`runDatasetEvaluation`)
- **Root Cause**: In `runDatasetEvaluation`, `model.forward(tokens, true)` hardcodes `useLora = true`. When the user trains with `hyperparams.loraMode = false` (full fine-tuning), training steps execute with `useLora = false`, but dataset evaluation evaluates with `useLora = true`.
- **Proposed Fix**: Pass `hyperparams.loraMode` into `model.forward(tokens, hyperparams.loraMode)` so evaluation strictly mirrors the active training mode.

