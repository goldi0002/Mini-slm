# Local SLM TypeScript Studio — Project Context & Engineering Guide

## 1. Project Identity & Architecture
**Local SLM TypeScript Studio** is a pure client-side Small Language Model (SLM) running entirely in the browser using TypeScript, React 19, Tailwind CSS, and Vite. There are **no external ML dependencies** (no PyTorch, TensorFlow.js, ONNX, or WebGPU required); all tensor operations, forward passes, and backpropagation are implemented from scratch using native JavaScript typed arrays (`Float32Array`).

### Core Philosophy
A toy transformer (~50k–120k parameters) running in a browser tab cannot learn open-domain English grammar from scratch in seconds. This engine solves this through a **hybrid architecture**:
1. **Causal Transformer Neural Network (`src/slm/transformer.ts`)**: Learns contextual representations, attention weights across tokens, and task-specific patterns.
2. **Statistical Memory Layer (`src/slm/ngram.ts`)**: A smoothed trigram backoff language model tracking unigrams, bigrams, trigrams, utterance enders, and dialogue links.
3. **Logit Blending & Boundary Rules**: At inference time, neural logits and memory probabilities are blended, and a sentence-boundary stopping rule (`SENTENCE_END_STOP_MASS = 6`) guarantees complete, non-truncated, grammatically coherent replies without trailing stubs.

---

## 2. Directory & Module Structure

```
/
├── .claude/skills/ai-model-engineer/SKILL.md  # Comprehensive AI/ML engineering guidelines
├── scripts/
│   ├── diagnose-slm.ts                       # Quick sanity check for base & fine-tuned generation
│   ├── diag_script.ts                        # 800+ line end-to-end verification harness
│   ├── test_trained_data.ts                  # Multi-dataset benchmark runner
│   ├── debug-dist.ts                         # Token distribution debugging tool
│   └── diag_script.ts                        # Verification harness
├── src/
│   ├── slm/
│   │   ├── matrix.ts                         # Vector/matrix ops, GELU, LayerNorm, Softmax, Box-Muller
│   │   ├── corpus.ts                         # Dependency-free seed corpus + BPE seed words (breaks the tokenizer ↔ transformer import cycle)
│   │   ├── tokenizer.ts                      # Hybrid word-level + BPE subword tokenizer, special tokens, dynamic vocab expansion
│   │   ├── ngram.ts                          # Trigram model with backoff, dialogue pairing, boundary rules
│   │   ├── transformer.ts                    # SmallLanguageModel: attention, LoRA, forward, backward, generation
│   │   ├── predefinedModels.ts               # Model configurations (Assistant-48, NanoLM-Light) and warm-up
│   │   └── datasets.ts                       # Predefined dialogue datasets & corpus expansion generator
│   ├── components/
│   │   ├── Header.tsx                        # Navigation, model selector, memory & parameter metrics
│   │   ├── ChatPlayground.tsx                # Streaming chat, comparative base vs fine-tuned, sampling controls
│   │   ├── FineTuningStudio.tsx              # Interactive trainer, loss/perplexity charts, qualitative evaluation
│   │   ├── ArchitectureInspector.tsx         # Multi-head attention heatmaps, weight inspect, JSON export
│   │   ├── DatasetManager.tsx                # Custom turn editor, dataset import/export, tokenizer sync
│   │   └── TokenInspectorModal.tsx           # Per-token probability inspector modal
│   ├── types.ts                              # Core interfaces: ModelConfig, ChatMessage, LossPoint, etc.
│   ├── App.tsx                               # Root orchestration and dataset/model state management
│   ├── index.css                             # Tailwind CSS setup
│   └── main.tsx                              # React DOM mount point
├── metadata.json                             # Applet permissions and capabilities
├── package.json                              # Scripts (dev, build, lint) and dependencies
└── tsconfig.json                             # TypeScript compiler configuration
```

---

## 3. Mathematical & Engine Details

### Neural Transformer (`src/slm/transformer.ts`)
- **Layers**: Multi-Head Attention (Q, K, V projections + Output projection) + 2-layer MLP (`fc1` to `dFfn`, `fc2` to `dModel`) with GELU activation.
- **Normalization**: Pre-LayerNorm on attention block and FFN block, plus final LayerNorm before the LM head.
- **LoRA (Low-Rank Adaptation)**: Injected on query (`lora_q_A`, `lora_q_B`) and value (`lora_v_A`, `lora_v_B`) projections with scaling factor `loraAlpha / loraRank`. Full-model fine-tuning is also supported.
- **Zero-Heap Thrashing**: Reusable flat scratch buffers (`q_buf`, `k_buf`, `v_buf`, `attn_scores`, etc.) are allocated up-front. Resets use a flat snapshot buffer (`baseWeightsSnapshot`) without GC stalls.

### Hybrid Word-Level + BPE Subword Tokenizer (`src/slm/tokenizer.ts`)
The vocabulary is built in four deterministic stages, and only stages 3–4 ever append ids:
1. `CONVERSATIONAL_VOCAB` (~268 everyday words, punctuation, digits).
2. The single-character OOV fallback alphabet (lowercase, then uppercase letters that do not shadow a real word).
3. Every word of `BASE_CORPUS` as a whole-word token, so the fluent baseline never depends on subword composition.
4. BPE merge symbols learned from `BASE_CORPUS` + `BPE_SEED_WORDS` (≤ `MAX_BPE_MERGES = 256`).

Encoding rule: a word already in the vocabulary is **one token** (exactly as before, so dataset exact-recall, n-gram links and dialogue-case keys keep working); a word that is *not* in the vocabulary decomposes into learned subword pieces ("unbelievable" → `un` + `believable`, "improvisation" → `im` + `p` + `ro` + `v` + `is` + `ation`) instead of being spelled out one character at a time. Merge symbols are ordinary generatable tokens, not fallback chars, so `isFallbackCharToken()` still suppresses only genuine single-letter OOV fallbacks.

BPE details:
- `learnBpeMerges()` runs at construction, weights adjacent symbol pairs by word frequency, requires frequency ≥ 2, and breaks ties on the lexicographically smallest pair — the vocabulary is identical on every load.
- `bpeSplit()` applies the lowest-ranked (earliest-learned) merge present until no known pair remains; `wordToIds()` memoizes the result in `wordCache` (cleared past 8192 entries and invalidated by `addWord`, because a newly learned whole word must supersede a cached split).
- Id bookkeeping goes through the single private `addToken()` helper so `nextId`, `tokenToId`, `idToToken` and `vocabSize` can never drift apart.
- `corpus.ts` holds `BASE_CORPUS`/`BPE_SEED_WORDS` and imports nothing: the tokenizer learns merges from it, and importing the transformer there would create a tokenizer → transformer → tokenizer cycle.

### Dynamic Vocabulary Expansion
- Special tokens: `<pad>` (0), `<unk>` (1), `<bos>` (2), `<eos>` (3), `<user>` (4), `<assistant>` (5), `\n` (6).
- `defaultTokenizer.learnWords(words)`: When user edits or imports datasets in the Dataset Manager, new words are ingested into the vocabulary table. Because weight matrices and embedding tables depend on vocabulary size, adding new words rebuilds the model with the updated dimension.

### Statistical Memory Layer (`src/slm/ngram.ts`)
- Stores counts for:
  - `uni`: token frequencies
  - `bi`: bigram transitions ($w_t \mid w_{t-1}$)
  - `tri`: trigram transitions ($w_t \mid w_{t-2}, w_{t-1}$)
  - `links`: dialogue transitions from last user word to first assistant word
  - `openers`: global assistant utterance opening words
  - `enders`: tokens preceding `<eos>` to learn sentence boundary probabilities
  - `cases`: exact dataset utterance prompt-to-response mappings for precise recall
- `SENTENCE_END_STOP_MASS = 6`: Prevents truncated or run-on answers by boosting EOS probability once a natural sentence punctuation is generated.

---

## 4. Workflows & State Synchronization Rules

1. **Dataset & Tokenizer Sync**:
   - `datasets` state is owned in `src/App.tsx`.
   - Modifying or importing turns triggers `defaultTokenizer.learnWords(...)`. If new words are added, the model is re-initialized to match the expanded vocabulary table.
2. **Fine-Tuning State**:
   - Training can run on live datasets with LoRA or full weight fine-tuning. The studio defaults to LoRA (`loraMode: true`, 10 epochs, lr 0.015, wd 0.005) for a measured reason: on held-out sentences both modes cut loss by well over a nat, but the full retrain buys no gain at all over the cheaper adapter (see diag section 12).
   - `FULL_MODE_ADVISORY_EPOCHS` / `trainingRegimeAdvisory(hyperparams)` in `FineTuningStudio.tsx` drive the amber advisory shown under the adaptation-mode selector for a long full retrain (4+ epochs) — the threshold is a measurement, not taste, so re-measure before changing it.
   - `resetToBase()` resets both neural weights (restoring from typed array snapshot) and memory layer tables, cleanly clearing fine-tuned badges.
3. **Verification**:
   - Always run `npm run lint` (`tsc --noEmit`) to verify strict type correctness.
   - Verify builds via `compile_applet` before finishing tasks.
   - Diagnostic scripts like `scripts/diagnose-slm.ts` can verify generation and loss behavior.
   - `scripts/diag_script.ts` section 11 covers the BPE tokenizer (unseen words compose, known words stay whole tokens, special ids unchanged, id space round-trips).
   - `scripts/diag_script.ts` section 12 covers the training regimes: it re-pins `Math.random` per seed and trains both adaptation modes on sentences held out of training, because the per-seed winner flips and only an average supports the "LoRA is enough" claim behind the studio default. It is the slowest section (full backprop passes over every weight), so the whole harness takes roughly two minutes.
