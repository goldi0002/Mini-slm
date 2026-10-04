/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from 'react';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  CheckCircle2, 
  Sparkles, 
  LineChart, 
  Sliders, 
  TrendingDown, 
  HardDrive,
  Database,
  ArrowRight,
  Activity,
  Check,
  Send,
  HelpCircle,
  TriangleAlert
} from 'lucide-react';
import { 
  TrainingHyperparams, 
  LossPoint, 
  TrainingState,
  DatasetPreset
} from '../types';

export interface EvaluationResult {
  turnId: string;
  category: string;
  prompt: string;
  target: string;
  predicted: string;
  overlap: number;
  isPassed: boolean;
}

export const TARGET_LOSS = 0.3;

export interface EvaluationSummary {
  totalTurns: number;
  passedTurns: number;
  avgOverlap: number;
  avgLoss: number;
  results: EvaluationResult[];
  /** Turns held out of training and the cross-entropy the model reaches on them. */
  heldOutTurns: number;
  heldOutLoss: number;
  /** Share of generation the network earned on held-out text (0..1). */
  neuralInfluence: number;
  neuralPerplexity: number;
  memoryPerplexity: number;
}
import { SmallLanguageModel } from '../slm/transformer';
import { generateExpandedChatCorpus } from '../slm/datasets';
import { SPECIAL_TOKENS, ASSISTANT_ID } from '../slm/tokenizer';
import { ConversationTurn } from '../types';

/** The exact training text of a turn, shared by training, scoring and calibration. */
function formatTurn(turn: ConversationTurn): string {
  return `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
}

/**
 * Score one dataset turn against the model: the generated reply, its word
 * overlap with the ground-truth answer, and response-only cross-entropy loss.
 * Prompt/control tokens are excluded so the 0.3 training target measures the
 * actual assistant response the user is trying to teach.
 *
 * `useLora` must be the adaptation mode training ran with. Evaluating a full
 * retrain with the LoRA adapters enabled (or a LoRA run with them disabled)
 * measures a different network than the one that was just trained.
 */
export function evaluateTurn(
  model: SmallLanguageModel,
  turn: ConversationTurn,
  useLora: boolean
): { predicted: string; overlap: number; isPassed: boolean; loss: number } {
  const prompt = model.tokenizer.formatConversationPrompt(turn.user);
  const res = model.generate(
    prompt,
    {
      temperature: 0.2,
      topK: 10,
      topP: 0.9,
      repetitionPenalty: 1.1,
      maxNewTokens: 35,
    },
    useLora
  );

  // Word overlap against the target answer
  const targetWords = turn.assistant.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(Boolean);
  const predWords = res.text.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(Boolean);
  const targetSet = new Set(targetWords);
  let matchCount = 0;
  for (const w of predWords) {
    if (targetSet.has(w)) matchCount++;
  }
  const overlap = targetWords.length > 0 ? (matchCount / targetWords.length) * 100 : 0;

  // Cross-entropy loss for this turn, under the same adaptation mode
  const V = model.config.vocabSize;
  const formatted = `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
  const tokens = model.tokenizer.encode(formatted, true, true);
  const { logits, seqLen } = model.forward(tokens, useLora);
  let turnLoss = 0;
  let turnTokens = 0;
  const assistantIdx = tokens.lastIndexOf(ASSISTANT_ID);
  const lossStart = assistantIdx >= 0 ? assistantIdx + 1 : 0;
  for (let i = lossStart; i < seqLen - 1; i++) {
    const target = tokens[i + 1];
    if (target === 0) continue;
    const row = logits.subarray(i * V, (i + 1) * V);
    let maxLogit = -Infinity;
    for (let j = 0; j < V; j++) if (row[j] > maxLogit) maxLogit = row[j];
    let sumExp = 0;
    for (let j = 0; j < V; j++) sumExp += Math.exp(row[j] - maxLogit);
    const p = Math.max(1e-7, Math.exp(row[target] - maxLogit) / sumExp);
    turnLoss += -Math.log(p);
    turnTokens++;
  }

  return {
    predicted: res.text,
    overlap,
    isPassed: overlap >= 40,
    loss: turnTokens > 0 ? turnLoss / turnTokens : 0,
  };
}

/**
 * Full-mode epoch count from which the studio flags the run as a long full
 * retrain — the point where the extra epochs stop paying for themselves.
 *
 * The threshold comes from measurement, not taste: `scripts/diag_script.ts`
 * section 12 scores both adaptation modes on sentences held out of training.
 * A full retrain does reach lower held-out loss than the adapters (it updates
 * every weight, not a rank-r update inside the attention projections), but the
 * gap saturates after a few epochs while each additional epoch costs a full
 * backprop pass over the entire network and refits the turns already seen.
 */
export const FULL_MODE_ADVISORY_EPOCHS = 4;

/**
 * Advisory for a hyperparameter set that is expected to cost the user more
 * than it returns, or `null` when the configuration is in the safe regime.
 *
 * Returned as data (not rendered text) so the check is unit-testable and the
 * studio can render it wherever it fits.
 */
export function trainingRegimeAdvisory(hyperparams: TrainingHyperparams): string | null {
  if (!hyperparams.loraMode && hyperparams.epochs >= FULL_MODE_ADVISORY_EPOCHS) {
    return `Full retraining for ${hyperparams.epochs} epochs backpropagates through every weight in the network. It measured a lower held-out loss than the LoRA adapter, but most of that gain arrives in the first few epochs — later epochs cost a full backprop pass each and mostly refit turns the model has already seen. Keep a full retrain short (1–3 epochs), or switch to LoRA.`;
  }
  return null;
}

interface FineTuningStudioProps {
  model: SmallLanguageModel;
  /** The live datasets (owned by App) including user-authored and imported turns. */
  datasets: DatasetPreset[];
  onTrainingComplete: (datasetName: string) => void;
  /** Called after the weights are reset, so App can clear its fine-tuned state. */
  onWeightsReset: () => void;
  onNavigateToChat: () => void;
  activeDatasetId: string;
  setActiveDatasetId: (id: string) => void;
}

export const FineTuningStudio: React.FC<FineTuningStudioProps> = ({
  model,
  datasets,
  onTrainingComplete,
  onWeightsReset,
  onNavigateToChat,
  activeDatasetId,
  setActiveDatasetId,
}) => {
  const [datasetScale, setDatasetScale] = useState<'standard' | 'expanded' | 'large'>('standard');
  const [hyperparams, setHyperparams] = useState<TrainingHyperparams>({
    epochs: 10,
    learningRate: 0.015,
    batchSize: 1,
    weightDecay: 0.005,
    loraMode: true,
    loraRank: model.config.loraRank,
  });

  const [trainingState, setTrainingState] = useState<TrainingState>({
    isTraining: false,
    isPaused: false,
    currentEpoch: 0,
    totalEpochs: 10,
    currentStep: 0,
    totalSteps: 0,
    lossHistory: [],
    currentLoss: 0,
    currentPerplexity: 0,
    currentBlendedLoss: 0,
    sampleOutputs: [],
  });

  const [evalState, setEvalState] = useState<{
    isEvaluating: boolean;
    summary: EvaluationSummary | null;
  }>({
    isEvaluating: false,
    summary: null,
  });
  const [customTestPrompt, setCustomTestPrompt] = useState('');
  const [customTestOutput, setCustomTestOutput] = useState<{
    prompt: string;
    response: string;
    latencyMs: number;
    tokensCount: number;
  } | null>(null);
  const [isTestingPrompt, setIsTestingPrompt] = useState(false);

  const isTrainingRef = useRef(false);
  const isPausedRef = useRef(false);

  const selectedPreset =
    datasets.find((d) => d.id === activeDatasetId) || datasets[0];

  // Resolve active training dataset depending on scale selector
  const activeTurns = React.useMemo(() => {
    if (datasetScale === 'standard') return selectedPreset.turns;
    if (datasetScale === 'expanded') return generateExpandedChatCorpus(selectedPreset, 40);
    return generateExpandedChatCorpus(selectedPreset, 100);
  }, [selectedPreset, datasetScale]);

  /**
   * Held-out split.
   *
   * Every fourth turn is withheld from training and only used to score the
   * model. In-sample loss falls whether or not anything generalisable was
   * learned, so a held-out number is the only honest answer to "did it learn
   * English?" (ISS-20/ISS-22). The split is positional and deterministic, so
   * the same dataset always produces the same train/held-out division.
   */
  const heldOutTurns = React.useMemo(
    () => activeTurns.filter((_, i) => i % 4 === 3),
    [activeTurns]
  );
  const trainTurns = React.useMemo(
    () => activeTurns.filter((_, i) => i % 4 !== 3),
    [activeTurns]
  );

  const evalPrompt = selectedPreset.turns[0]?.user
    ? model.tokenizer.formatConversationPrompt(selectedPreset.turns[0].user)
    : 'User: hello who are you\nAssistant: ';

  const memoryStats = model.getMemoryStats();

  // Execute streaming training loop
  const startTraining = async () => {
    if (trainingState.isPaused) {
      isTrainingRef.current = true;
      isPausedRef.current = false;
      setTrainingState((prev) => ({ ...prev, isTraining: true, isPaused: false }));
      return;
    }

    isTrainingRef.current = true;
    isPausedRef.current = false;

    // Train on the split only: the held-out turns are reserved for scoring.
    const dataset = trainTurns.length > 0 ? trainTurns : activeTurns;
    const totalEpochs = hyperparams.epochs;
    const totalSteps = totalEpochs * dataset.length;

    setTrainingState({
      isTraining: true,
      isPaused: false,
      currentEpoch: 0,
      totalEpochs,
      currentStep: 0,
      totalSteps,
      lossHistory: [],
      currentLoss: 0,
      currentPerplexity: 0,
      currentBlendedLoss: 0,
      sampleOutputs: [],
    });

    const lossHistory: LossPoint[] = [];
    let stepCount = 0;

    for (let epoch = 1; epoch <= totalEpochs; epoch++) {
      if (!isTrainingRef.current) break;

      let epochLossSum = 0;

      for (let i = 0; i < dataset.length; i++) {
        const turn = dataset[i];

        while (isPausedRef.current) {
          await new Promise((r) => setTimeout(r, 100));
          if (!isTrainingRef.current) break;
        }

        if (!isTrainingRef.current) break;

        // Note: EOS is appended by the tokenizer (addEos) — do not also embed
        // the '<eos>' string here or sequences would end with a double EOS.
        const formattedText = formatTurn(turn);
        const tokens = model.tokenizer.encode(formattedText, true, true);

        // Learning rate decay over epochs
        const currentLr =
          hyperparams.learningRate * (1.0 - (epoch - 1) / Math.max(1, totalEpochs * 1.2));

        const { loss, perplexity, neuralLoss, blendedLoss } = model.trainStep(
          tokens,
          currentLr,
          hyperparams.loraMode,
          hyperparams.weightDecay
        );

        // Record the number the optimizer actually moves, plus the blend-epoch
        // ghost loss so the curve labels stay honest.
        const epochLossToRecord =
          hyperparams.loraMode && hyperparams.learningRate > 0 ? neuralLoss : loss;

        epochLossSum += loss;
        stepCount++;

        // For large datasets, keep loss history size bounded to save memory
        const shouldRecordLoss = dataset.length < 50 || stepCount % Math.ceil(dataset.length / 25) === 0;
        if (shouldRecordLoss) {
          lossHistory.push({
            step: stepCount,
            epoch,
            loss: parseFloat(loss.toFixed(3)),
            perplexity: parseFloat(perplexity.toFixed(1)),
            lr: currentLr,
          });
        }

        // Throttle React state updates to avoid unnecessary renders
        if (stepCount % 2 === 0 || stepCount === totalSteps) {
          setTrainingState((prev) => ({
            ...prev,
            currentEpoch: epoch,
            currentStep: stepCount,
            currentLoss: epochLossToRecord,
            currentBlendedLoss: blendedLoss ?? epochLossToRecord,
            currentPerplexity: perplexity,
            lossHistory: [...lossHistory],
          }));
        }

        // Keep browser UI interactive
        if (stepCount % 2 === 0) {
          await new Promise((r) => setTimeout(r, 8));
        }
      }

      if (!isTrainingRef.current) break;

      // Generate a live sample completion at the end of each epoch, under the
      // same adaptation mode the epoch trained with.
      const sampleGeneration = model.generate(
        evalPrompt,
        {
          temperature: 0.6,
          topK: 20,
          topP: 0.85,
          repetitionPenalty: 1.1,
          maxNewTokens: 20,
        },
        hyperparams.loraMode
      );

      setTrainingState((prev) => ({
        ...prev,
        sampleOutputs: [
          ...prev.sampleOutputs,
          {
            epoch,
            prompt: selectedPreset.turns[0]?.user ?? 'hello',
            response: sampleGeneration.text || 'Thinking...',
          },
        ],
      }));
    }

    const completedSuccessfully = isTrainingRef.current && stepCount === totalSteps;
    isTrainingRef.current = false;
    setTrainingState((prev) => ({
      ...prev,
      isTraining: false,
      isPaused: false,
      currentBlendedLoss: 0,
    }));

    if (completedSuccessfully) {
      onTrainingComplete(selectedPreset.name);
    }
  };

  const pauseTraining = () => {
    isPausedRef.current = true;
    setTrainingState((prev) => ({ ...prev, isPaused: true }));
  };

  const stopTraining = () => {
    isTrainingRef.current = false;
    isPausedRef.current = false;
    setTrainingState((prev) => ({ ...prev, isTraining: false, isPaused: false }));
  };

  const resetWeights = () => {
    stopTraining();
    model.resetToBase();
    // Tell App the model is back at its base checkpoint so the header badge,
    // the chat playground's LoRA switch and the dataset label all follow.
    onWeightsReset();
    setTrainingState({
      isTraining: false,
      isPaused: false,
      currentEpoch: 0,
      totalEpochs: hyperparams.epochs,
      currentStep: 0,
      totalSteps: 0,
      lossHistory: [],
      currentLoss: 0,
      currentPerplexity: 0,
      currentBlendedLoss: 0,
      sampleOutputs: [],
    });
    setEvalState({ isEvaluating: false, summary: null });
    setCustomTestOutput(null);
  };

  const runDatasetEvaluation = async () => {
    if (evalState.isEvaluating || trainingState.isTraining) return;
    setEvalState((prev) => ({ ...prev, isEvaluating: true }));

    // Yield to let the spinner render
    await new Promise((r) => setTimeout(r, 20));

    try {
      const turns = selectedPreset.turns;
      const results: EvaluationResult[] = [];
      let totalOverlap = 0;
      let totalLoss = 0;
      let passedCount = 0;

      for (const turn of turns) {
        // Mirror the adaptation mode training used, so the scorecard describes
        // the network that was actually adapted (LoRA adapters vs. full retrain).
        const evaluated = evaluateTurn(model, turn, hyperparams.loraMode);
        totalOverlap += evaluated.overlap;
        if (evaluated.isPassed) passedCount++;
        totalLoss += evaluated.loss;

        results.push({
          turnId: turn.id,
          category: turn.category || 'General',
          prompt: turn.user,
          target: turn.assistant,
          predicted: evaluated.predicted,
          overlap: parseFloat(evaluated.overlap.toFixed(1)),
          isPassed: evaluated.isPassed,
        });
      }

      // Held-out scoring: turns the trainer never saw, under the active
      // adaptation mode. This is the number that answers "did it learn
      // English?" — in-sample loss falls whether or not anything generalisable
      // was learned (ISS-20/ISS-22).
      const scoringTurns = heldOutTurns.length > 0 ? heldOutTurns : turns;
      let heldOutLoss = 0;
      for (const turn of scoringTurns) {
        heldOutLoss += evaluateTurn(model, turn, hyperparams.loraMode).loss;
      }
      heldOutLoss /= scoringTurns.length;

      // Fit the neural/memory blend on those same unseen turns. A network that
      // has learned something earns a larger share of the sampling mass here;
      // the fix is measured, not assumed (ISS-16).
      const calibration = model.calibrateNeuralMix(
        scoringTurns.map(formatTurn),
        undefined,
        hyperparams.loraMode
      );

      setEvalState({
        isEvaluating: false,
        summary: {
          totalTurns: turns.length,
          passedTurns: passedCount,
          avgOverlap: parseFloat((totalOverlap / turns.length).toFixed(1)),
          avgLoss: parseFloat((totalLoss / turns.length).toFixed(3)),
          results,
          heldOutTurns: scoringTurns.length,
          heldOutLoss: parseFloat(heldOutLoss.toFixed(3)),
          neuralInfluence: parseFloat(calibration.mix.toFixed(3)),
          neuralPerplexity: parseFloat(calibration.neuralPerplexity.toFixed(1)),
          memoryPerplexity: parseFloat(calibration.memoryPerplexity.toFixed(1)),
        },
      });
    } catch (err) {
      console.error('Dataset evaluation failed:', err);
      setEvalState((prev) => ({ ...prev, isEvaluating: false }));
    }
  };

  const handleTestSinglePrompt = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const q = customTestPrompt.trim();
    if (!q || isTestingPrompt) return;

    setIsTestingPrompt(true);
    const start = performance.now();
    const prompt = model.tokenizer.formatConversationPrompt(q);
    const res = model.generate(
      prompt,
      {
        temperature: 0.3,
        topK: 12,
        topP: 0.85,
        repetitionPenalty: 1.15,
        maxNewTokens: 40,
      },
      hyperparams.loraMode
    );
    const latency = Math.round(performance.now() - start);

    setCustomTestOutput({
      prompt: q,
      response: res.text || 'No response generated.',
      latencyMs: latency,
      tokensCount: res.tokens.length,
    });
    setIsTestingPrompt(false);
  };

  // SVG Chart Dimensions & Computations
  const chartWidth = 600;
  const chartHeight = 180;
  const padding = 30;

  const points = trainingState.lossHistory;
  const initialLoss = points.length > 0 ? points[0].loss : 0;
  const currentLoss = trainingState.currentLoss || (points.length > 0 ? points[points.length - 1].loss : 0);
  const maxObservedLoss = points.length > 0 ? Math.max(...points.map((p) => p.loss)) : 1;
  const minObservedLoss = points.length > 0 ? Math.min(...points.map((p) => p.loss)) : TARGET_LOSS;
  const chartTop = Math.max(maxObservedLoss, initialLoss, TARGET_LOSS) + 0.15;
  const chartBottom = Math.min(minObservedLoss, TARGET_LOSS) - 0.15;
  const lossRange = Math.max(0.2, chartTop - chartBottom);
  const lossImprovement = initialLoss > 0 ? Math.max(0, ((initialLoss - currentLoss) / initialLoss) * 100) : 0;
  const targetGap = Math.max(0, currentLoss - TARGET_LOSS);

  const getSvgCoordinates = (point: LossPoint, idx: number) => {
    const x = padding + (idx / Math.max(1, points.length - 1)) * (chartWidth - padding * 2);
    const y =
      chartHeight -
      padding -
      ((point.loss - chartBottom) / lossRange) * (chartHeight - padding * 2);
    return `${x},${y}`;
  };

  const polylinePoints = points.map((p, idx) => getSvgCoordinates(p, idx)).join(' ');

  const regimeAdvisory = trainingRegimeAdvisory(hyperparams);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 space-y-4">
      {/* Header Banner */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs flex flex-col sm:flex-row sm:flex-wrap sm:items-center sm:justify-between gap-3 min-w-0">
        <div>
          <div className="flex items-start sm:items-center gap-2 min-w-0">
            <span className="p-1.5 shrink-0 bg-amber-50 text-amber-600 rounded-lg border border-amber-200">
              <TrendingDown className="w-5 h-5" />
            </span>
            <h2 className="text-base font-bold text-slate-900 break-words">
              Conversational Fine-Tuning Studio
            </h2>
            <span className="text-[11px] bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-md border border-emerald-200 font-medium flex items-center gap-1">
              <HardDrive className="w-3 h-3" />
              {memoryStats.totalMemoryFormatted} RAM
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Train and adapt conversational Small Language Model weights with streaming batch processing and low memory consumption.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex flex-wrap items-center gap-2 w-full sm:w-auto">
          {trainingState.isTraining && !trainingState.isPaused ? (
            <button
              onClick={pauseTraining}
              id="pause-training-btn"
              className="bg-amber-500 hover:bg-amber-600 text-white px-3.5 py-2 rounded-lg text-xs font-semibold flex items-center gap-1.5 shadow-2xs transition-colors cursor-pointer"
            >
              <Pause className="w-3.5 h-3.5" />
              <span>Pause</span>
            </button>
          ) : (
            <button
              onClick={startTraining}
              id="start-training-btn"
              className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-1.5 shadow-2xs transition-colors cursor-pointer"
            >
              <Play className="w-3.5 h-3.5" />
              <span>{trainingState.isPaused ? 'Resume Training' : 'Start Fine-Tuning'}</span>
            </button>
          )}

          {trainingState.isTraining && (
            <button
              onClick={stopTraining}
              id="stop-training-btn"
              className="bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-2 rounded-lg text-xs font-medium transition-colors cursor-pointer"
            >
              Stop
            </button>
          )}

          <button
            onClick={resetWeights}
            disabled={trainingState.isTraining}
            id="reset-studio-btn"
            title="Reset model to base pretrained checkpoint"
            className="bg-white hover:bg-slate-50 border border-slate-200 text-slate-600 hover:text-rose-600 px-3 py-2 rounded-lg text-xs font-medium flex items-center gap-1 transition-colors disabled:opacity-50 cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Weights</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start min-w-0">
        {/* Left Column: Dataset & Hyperparameters */}
        <div className="lg:col-span-4 space-y-4 min-w-0">
          {/* Dataset Scale Selector (Supports Big Datasets) */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-slate-900 flex items-center gap-1.5 uppercase tracking-wider">
                <Database className="w-3.5 h-3.5 text-indigo-600" />
                <span>Dataset Scale & Volume</span>
              </h3>
              <span className="text-[11px] font-mono text-indigo-600 font-semibold">
                {activeTurns.length} turns
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5 text-xs min-w-0">
              <button
                type="button"
                onClick={() => !trainingState.isTraining && setDatasetScale('standard')}
                disabled={trainingState.isTraining}
                className={`py-1.5 px-2 rounded-lg border font-medium text-center transition-all cursor-pointer ${
                  datasetScale === 'standard'
                    ? 'bg-indigo-50 border-indigo-300 text-indigo-700 font-semibold'
                    : 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100'
                }`}
              >
                Standard (8)
              </button>
              <button
                type="button"
                onClick={() => !trainingState.isTraining && setDatasetScale('expanded')}
                disabled={trainingState.isTraining}
                className={`py-1.5 px-2 rounded-lg border font-medium text-center transition-all cursor-pointer ${
                  datasetScale === 'expanded'
                    ? 'bg-indigo-50 border-indigo-300 text-indigo-700 font-semibold'
                    : 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100'
                }`}
              >
                Expanded (40)
              </button>
              <button
                type="button"
                onClick={() => !trainingState.isTraining && setDatasetScale('large')}
                disabled={trainingState.isTraining}
                className={`py-1.5 px-2 rounded-lg border font-medium text-center transition-all cursor-pointer ${
                  datasetScale === 'large'
                    ? 'bg-indigo-50 border-indigo-300 text-indigo-700 font-semibold'
                    : 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100'
                }`}
              >
                Big Dataset (100)
              </button>
            </div>
            <p className="text-[11px] text-slate-500">
              Streams and trains through micro-batches without increasing memory footprint.
            </p>
          </div>

          {/* Target Conversational Persona */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-bold text-slate-900 flex items-center gap-1.5 uppercase tracking-wider">
                <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                <span>Conversational Style</span>
              </h3>
            </div>

            <div className="space-y-2">
              {datasets.map((dataset) => {
                const isSelected = dataset.id === activeDatasetId;
                return (
                  <button
                    key={dataset.id}
                    onClick={() => !trainingState.isTraining && setActiveDatasetId(dataset.id)}
                    disabled={trainingState.isTraining}
                    className={`w-full text-left p-2.5 rounded-lg border text-xs transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-indigo-50/70 border-indigo-300 ring-1 ring-indigo-500'
                        : 'bg-white border-slate-200 hover:bg-slate-50'
                    } disabled:opacity-60 disabled:cursor-not-allowed`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-slate-900">{dataset.name}</span>
                      <span className="text-[10px] bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded font-medium">
                        {dataset.badge}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-500 line-clamp-2 leading-relaxed">
                      {dataset.description}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Hyperparameters Config */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-3">
            <h3 className="text-xs font-bold text-slate-900 flex items-center gap-1.5 uppercase tracking-wider">
              <Sliders className="w-3.5 h-3.5 text-indigo-600" />
              <span>Training Hyperparameters</span>
            </h3>

            <div>
              <label className="text-[11px] font-medium text-slate-600 block mb-1">
                Adaptation Mode
              </label>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <button
                  type="button"
                  onClick={() => setHyperparams({ ...hyperparams, loraMode: true })}
                  disabled={trainingState.isTraining}
                  className={`p-2 rounded-lg border font-medium text-center transition-all cursor-pointer ${
                    hyperparams.loraMode
                      ? 'bg-indigo-600 text-white border-indigo-600'
                      : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                  }`}
                >
                  LoRA Adapter
                </button>
                <button
                  type="button"
                  onClick={() => setHyperparams({ ...hyperparams, loraMode: false })}
                  disabled={trainingState.isTraining}
                  className={`p-2 rounded-lg border font-medium text-center transition-all cursor-pointer ${
                    !hyperparams.loraMode
                      ? 'bg-indigo-600 text-white border-indigo-600'
                      : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                  }`}
                >
                  Full Retrain
                </button>
              </div>
            </div>

            {regimeAdvisory && (
              <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-2.5 text-[11px] leading-relaxed text-amber-800">
                <TriangleAlert className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-amber-600" />
                <span>{regimeAdvisory}</span>
              </div>
            )}

            <div>
              <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                <span>Epochs</span>
                <span className="font-mono text-indigo-600">{hyperparams.epochs}</span>
              </div>
              <input
                type="range"
                min="3"
                max="25"
                step="1"
                disabled={trainingState.isTraining}
                value={hyperparams.epochs}
                onChange={(e) =>
                  setHyperparams({ ...hyperparams, epochs: parseInt(e.target.value) })
                }
                className="w-full accent-indigo-600 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                <span>Learning Rate</span>
                <span className="font-mono text-indigo-600">
                  {hyperparams.learningRate.toFixed(3)}
                </span>
              </div>
              <input
                type="range"
                min="0.002"
                max="0.04"
                step="0.002"
                disabled={trainingState.isTraining}
                value={hyperparams.learningRate}
                onChange={(e) =>
                  setHyperparams({ ...hyperparams, learningRate: parseFloat(e.target.value) })
                }
                className="w-full accent-indigo-600 cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Center & Right Column: Metrics & Live Loss Curve */}
        <div className="lg:col-span-8 space-y-4 min-w-0">
          {/* Progress & Loss Card */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-4 min-w-0 overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100 min-w-0">
              <div className="flex items-center gap-2">
                <LineChart className="w-4 h-4 text-indigo-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  Loss Curve & Perplexity
                </h3>
              </div>

              {trainingState.isTraining && (
                <div className="flex items-center gap-2 text-xs">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
                  <span className="font-semibold text-emerald-700">
                    Step {trainingState.currentStep} / {trainingState.totalSteps}
                  </span>
                </div>
              )}
            </div>

            {/* Quick Metrics Dashboard */}
            <div className="grid grid-cols-2 xl:grid-cols-4 gap-2.5 text-center min-w-0">
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                <span className="text-[11px] text-slate-500 block">Current Loss</span>
                <span className="text-base font-bold font-mono text-indigo-600">
                  {trainingState.currentLoss ? trainingState.currentLoss.toFixed(4) : '--'}
                </span>
              </div>

              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                <span className="text-[11px] text-slate-500 block">Perplexity (PPL)</span>
                <span className="text-base font-bold font-mono text-emerald-600">
                  {trainingState.currentPerplexity ? trainingState.currentPerplexity.toFixed(1) : '--'}
                </span>
              </div>

              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                <span className="text-[11px] text-slate-500 block">Epoch</span>
                <span className="text-base font-bold font-mono text-slate-800">
                  {trainingState.currentEpoch} / {hyperparams.epochs}
                </span>
              </div>

              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                <span className="text-[11px] text-slate-500 block">RAM Usage</span>
                <span className="text-base font-bold font-mono text-slate-800">
                  {memoryStats.totalMemoryFormatted}
                </span>
              </div>

              <div className="col-span-2 xl:col-span-4 grid grid-cols-1 sm:grid-cols-3 gap-2">
                <div className="rounded-lg border border-emerald-200 bg-emerald-50/70 p-2.5 text-left">
                  <span className="text-[11px] text-emerald-700 block">Target Loss</span>
                  <span className="text-base font-bold font-mono text-emerald-800">{TARGET_LOSS.toFixed(2)}</span>
                </div>
                <div className="rounded-lg border border-indigo-200 bg-indigo-50/70 p-2.5 text-left">
                  <span className="text-[11px] text-indigo-700 block">Improvement</span>
                  <span className="text-base font-bold font-mono text-indigo-800">{lossImprovement.toFixed(1)}%</span>
                </div>
                <div className="rounded-lg border border-amber-200 bg-amber-50/70 p-2.5 text-left">
                  <span className="text-[11px] text-amber-700 block">Gap to Target</span>
                  <span className="text-base font-bold font-mono text-amber-800">{targetGap.toFixed(3)}</span>
                </div>
              </div>
            </div>

            {/* SVG Loss Curve */}
            <div className="bg-slate-900 rounded-xl p-3 relative overflow-hidden min-w-0">
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-1 text-[10px] sm:text-[11px] text-slate-400 mb-2 px-1">
                <span>Top: {chartTop.toFixed(2)}</span>
                <span className="font-mono text-indigo-400 text-center">Target: {TARGET_LOSS.toFixed(2)}</span>
                <span className="text-right">Bottom: {chartBottom.toFixed(2)}</span>
              </div>

              <div className="w-full h-44 flex items-center justify-center">
                {points.length > 1 ? (
                  <svg
                    viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                    className="w-full h-full overflow-visible"
                  >
                    <line
                      x1={padding}
                      y1={chartHeight - padding}
                      x2={chartWidth - padding}
                      y2={chartHeight - padding}
                      stroke="#334155"
                      strokeWidth="1"
                    />
                    <line
                      x1={padding}
                      y1={chartHeight - padding - ((TARGET_LOSS - chartBottom) / lossRange) * (chartHeight - padding * 2)}
                      x2={chartWidth - padding}
                      y2={chartHeight - padding - ((TARGET_LOSS - chartBottom) / lossRange) * (chartHeight - padding * 2)}
                      stroke="#34d399"
                      strokeWidth="1.5"
                      strokeDasharray="6 4"
                    />

                    <polyline
                      fill="none"
                      stroke="#6366f1"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      points={polylinePoints}
                    />

                    {points.map((p, idx) => {
                      const coords = getSvgCoordinates(p, idx).split(',');
                      return (
                        <circle
                          key={idx}
                          cx={coords[0]}
                          cy={coords[1]}
                          r="3"
                          fill="#818cf8"
                          className="hover:r-4 transition-all"
                        />
                      );
                    })}
                  </svg>
                ) : (
                  <div className="text-slate-500 text-xs text-center space-y-1">
                    <p>No training steps recorded yet.</p>
                    <p className="text-[11px] text-slate-600">
                      Click "Start Fine-Tuning" to watch the loss decrease in real time.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Epoch Evaluation Completions */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100">
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                <span>Conversational Progress per Epoch</span>
              </h3>
              <button
                onClick={onNavigateToChat}
                className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 flex items-center gap-1 cursor-pointer"
              >
                <span>Test in Chat</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="space-y-2 max-h-56 overflow-y-auto min-w-0">
              {trainingState.sampleOutputs.length > 0 ? (
                trainingState.sampleOutputs.map((out, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs space-y-1"
                  >
                    <div className="flex items-center justify-between text-slate-500 text-[11px]">
                      <span className="font-semibold text-slate-700">Epoch {out.epoch}</span>
                      <span className="font-mono text-[10px]">Prompt: "{out.prompt}"</span>
                    </div>
                    <p className="text-indigo-950 font-medium pl-2 border-l-2 border-indigo-400">
                      {out.response}
                    </p>
                  </div>
                ))
              ) : (
                <p className="text-xs text-slate-400 py-3 text-center">
                  Live responses generated after each epoch will appear here as the model learns.
                </p>
              )}
            </div>
          </div>

          {/* Trained Data Verification & Evaluation Suite */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 shadow-2xs space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
              <div>
                <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-indigo-600" />
                  <span>Trained Data Verification & Accuracy Suite</span>
                </h3>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Evaluate current model performance against all {selectedPreset.turns.length} ground-truth turns in{' '}
                  <span className="font-semibold text-slate-700">"{selectedPreset.name}"</span>.
                </p>
              </div>
              <button
                type="button"
                onClick={runDatasetEvaluation}
                disabled={evalState.isEvaluating || trainingState.isTraining}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                  evalState.isEvaluating || trainingState.isTraining
                    ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200'
                    : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-xs'
                }`}
              >
                {evalState.isEvaluating ? (
                  <>
                    <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Evaluating Turns...</span>
                  </>
                ) : (
                  <>
                    <Check className="w-3.5 h-3.5" />
                    <span>Run Full Evaluation</span>
                  </>
                )}
              </button>
            </div>

            {/* Scorecard Strip */}
            {evalState.summary ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-2.5 min-w-0">
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                    <span className="text-[11px] font-medium text-slate-500 block">Avg Word Match</span>
                    <span className="text-lg font-bold text-slate-900">
                      {evalState.summary.avgOverlap}%
                    </span>
                  </div>
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                    <span className="text-[11px] font-medium text-slate-500 block">Turns Passed</span>
                    <div className="flex items-baseline gap-1">
                      <span className="text-lg font-bold text-emerald-700">
                        {evalState.summary.passedTurns}
                      </span>
                      <span className="text-xs text-slate-400">/ {evalState.summary.totalTurns}</span>
                    </div>
                  </div>
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                    <span className="text-[11px] font-medium text-slate-500 block">Cross-Entropy Loss</span>
                    <span className="text-lg font-bold text-slate-900 font-mono">
                      {evalState.summary.avgLoss}
                    </span>
                  </div>
                  <div className="bg-amber-50/60 border border-amber-200 rounded-lg p-3">
                    <span className="text-[11px] font-medium text-amber-700 block">Held-Out Loss</span>
                    <div className="flex items-baseline gap-1">
                      <span className="text-lg font-bold text-slate-900 font-mono">
                        {evalState.summary.heldOutLoss}
                      </span>
                      <span className="text-[10px] text-amber-700">
                        {evalState.summary.heldOutTurns} unseen
                      </span>
                    </div>
                  </div>
                  <div className="bg-indigo-50/60 border border-indigo-200 rounded-lg p-3">
                    <span className="text-[11px] font-medium text-indigo-700 block">Neural Influence</span>
                    <div className="flex items-baseline gap-1">
                      <span className="text-lg font-bold text-slate-900 font-mono">
                        {Math.round(evalState.summary.neuralInfluence * 100)}%
                      </span>
                      <span className="text-[10px] text-indigo-700">
                        ppl {evalState.summary.neuralPerplexity} vs {evalState.summary.memoryPerplexity}
                      </span>
                    </div>
                  </div>
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                    <span className="text-[11px] font-medium text-slate-500 block">Model Status</span>
                    {(() => {
                      const isAdapted = model.isFineTuned();
                      return (
                        <span className={`inline-block text-xs font-semibold px-2 py-0.5 mt-0.5 rounded ${
                          isAdapted
                            ? 'bg-indigo-50 text-indigo-700 border border-indigo-200'
                            : 'bg-amber-50 text-amber-700 border border-amber-200'
                        }`}>
                          {isAdapted ? 'LoRA Adapted' : 'Base Pretrained'}
                        </span>
                      );
                    })()}
                  </div>
                </div>

                {/* Per-Turn Comparison Table */}
                <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                  {evalState.summary.results.map((res, idx) => (
                    <div
                      key={res.turnId}
                      className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs space-y-1.5"
                    >
                      <div className="flex flex-wrap items-start justify-between gap-2">
                        <div className="flex flex-wrap items-center gap-2 min-w-0">
                          <span className="font-semibold text-slate-700 text-[11px]">
                            Turn {idx + 1}
                          </span>
                          <span className="text-[10px] px-1.5 py-0.5 bg-slate-200 text-slate-600 rounded">
                            {res.category}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[11px] font-semibold text-slate-600">
                            {res.overlap}% overlap
                          </span>
                          <span
                            className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                              res.overlap >= 60
                                ? 'bg-emerald-100 text-emerald-800'
                                : res.overlap >= 40
                                ? 'bg-indigo-100 text-indigo-800'
                                : 'bg-amber-100 text-amber-800'
                            }`}
                          >
                            {res.isPassed ? 'PASS' : 'PARTIAL'}
                          </span>
                        </div>
                      </div>

                      <div className="text-[11px] text-slate-600 break-words">
                        <span className="font-semibold text-slate-800">Q: </span>
                        {res.prompt}
                      </div>

                      <div className="text-[11px] text-slate-500 pl-2 border-l-2 border-slate-300 break-words">
                        <span className="font-semibold text-slate-600">Expected: </span>
                        {res.target}
                      </div>

                      <div className="text-[11px] text-indigo-950 font-medium pl-2 border-l-2 border-indigo-400 bg-white/60 py-1 rounded-r break-words">
                        <span className="font-semibold text-indigo-700">Model Output: </span>
                        {res.predicted}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 text-center space-y-1">
                <p className="text-xs text-slate-600 font-medium">
                  Click "Run Full Evaluation" to test the model across all turns of the current dataset.
                </p>
                <p className="text-[11px] text-slate-400">
                  Measures lexical target overlap %, cross-entropy loss, and shows side-by-side completions.
                </p>
              </div>
            )}

            {/* Quick Single Prompt Playground Tester */}
            <div className="pt-3 border-t border-slate-100 space-y-2">
              <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider block">
                Quick Single-Prompt Tester
              </span>
              <form onSubmit={handleTestSinglePrompt} className="flex gap-2">
                <input
                  type="text"
                  value={customTestPrompt}
                  onChange={(e) => setCustomTestPrompt(e.target.value)}
                  placeholder={`Try e.g., "${selectedPreset.turns[0]?.user || 'hello how are you'}"`}
                  className="flex-1 px-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
                <button
                  type="submit"
                  disabled={!customTestPrompt.trim() || isTestingPrompt}
                  className="px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-white rounded-lg text-xs font-semibold flex items-center gap-1 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Send className="w-3 h-3" />
                  <span>Test</span>
                </button>
              </form>

              {customTestOutput && (
                <div className="p-3 bg-indigo-50/50 rounded-lg border border-indigo-100 text-xs space-y-1">
                  <div className="flex items-center justify-between text-[10px] text-slate-500">
                    <span className="font-semibold text-slate-700">Q: "{customTestOutput.prompt}"</span>
                    <span>{customTestOutput.latencyMs}ms ({customTestOutput.tokensCount} tokens)</span>
                  </div>
                  <p className="text-indigo-950 font-medium pl-2 border-l-2 border-indigo-400">
                    {customTestOutput.response}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
