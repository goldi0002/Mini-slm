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
  ArrowRight
} from 'lucide-react';
import { 
  TrainingHyperparams, 
  LossPoint, 
  TrainingState 
} from '../types';
import { SmallLanguageModel } from '../slm/transformer';
import { PREDEFINED_DATASETS, generateExpandedChatCorpus } from '../slm/datasets';
import { SPECIAL_TOKENS } from '../slm/tokenizer';

interface FineTuningStudioProps {
  model: SmallLanguageModel;
  onTrainingComplete: (datasetName: string) => void;
  onNavigateToChat: () => void;
  activeDatasetId: string;
  setActiveDatasetId: (id: string) => void;
}

export const FineTuningStudio: React.FC<FineTuningStudioProps> = ({
  model,
  onTrainingComplete,
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
    sampleOutputs: [],
  });

  const isTrainingRef = useRef(false);
  const isPausedRef = useRef(false);

  const selectedPreset =
    PREDEFINED_DATASETS.find((d) => d.id === activeDatasetId) || PREDEFINED_DATASETS[0];

  // Resolve active training dataset depending on scale selector
  const activeTurns = React.useMemo(() => {
    if (datasetScale === 'standard') return selectedPreset.turns;
    if (datasetScale === 'expanded') return generateExpandedChatCorpus(selectedPreset, 40);
    return generateExpandedChatCorpus(selectedPreset, 100);
  }, [selectedPreset, datasetScale]);

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

    const dataset = activeTurns;
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
        const formattedText = `${SPECIAL_TOKENS.USER} ${turn.user} ${SPECIAL_TOKENS.NEWLINE}${SPECIAL_TOKENS.ASSISTANT} ${turn.assistant}`;
        const tokens = model.tokenizer.encode(formattedText, true, true);

        // Learning rate decay over epochs
        const currentLr =
          hyperparams.learningRate * (1.0 - (epoch - 1) / Math.max(1, totalEpochs * 1.2));

        const { loss, perplexity } = model.trainStep(
          tokens,
          currentLr,
          hyperparams.loraMode,
          hyperparams.weightDecay
        );

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
            currentLoss: loss,
            currentPerplexity: perplexity,
            lossHistory: [...lossHistory],
          }));
        }

        // Keep browser UI interactive
        if (dataset.length > 50) {
          if (stepCount % 4 === 0) await new Promise((r) => setTimeout(r, 8));
        } else {
          await new Promise((r) => setTimeout(r, 14));
        }
      }

      // Generate live sample completion at the end of each epoch
      const sampleGeneration = model.generate(
        evalPrompt,
        {
          temperature: 0.6,
          topK: 20,
          topP: 0.85,
          repetitionPenalty: 1.1,
          maxNewTokens: 20,
        },
        true
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

    isTrainingRef.current = false;
    setTrainingState((prev) => ({
      ...prev,
      isTraining: false,
      isPaused: false,
    }));

    onTrainingComplete(selectedPreset.name);
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
      sampleOutputs: [],
    });
  };

  // SVG Chart Dimensions & Computations
  const chartWidth = 600;
  const chartHeight = 180;
  const padding = 30;

  const points = trainingState.lossHistory;
  const maxLoss = points.length > 0 ? Math.max(...points.map((p) => p.loss), 1.0) : 4.0;
  const minLoss = points.length > 0 ? Math.min(...points.map((p) => p.loss), 0.0) : 0.0;
  const lossRange = Math.max(0.1, maxLoss - minLoss);

  const getSvgCoordinates = (point: LossPoint, idx: number) => {
    const x = padding + (idx / Math.max(1, points.length - 1)) * (chartWidth - padding * 2);
    const y =
      chartHeight -
      padding -
      ((point.loss - minLoss) / lossRange) * (chartHeight - padding * 2);
    return `${x},${y}`;
  };

  const polylinePoints = points.map((p, idx) => getSvgCoordinates(p, idx)).join(' ');

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 space-y-4">
      {/* Header Banner */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 bg-amber-50 text-amber-600 rounded-lg border border-amber-200">
              <TrendingDown className="w-5 h-5" />
            </span>
            <h2 className="text-base font-bold text-slate-900">
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
        <div className="flex items-center gap-2">
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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column: Dataset & Hyperparameters */}
        <div className="space-y-4">
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

            <div className="grid grid-cols-3 gap-1.5 text-xs">
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
              {PREDEFINED_DATASETS.map((dataset) => {
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
        <div className="lg:col-span-2 space-y-4">
          {/* Progress & Loss Card */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
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
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
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
            </div>

            {/* SVG Loss Curve */}
            <div className="bg-slate-900 rounded-xl p-3 relative overflow-hidden">
              <div className="flex items-center justify-between text-[11px] text-slate-400 mb-2 px-1">
                <span>Loss: {maxLoss.toFixed(2)}</span>
                <span className="font-mono text-indigo-400">Cross-Entropy Optimization</span>
                <span>Loss: {minLoss.toFixed(2)}</span>
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
                      y1={padding}
                      x2={chartWidth - padding}
                      y2={padding}
                      stroke="#334155"
                      strokeWidth="1"
                      strokeDasharray="4 4"
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

            <div className="space-y-2 max-h-56 overflow-y-auto">
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
        </div>
      </div>
    </div>
  );
};
