/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo } from 'react';
import { Header } from './components/Header';
import { ChatPlayground } from './components/ChatPlayground';
import { FineTuningStudio } from './components/FineTuningStudio';
import { ArchitectureInspector } from './components/ArchitectureInspector';
import { DatasetManager } from './components/DatasetManager';
import { PREDEFINED_MODELS, initializePretrainedModel } from './slm/predefinedModels';
import { PREDEFINED_DATASETS } from './slm/datasets';
import { defaultTokenizer } from './slm/tokenizer';
import { DatasetPreset, ModelConfig } from './types';
import { SmallLanguageModel } from './slm/transformer';

// Building a model also pre-trains it on the warm-up corpus, and StrictMode
// invokes state initialisers twice in development. Caching the instance keeps
// startup from paying for that twice.
let initialModel: SmallLanguageModel | null = null;
function getInitialModel(): SmallLanguageModel {
  if (!initialModel) initialModel = initializePretrainedModel(PREDEFINED_MODELS[0]);
  return initialModel;
}

export default function App() {
  const [selectedModelConfig, setSelectedModelConfig] = useState<ModelConfig>(PREDEFINED_MODELS[0]);

  // Initialize the SLM instance in state
  const [model, setModel] = useState(getInitialModel);
  const [isFinetuned, setIsFinetuned] = useState(false);

  // Conversation datasets are owned here so the Dataset Manager's edits and
  // imports are the same turns the Fine-Tuning Studio trains on.
  const [datasets, setDatasets] = useState<DatasetPreset[]>(PREDEFINED_DATASETS);
  const [activeDatasetId, setActiveDatasetId] = useState(PREDEFINED_DATASETS[0].id);
  const [activeDatasetName, setActiveDatasetName] = useState('');
  const [activeTab, setActiveTab] = useState<'chat' | 'train' | 'inspect' | 'datasets'>('chat');

  // Clear the fine-tuned markers in one place, so the header, the chat
  // playground and the trainer can never disagree about whether the current
  // weights are still adapted.
  const syncResetState = () => {
    setIsFinetuned(false);
    setActiveDatasetName('');
  };

  // Handle model change
  const handleSelectModel = (newConfig: ModelConfig) => {
    setSelectedModelConfig(newConfig);
    setModel(initializePretrainedModel(newConfig));
    syncResetState();
  };

  // Handle reset to base
  const handleResetToBase = () => {
    model.resetToBase();
    syncResetState();
  };

  // Handle fine-tuning completion
  const handleTrainingComplete = (datasetName: string) => {
    setIsFinetuned(true);
    setActiveDatasetName(datasetName);
  };

  /**
   * Apply a dataset edit, teaching the tokenizer any words it has not seen yet.
   *
   * New words must be learned *before* a model is built: the embedding table and
   * LM head are sized from the vocabulary at construction time, so a token with
   * no output row could never be generated. Growing the vocabulary therefore
   * rebuilds the model from its base checkpoint.
   */
  const handleDatasetsChange = (next: DatasetPreset[]) => {
    setDatasets(next);

    const learned = defaultTokenizer.learnWords(
      next.flatMap((d) => d.turns.flatMap((t) => [t.user, t.assistant]))
    );

    if (learned > 0) {
      setModel(initializePretrainedModel(selectedModelConfig));
      syncResetState();
    }
  };

  const paramStats = useMemo(() => {
    return model.countParameters();
  }, [model, selectedModelConfig]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans selection:bg-indigo-100 selection:text-indigo-900">
      {/* Top Navigation & Status Header */}
      <Header
        models={PREDEFINED_MODELS}
        currentModel={selectedModelConfig}
        onSelectModel={handleSelectModel}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isFinetuned={isFinetuned}
        onResetToBase={handleResetToBase}
        paramStats={paramStats}
        memoryFormatted={model.getMemoryStats().totalMemoryFormatted}
      />

      {/* Main Content Areas */}
      <main className="flex-1 overflow-x-hidden">
        {activeTab === 'chat' && (
          <ChatPlayground
            model={model}
            isFinetuned={isFinetuned}
            activeDatasetName={activeDatasetName}
            createBaseModel={() => initializePretrainedModel(model.config)}
          />
        )}

        {activeTab === 'train' && (
          <FineTuningStudio
            model={model}
            datasets={datasets}
            onTrainingComplete={handleTrainingComplete}
            onWeightsReset={syncResetState}
            onNavigateToChat={() => setActiveTab('chat')}
            activeDatasetId={activeDatasetId}
            setActiveDatasetId={setActiveDatasetId}
          />
        )}

        {activeTab === 'inspect' && (
          <ArchitectureInspector
            model={model}
            isFinetuned={isFinetuned}
          />
        )}

        {activeTab === 'datasets' && (
          <DatasetManager
            datasets={datasets}
            onDatasetsChange={handleDatasetsChange}
            activeDatasetId={activeDatasetId}
            setActiveDatasetId={setActiveDatasetId}
            onNavigateToTrain={() => setActiveTab('train')}
          />
        )}
      </main>
    </div>
  );
}
