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

let initialModel: SmallLanguageModel | null = null;
function getInitialModel(): SmallLanguageModel {
  if (!initialModel) initialModel = initializePretrainedModel(PREDEFINED_MODELS[0]);
  return initialModel;
}

export default function App() {
  const [selectedModelConfig, setSelectedModelConfig] = useState<ModelConfig>(PREDEFINED_MODELS[0]);
  const [model, setModel] = useState(getInitialModel);
  const [isFinetuned, setIsFinetuned] = useState(false);
  const [datasets, setDatasets] = useState<DatasetPreset[]>(PREDEFINED_DATASETS);
  const [activeDatasetId, setActiveDatasetId] = useState(PREDEFINED_DATASETS[0].id);
  const [activeDatasetName, setActiveDatasetName] = useState('');
  const [activeTab, setActiveTab] = useState<'chat' | 'train' | 'inspect' | 'datasets'>('chat');

  const syncResetState = () => {
    setIsFinetuned(false);
    setActiveDatasetName('');
  };

  const handleSelectModel = (newConfig: ModelConfig) => {
    setSelectedModelConfig(newConfig);
    setModel(initializePretrainedModel(newConfig));
    syncResetState();
  };

  const handleResetToBase = () => {
    model.resetToBase();
    syncResetState();
  };

  const handleTrainingComplete = (datasetName: string) => {
    setIsFinetuned(true);
    setActiveDatasetName(datasetName);
  };

  const handleDatasetsChange = (next: DatasetPreset[]) => {
    setDatasets(next);

    const learned = defaultTokenizer.learnWords(
      next.flatMap((d) => d.turns.flatMap((t) => [t.user, t.assistant]))
    );

    if (learned > 0 && model.resizeVocabulary(defaultTokenizer.vocabSize)) {
      setSelectedModelConfig((prev) => ({ ...prev, vocabSize: model.config.vocabSize }));
    }
  };

  const paramStats = useMemo(() => model.countParameters(), [model, selectedModelConfig]);

  return (
    <div className="app-shell min-h-screen text-slate-900">
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

      <main className="app-main">
        <div className="app-main__inner">
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
            <ArchitectureInspector model={model} isFinetuned={isFinetuned} />
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
        </div>
      </main>
    </div>
  );
}
