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
import { ModelConfig } from './types';

export default function App() {
  const [selectedModelConfig, setSelectedModelConfig] = useState<ModelConfig>(PREDEFINED_MODELS[0]);
  
  // Initialize the SLM instance in state
  const [model, setModel] = useState(() => initializePretrainedModel(PREDEFINED_MODELS[0]));
  const [isFinetuned, setIsFinetuned] = useState(false);
  const [activeDatasetId, setActiveDatasetId] = useState(PREDEFINED_DATASETS[0].id);
  const [activeDatasetName, setActiveDatasetName] = useState('');
  const [activeTab, setActiveTab] = useState<'chat' | 'train' | 'inspect' | 'datasets'>('chat');

  // Handle model change
  const handleSelectModel = (newConfig: ModelConfig) => {
    setSelectedModelConfig(newConfig);
    const newModel = initializePretrainedModel(newConfig);
    setModel(newModel);
    setIsFinetuned(false);
    setActiveDatasetName('');
  };

  // Handle reset to base
  const handleResetToBase = () => {
    model.resetToBase();
    setIsFinetuned(false);
    setActiveDatasetName('');
  };

  // Handle fine-tuning completion
  const handleTrainingComplete = (datasetName: string) => {
    setIsFinetuned(true);
    setActiveDatasetName(datasetName);
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
          />
        )}

        {activeTab === 'train' && (
          <FineTuningStudio
            model={model}
            onTrainingComplete={handleTrainingComplete}
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
            activeDatasetId={activeDatasetId}
            setActiveDatasetId={setActiveDatasetId}
            onNavigateToTrain={() => setActiveTab('train')}
          />
        )}
      </main>
    </div>
  );
}
