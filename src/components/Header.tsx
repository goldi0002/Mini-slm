/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { 
  Bot, 
  Cpu, 
  Flame, 
  Layers, 
  Database, 
  RotateCcw,
  Sparkles,
  HardDrive
} from 'lucide-react';
import { ModelConfig } from '../types';

interface HeaderProps {
  models: ModelConfig[];
  currentModel: ModelConfig;
  onSelectModel: (model: ModelConfig) => void;
  activeTab: 'chat' | 'train' | 'inspect' | 'datasets';
  setActiveTab: (tab: 'chat' | 'train' | 'inspect' | 'datasets') => void;
  isFinetuned: boolean;
  onResetToBase: () => void;
  paramStats: { total: number; trainable: number; loraOnly: number };
  memoryFormatted?: string;
}

export const Header: React.FC<HeaderProps> = ({
  models,
  currentModel,
  onSelectModel,
  activeTab,
  setActiveTab,
  isFinetuned,
  onResetToBase,
  paramStats,
  memoryFormatted = '1.8 MB',
}) => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo and Identity */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center shadow-sm">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold text-slate-900 leading-none">
                  Conversational SLM
                </h1>
                <span className="px-2 py-0.5 text-[11px] font-semibold bg-emerald-50 text-emerald-700 rounded-full border border-emerald-200 flex items-center gap-1">
                  <HardDrive className="w-3 h-3" />
                  {memoryFormatted} RAM
                </span>
                {isFinetuned ? (
                  <span className="px-2 py-0.5 text-[11px] font-semibold bg-indigo-50 text-indigo-700 rounded-full border border-indigo-200 flex items-center gap-1">
                    <Sparkles className="w-3 h-3" /> Fine-Tuned
                  </span>
                ) : (
                  <span className="px-2 py-0.5 text-[11px] font-medium bg-slate-100 text-slate-600 rounded-full">
                    Base Model
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-500 mt-0.5 hidden sm:block">
                Ultra-lightweight conversational AI assistant running 100% locally in browser memory
              </p>
            </div>
          </div>

          {/* Model Selector and Reset */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <select
                id="model-selector"
                value={currentModel.id}
                onChange={(e) => {
                  const m = models.find((mod) => mod.id === e.target.value);
                  if (m) onSelectModel(m);
                }}
                aria-label="Select Small Language Model"
                className="text-xs font-medium bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-800 rounded-lg px-3 py-1.5 focus:outline-hidden focus:ring-2 focus:ring-indigo-500 cursor-pointer pr-8"
              >
                {models.map((mod) => (
                  <option key={mod.id} value={mod.id}>
                    {mod.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="hidden lg:flex items-center gap-2 text-xs bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1 text-slate-600">
              <span className="text-slate-400">Params:</span>
              <span className="font-mono font-semibold text-slate-800">
                {paramStats.total.toLocaleString()}
              </span>
            </div>

            {isFinetuned && (
              <button
                onClick={onResetToBase}
                id="reset-base-btn"
                title="Revert model weights back to base pretrained state"
                className="text-xs font-medium text-slate-600 hover:text-rose-600 px-2.5 py-1.5 rounded-lg border border-slate-200 hover:border-rose-200 bg-white hover:bg-rose-50 transition-colors flex items-center gap-1.5 cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Reset Weights</span>
              </button>
            )}
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-1 border-t border-slate-100 pt-1 pb-2 overflow-x-auto">
          <button
            onClick={() => setActiveTab('chat')}
            id="nav-chat-tab"
            className={`px-3.5 py-1.5 text-xs font-medium rounded-lg transition-all flex items-center gap-2 cursor-pointer ${
              activeTab === 'chat'
                ? 'bg-indigo-50 text-indigo-700 font-semibold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <Bot className="w-4 h-4" />
            <span>Chat with Assistant</span>
          </button>

          <button
            onClick={() => setActiveTab('train')}
            id="nav-train-tab"
            className={`px-3.5 py-1.5 text-xs font-medium rounded-lg transition-all flex items-center gap-2 cursor-pointer ${
              activeTab === 'train'
                ? 'bg-indigo-50 text-indigo-700 font-semibold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <Flame className="w-4 h-4 text-amber-500" />
            <span>Train & Fine-Tune (Big Datasets)</span>
          </button>

          <button
            onClick={() => setActiveTab('datasets')}
            id="nav-datasets-tab"
            className={`px-3.5 py-1.5 text-xs font-medium rounded-lg transition-all flex items-center gap-2 cursor-pointer ${
              activeTab === 'datasets'
                ? 'bg-indigo-50 text-indigo-700 font-semibold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <Database className="w-4 h-4 text-emerald-500" />
            <span>User & Assistant Datasets</span>
          </button>

          <button
            onClick={() => setActiveTab('inspect')}
            id="nav-inspect-tab"
            className={`px-3.5 py-1.5 text-xs font-medium rounded-lg transition-all flex items-center gap-2 cursor-pointer ${
              activeTab === 'inspect'
                ? 'bg-indigo-50 text-indigo-700 font-semibold shadow-2xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <Layers className="w-4 h-4 text-sky-500" />
            <span>Architecture & Memory</span>
          </button>
        </div>
      </div>
    </header>
  );
};
