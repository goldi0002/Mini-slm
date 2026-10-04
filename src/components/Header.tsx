/** @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import {
  Bot,
  Database,
  HardDrive,
  Layers,
  RotateCcw,
  Sparkles,
  Flame,
  Cpu,
  BookOpen,
} from 'lucide-react';
import { ModelConfig } from '../types';
import { defaultTokenizer } from '../slm/tokenizer';

interface HeaderProps {
  models: ModelConfig[];
  currentModel: ModelConfig;
  onSelectModel: (model: ModelConfig) => void;
  activeTab: 'chat' | 'train' | 'inspect' | 'datasets' | 'knowledge';
  setActiveTab: (tab: 'chat' | 'train' | 'inspect' | 'datasets' | 'knowledge') => void;
  isFinetuned: boolean;
  onResetToBase: () => void;
  paramStats: { total: number; trainable: number; loraOnly: number };
  memoryFormatted?: string;
  knowledgeCount?: number;
}

const navItems = [
  { id: 'chat' as const, label: 'Chat', icon: Bot },
  { id: 'train' as const, label: 'Fine-tune', icon: Flame },
  { id: 'datasets' as const, label: 'Datasets', icon: Database },
  { id: 'knowledge' as const, label: 'Knowledge', icon: BookOpen },
  { id: 'inspect' as const, label: 'Inspect', icon: Layers },
];

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
  knowledgeCount = 0,
}) => (
  <header className="app-header">
    <div className="app-header__inner">
      <div className="app-header__top">
        <div className="app-brand">
          <div className="app-brand__icon" aria-hidden="true">
            <Bot className="h-5 w-5" strokeWidth={2.2} />
          </div>
          <div className="app-brand__copy">
            <div className="app-brand__title-row">
              <h1>Mini SLM Studio</h1>
              <span className="app-badge app-badge--local">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                Local
              </span>
              <span className={'app-badge ' + (isFinetuned ? 'app-badge--tuned' : 'app-badge--base')}>
                {isFinetuned ? <Sparkles className="h-3 w-3" /> : <Cpu className="h-3 w-3" />}
                {isFinetuned ? 'Fine-tuned' : 'Base model'}
              </span>
            </div>
            <p className="app-brand__description">
              Train, inspect, and chat with a tiny language model in your browser.
            </p>
          </div>
        </div>

        <div className="app-header__actions">
          <div className="app-runtime-stat" title="Current model memory footprint">
            <HardDrive className="h-3.5 w-3.5" />
            <strong>{memoryFormatted}</strong>
            <span>RAM</span>
          </div>

          <label className="sr-only" htmlFor="model-selector">Select model</label>
          <select
            id="model-selector"
            value={currentModel.id}
            onChange={(e) => {
              const selected = models.find((item) => item.id === e.target.value);
              if (selected) onSelectModel(selected);
            }}
            aria-label="Select Small Language Model"
            className="app-model-select"
          >
            {models.map((model) => (
              <option key={model.id} value={model.id}>{model.name}</option>
            ))}
          </select>

          {isFinetuned && (
            <button
              onClick={onResetToBase}
              id="reset-base-btn"
              title="Revert model weights back to base pretrained state"
              aria-label="Reset model weights"
              className="app-reset"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Reset</span>
            </button>
          )}
        </div>
      </div>

      <div className="app-model-row">
        <span className="app-model-row__label">Model</span>
        <select
          id="mobile-model-selector"
          value={currentModel.id}
          onChange={(e) => {
            const selected = models.find((item) => item.id === e.target.value);
            if (selected) onSelectModel(selected);
          }}
          aria-label="Select model"
          className="app-model-select app-model-select--mobile"
        >
          {models.map((model) => (
            <option key={model.id} value={model.id}>{model.name}</option>
          ))}
        </select>
        <span className="app-model-row__meta">
          {(currentModel.displayParameterCount ?? paramStats.total).toLocaleString()} params · {currentModel.displayMemory ?? memoryFormatted}
        </span>
      </div>

      <nav className="app-nav" aria-label="Studio sections">
        <div className="app-nav__items">
          {navItems.map(({ id, label, icon: Icon }) => {
            const active = activeTab === id;
            return (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                id={'nav-' + id + '-tab'}
                aria-current={active ? 'page' : undefined}
                className={'app-nav__item ' + (active ? 'app-nav__item--active' : '')}
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </button>
            );
          })}
        </div>

        <div className="app-nav__metrics" aria-label="Model metrics">
          <span><strong>{(currentModel.displayParameterCount ?? paramStats.total).toLocaleString()}</strong> params</span>
          <i />
          <span><strong>{defaultTokenizer.vocabSize.toLocaleString()}</strong> vocab</span>
          <i />
          <span><strong>{defaultTokenizer.bpeMergeCount}</strong> BPE</span>
          <i />
          <span><strong>{knowledgeCount}</strong> knowledge</span>
        </div>
      </nav>
    </div>
  </header>
);
