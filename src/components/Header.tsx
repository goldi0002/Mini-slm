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
} from 'lucide-react';
import { ModelConfig } from '../types';
import { defaultTokenizer } from '../slm/tokenizer';

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

const navItems = [
  { id: 'chat' as const, label: 'Chat', icon: Bot },
  { id: 'train' as const, label: 'Fine-tune', icon: Flame },
  { id: 'datasets' as const, label: 'Datasets', icon: Database },
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
}) => (
  <header className="app-header">
    <div className="app-header__inner">
      <div className="app-header__top">
        <div className="app-brand">
          <div className="app-brand__icon" aria-hidden="true">
            <Bot className="h-5 w-5" strokeWidth={2.2} />
          </div>
          <div className="app-brand__copy">
            <div className="flex min-w-0 items-center gap-2">
              <h1 className="truncate text-[15px] font-semibold tracking-tight text-slate-950 sm:text-base">
                Mini SLM Studio
              </h1>
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
          <span><strong>{paramStats.total.toLocaleString()}</strong> params</span>
          <i />
          <span><strong>{defaultTokenizer.vocabSize.toLocaleString()}</strong> vocab</span>
          <i />
          <span><strong>{defaultTokenizer.bpeMergeCount}</strong> BPE</span>
        </div>
      </nav>
    </div>
  </header>
);
