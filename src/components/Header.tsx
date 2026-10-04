/**
 * @license
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
  <header className="sticky top-0 z-30 border-b border-slate-200/80 bg-white/90 backdrop-blur-xl">
    <div className="mx-auto w-full max-w-[1180px] px-4 sm:px-6 lg:px-8">
      <div className="flex min-h-[68px] flex-wrap items-center justify-between gap-3 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[13px] bg-slate-950 text-white shadow-sm">
            <Bot className="h-5 w-5" strokeWidth={2.1} />
          </div>

          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="truncate text-[15px] font-semibold tracking-[-0.01em] text-slate-950 sm:text-base">
                Mini SLM Studio
              </h1>
              <span className="hidden items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 sm:flex">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                Local
              </span>
              <span className={'flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold ' + (
                isFinetuned
                  ? 'border-indigo-200 bg-indigo-50 text-indigo-700'
                  : 'border-slate-200 bg-slate-50 text-slate-600'
              )}>
                {isFinetuned ? <Sparkles className="h-3 w-3" /> : <Cpu className="h-3 w-3" />}
                {isFinetuned ? 'Fine-tuned' : 'Base model'}
              </span>
            </div>
            <p className="mt-0.5 hidden truncate text-xs text-slate-500 md:block">
              Train, inspect, and chat with a tiny language model in your browser.
            </p>
          </div>
        </div>

        <div className="flex w-full shrink-0 items-center justify-end gap-2 sm:w-auto">
          <div className="hidden items-center gap-1.5 rounded-xl border border-slate-200 bg-slate-50 px-2.5 py-2 text-[11px] text-slate-600 lg:flex">
            <HardDrive className="h-3.5 w-3.5 text-slate-400" />
            <span className="font-medium text-slate-700">{memoryFormatted}</span>
            <span className="text-slate-400">RAM</span>
          </div>

          <label className="sr-only" htmlFor="model-selector">Select model</label>
          <select
            id="model-selector"
            value={currentModel.id}
            onChange={(e) => {
              const model = models.find((item) => item.id === e.target.value);
              if (model) onSelectModel(model);
            }}
            aria-label="Select Small Language Model"
            className="min-w-0 max-w-[170px] flex-1 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800 shadow-sm transition hover:border-slate-300 hover:bg-slate-50 focus:outline-none"
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
              className="hidden h-9 items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-600 transition hover:border-rose-200 hover:bg-rose-50 hover:text-rose-600 sm:flex"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Reset
            </button>
          )}
        </div>
      </div>

      <div className="flex items-center gap-1 overflow-x-auto border-t border-slate-100 py-2 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {navItems.map(({ id, label, icon: Icon }) => {
          const active = activeTab === id;
          return (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              id={'nav-' + id + '-tab'}
              aria-current={active ? 'page' : undefined}
              className={'group flex min-h-9 shrink-0 items-center gap-2 rounded-xl px-3 text-xs font-semibold transition ' + (
                active
                  ? 'bg-slate-950 text-white shadow-sm'
                  : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
              )}
            >
              <Icon className={'h-3.5 w-3.5 ' + (
                active
                  ? 'text-white'
                  : id === 'train'
                    ? 'text-amber-500'
                    : id === 'datasets'
                      ? 'text-emerald-500'
                      : id === 'inspect'
                        ? 'text-sky-500'
                        : 'text-indigo-500'
              )} />
              {label}
            </button>
          );
        })}

        <div className="ml-auto hidden items-center gap-3 pl-4 text-[10px] text-slate-400 lg:flex">
          <span><strong className="font-semibold text-slate-600">{paramStats.total.toLocaleString()}</strong> params</span>
          <span className="h-3 w-px bg-slate-200" />
          <span><strong className="font-semibold text-slate-600">{defaultTokenizer.vocabSize.toLocaleString()}</strong> vocab</span>
          <span className="h-3 w-px bg-slate-200" />
          <span><strong className="font-semibold text-slate-600">{defaultTokenizer.bpeMergeCount}</strong> BPE merges</span>
        </div>
      </div>
    </div>
  </header>
);
