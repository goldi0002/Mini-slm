/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { X, Sparkles, Hash, BarChart3 } from 'lucide-react';
import { GeneratedTokenInfo } from '../types';

interface TokenInspectorModalProps {
  tokenInfo: GeneratedTokenInfo | null;
  onClose: () => void;
}

export const TokenInspectorModal: React.FC<TokenInspectorModalProps> = ({ tokenInfo, onClose }) => {
  if (!tokenInfo) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4 animate-in fade-in duration-150"
      onClick={onClose}
      role="presentation"
    >
      <div 
        className="bg-white rounded-2xl shadow-xl border border-slate-200 w-full max-w-md overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-100 text-indigo-700 flex items-center justify-center font-mono font-bold text-sm">
              <Hash className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">Token Logits & Probabilities</h3>
              <p className="text-xs text-slate-500">Inspecting SLM output projection layer</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 p-1.5 rounded-lg hover:bg-slate-200 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          {/* Selected Token Card */}
          <div className="bg-indigo-50/60 border border-indigo-100 rounded-xl p-3.5">
            <div className="text-xs text-indigo-700 font-semibold uppercase tracking-wider mb-1">
              Sampled Token
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-mono text-base font-bold bg-white px-2.5 py-1 rounded-lg border border-indigo-200 text-indigo-900">
                  {tokenInfo.token === ' ' ? '␣ [space]' : tokenInfo.token}
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  ID: {tokenInfo.id}
                </span>
              </div>
              <div className="text-right">
                <div className="text-sm font-bold text-indigo-700">
                  {(tokenInfo.prob * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] text-indigo-500">likelihood</div>
              </div>
            </div>
          </div>

          {/* Candidate Logits Table */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
                <BarChart3 className="w-3.5 h-3.5 text-indigo-600" />
                <span>Top Candidate Predictions</span>
              </div>
              <span className="text-[11px] text-slate-400">Softmax distribution</span>
            </div>

            <div className="space-y-2">
              {tokenInfo.topCandidates.map((cand, idx) => {
                const isSelected = cand.id === tokenInfo.id;
                const percentage = Math.min(100, Math.max(0.5, cand.prob * 100));

                return (
                  <div
                    key={cand.id}
                    className={`p-2.5 rounded-lg border text-xs transition-colors ${
                      isSelected
                        ? 'bg-indigo-50/50 border-indigo-200 text-indigo-950 font-medium'
                        : 'bg-white border-slate-100 text-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2">
                        <span className="w-4 text-slate-400 font-mono text-[10px]">
                          #{idx + 1}
                        </span>
                        <span className="font-mono bg-slate-100 px-1.5 py-0.5 rounded text-slate-800">
                          {cand.token === ' ' ? '␣' : cand.token}
                        </span>
                        <span className="text-[10px] text-slate-400 font-mono">
                          ID: {cand.id}
                        </span>
                        {isSelected && (
                          <span className="text-[10px] bg-indigo-600 text-white px-1.5 py-0.2 rounded-full font-semibold">
                            Chosen
                          </span>
                        )}
                      </div>
                      <span className="font-mono font-semibold text-slate-800">
                        {(cand.prob * 100).toFixed(1)}%
                      </span>
                    </div>

                    {/* Probability Bar */}
                    <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          isSelected ? 'bg-indigo-600' : 'bg-slate-400'
                        }`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="p-3 bg-slate-50 border-t border-slate-100 text-center">
          <p className="text-[11px] text-slate-500">
            Click any token chip in the assistant response to inspect its live logits
          </p>
        </div>
      </div>
    </div>
  );
};
