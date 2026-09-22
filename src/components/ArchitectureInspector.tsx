/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useMemo, useState } from 'react';
import { 
  Layers, 
  Cpu, 
  Eye, 
  Download, 
  Check, 
  Activity,
  Grid,
  HardDrive,
  ShieldCheck
} from 'lucide-react';
import { SmallLanguageModel } from '../slm/transformer';

interface ArchitectureInspectorProps {
  model: SmallLanguageModel;
  isFinetuned: boolean;
}

export const ArchitectureInspector: React.FC<ArchitectureInspectorProps> = ({
  model,
  isFinetuned,
}) => {
  const { config, weights } = model;
  const paramStats = model.countParameters();
  const memStats = model.getMemoryStats();

  const [testPrompt, setTestPrompt] = useState('User: hello how are you Assistant: I am doing well');
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [exportSuccess, setExportSuccess] = useState(false);

  const tokens = useMemo(
    () => model.tokenizer.encode(testPrompt, true, false),
    [model, testPrompt]
  );

  const tokenLabels = useMemo(
    () => tokens.map((id) => model.tokenizer.getTokenString(id)),
    [model, tokens]
  );

  // A different model may have fewer layers or heads than the current
  // selection, which would leave the heatmap showing an empty, masked grid.
  useEffect(() => {
    setSelectedLayer(0);
    setSelectedHead(0);
  }, [model]);

  // The forward pass mutates the model's scratch buffers and attention maps, so
  // it is a side effect: run it after render, keyed on its real inputs, instead
  // of recomputing it during every render (including unrelated ones such as
  // selecting a different layer or head).
  const [attentionMaps, setAttentionMaps] = useState<number[][][][]>([]);

  useEffect(() => {
    model.forward(tokens, isFinetuned);
    setAttentionMaps(model.lastAttentionMaps);
  }, [model, tokens, isFinetuned]);

  // Retrieve attention map for current layer & head
  const currentAttnMatrix = attentionMaps[selectedLayer]?.[selectedHead] || [];

  // Export weights as JSON file
  const handleExportWeights = () => {
    const exportData = {
      config: model.config,
      wte: Array.from(weights.wte),
      wpe: Array.from(weights.wpe),
      lm_head: Array.from(weights.lm_head),
      layers: weights.layers.map((l) => ({
        q_proj: Array.from(l.q_proj),
        k_proj: Array.from(l.k_proj),
        v_proj: Array.from(l.v_proj),
        out_proj: Array.from(l.out_proj),
        lora_q_A: Array.from(l.lora_q_A),
        lora_q_B: Array.from(l.lora_q_B),
        lora_v_A: Array.from(l.lora_v_A),
        lora_v_B: Array.from(l.lora_v_B),
        fc1: Array.from(l.fc1),
        fc2: Array.from(l.fc2),
      })),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${config.id}-weights.json`;
    a.click();
    URL.revokeObjectURL(url);

    setExportSuccess(true);
    setTimeout(() => setExportSuccess(false), 2500);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 space-y-4">
      {/* Overview Banner */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 bg-sky-50 text-sky-600 rounded-lg border border-sky-200">
              <Layers className="w-5 h-5" />
            </span>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-slate-900">
                  SLM Architecture & Low-Memory Footprint
                </h2>
                <span className="text-[11px] bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-md border border-emerald-200 font-medium flex items-center gap-1">
                  <HardDrive className="w-3 h-3" />
                  {memStats.totalMemoryFormatted} RAM
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-0.5">
                Inspect lightweight causal transformer layers, pre-allocated scratch buffers, and multi-head attention maps.
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleExportWeights}
            id="export-weights-btn"
            className="text-xs font-semibold px-3 py-2 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 flex items-center gap-1.5 transition-colors shadow-2xs cursor-pointer"
          >
            {exportSuccess ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-emerald-700">Weights Exported</span>
              </>
            ) : (
              <>
                <Download className="w-3.5 h-3.5" />
                <span>Export Weights JSON</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Memory Efficiency & Scratch Buffers Card */}
      <div className="bg-gradient-to-r from-slate-900 to-indigo-950 text-white rounded-xl p-4 shadow-2xs">
        <div className="flex items-center justify-between mb-3 border-b border-white/10 pb-2">
          <div className="flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-emerald-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-white">
              Flat-Memory In-Browser Allocation Engine
            </h3>
          </div>
          <span className="text-xs font-mono text-emerald-300 font-semibold">
            Zero GC Thrashing • Pre-allocated Float32
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="bg-white/5 border border-white/10 rounded-lg p-2.5">
            <span className="text-slate-400 text-[11px] block">Total Model RAM</span>
            <span className="text-lg font-bold font-mono text-emerald-400 mt-0.5 block">
              {memStats.totalMemoryFormatted}
            </span>
            <span className="text-[10px] text-slate-400">Fits in any browser</span>
          </div>

          <div className="bg-white/5 border border-white/10 rounded-lg p-2.5">
            <span className="text-slate-400 text-[11px] block">Model Weights Array</span>
            <span className="text-lg font-bold font-mono text-indigo-300 mt-0.5 block">
              {memStats.weightsMemoryFormatted}
            </span>
            <span className="text-[10px] text-slate-400">Single flat Float32 snapshot</span>
          </div>

          <div className="bg-white/5 border border-white/10 rounded-lg p-2.5">
            <span className="text-slate-400 text-[11px] block">Scratch Buffers</span>
            <span className="text-lg font-bold font-mono text-sky-300 mt-0.5 block">
              {memStats.scratchBuffersFormatted}
            </span>
            <span className="text-[10px] text-slate-400">Reused across all tokens</span>
          </div>

          <div className="bg-white/5 border border-white/10 rounded-lg p-2.5">
            <span className="text-slate-400 text-[11px] block">Vocabulary Density</span>
            <span className="text-lg font-bold font-mono text-amber-300 mt-0.5 block">
              {config.vocabSize} tokens
            </span>
            <span className="text-[10px] text-slate-400">Pure conversational lexicon</span>
          </div>
        </div>
      </div>

      {/* Model Spec & Parameters Breakdown */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
        <div className="bg-white border border-slate-200 rounded-xl p-3 shadow-2xs">
          <span className="text-[11px] text-slate-400 font-medium">Vocabulary</span>
          <div className="text-lg font-bold font-mono text-slate-900 mt-0.5">
            {config.vocabSize}
          </div>
          <span className="text-[10px] text-slate-500">Conversational tokens</span>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-3 shadow-2xs">
          <span className="text-[11px] text-slate-400 font-medium">Embedding Dim</span>
          <div className="text-lg font-bold font-mono text-indigo-600 mt-0.5">
            d = {config.dModel}
          </div>
          <span className="text-[10px] text-slate-500">Vector representation</span>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-3 shadow-2xs">
          <span className="text-[11px] text-slate-400 font-medium">Attention Heads</span>
          <div className="text-lg font-bold font-mono text-slate-900 mt-0.5">
            {config.nHeads} heads
          </div>
          <span className="text-[10px] text-slate-500">
            {config.dModel / config.nHeads} dim per head
          </span>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-3 shadow-2xs">
          <span className="text-[11px] text-slate-400 font-medium">Transformer Layers</span>
          <div className="text-lg font-bold font-mono text-slate-900 mt-0.5">
            {config.nLayers}
          </div>
          <span className="text-[10px] text-slate-500">Self-attention blocks</span>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-3 shadow-2xs">
          <span className="text-[11px] text-slate-400 font-medium">Context Window</span>
          <div className="text-lg font-bold font-mono text-slate-900 mt-0.5">
            {config.maxSeqLen} tokens
          </div>
          <span className="text-[10px] text-slate-500">Max sequence length</span>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-3 shadow-2xs">
          <span className="text-[11px] text-slate-400 font-medium">Total Parameters</span>
          <div className="text-lg font-bold font-mono text-emerald-600 mt-0.5">
            {paramStats.total.toLocaleString()}
          </div>
          <span className="text-[10px] text-slate-500">
            {paramStats.loraOnly.toLocaleString()} LoRA params
          </span>
        </div>
      </div>

      {/* Attention Heatmap Inspector */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <Grid className="w-4 h-4 text-indigo-600" />
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              Multi-Head Attention Heatmap Visualizer
            </h3>
          </div>

          {/* Layer and Head Selector */}
          <div className="flex items-center gap-2 text-xs">
            <div className="flex items-center gap-1 bg-slate-50 border border-slate-200 rounded-lg p-1">
              <span className="text-slate-500 text-[11px] px-1 font-medium">Layer:</span>
              {Array.from({ length: config.nLayers }).map((_, l) => (
                <button
                  key={l}
                  onClick={() => setSelectedLayer(l)}
                  className={`px-2 py-0.5 rounded font-mono font-medium cursor-pointer ${
                    selectedLayer === l
                      ? 'bg-indigo-600 text-white shadow-2xs'
                      : 'text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  L{l + 1}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-1 bg-slate-50 border border-slate-200 rounded-lg p-1">
              <span className="text-slate-500 text-[11px] px-1 font-medium">Head:</span>
              {Array.from({ length: config.nHeads }).map((_, h) => (
                <button
                  key={h}
                  onClick={() => setSelectedHead(h)}
                  className={`px-2 py-0.5 rounded font-mono font-medium cursor-pointer ${
                    selectedHead === h
                      ? 'bg-indigo-600 text-white shadow-2xs'
                      : 'text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  H{h + 1}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Prompt Input for Attention Map */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-500 font-medium shrink-0">Test Sentence:</span>
          <input
            type="text"
            value={testPrompt}
            onChange={(e) => setTestPrompt(e.target.value)}
            className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 text-slate-800 focus:outline-hidden focus:ring-1 focus:ring-indigo-500"
          />
        </div>

        {/* Matrix Visualization */}
        <div className="overflow-x-auto p-2 bg-slate-50/50 rounded-xl border border-slate-200">
          <div className="inline-block min-w-full">
            {/* Horizontal Header (Keys) */}
            <div className="flex pl-20 pb-2 text-[10px] font-mono border-b border-slate-200 mb-2">
              <div className="flex gap-1">
                {tokenLabels.map((tok, j) => (
                  <span
                    key={j}
                    className="w-10 text-center truncate font-medium text-slate-600"
                    title={`Key Token: ${tok}`}
                  >
                    {tok === ' ' ? '␣' : tok}
                  </span>
                ))}
              </div>
            </div>

            <div className="space-y-1">
              {tokenLabels.map((qTok, i) => (
                <div key={i} className="flex items-center text-xs">
                  <span
                    className="w-20 shrink-0 text-right pr-2 font-mono text-[11px] truncate text-slate-700 font-medium"
                    title={`Query Token: ${qTok}`}
                  >
                    {qTok === ' ' ? '␣' : qTok}
                  </span>

                  <div className="flex gap-1">
                    {tokenLabels.map((_, j) => {
                      const isCausalMasked = j > i;
                      const weight = isCausalMasked ? 0 : currentAttnMatrix[i]?.[j] ?? 0;
                      const opacity = isCausalMasked ? 0.05 : Math.max(0.1, weight);

                      return (
                        <div
                          key={j}
                          title={
                            isCausalMasked
                              ? 'Causal masked (future token)'
                              : `Attn(${qTok} -> ${tokenLabels[j]}): ${(weight * 100).toFixed(1)}%`
                          }
                          className="w-10 h-8 rounded flex items-center justify-center font-mono text-[9px] cursor-pointer transition-transform hover:scale-110"
                          style={{
                            backgroundColor: isCausalMasked
                              ? '#f1f5f9'
                              : `rgba(79, 70, 229, ${opacity})`,
                            color: weight > 0.4 ? '#ffffff' : '#334155',
                          }}
                        >
                          {isCausalMasked ? '—' : (weight * 100).toFixed(0) + '%'}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <p className="text-[11px] text-slate-500 italic">
          * Causal self-attention mask enforces that each token position only attends to previous tokens in the conversational prompt.
        </p>
      </div>

      {/* Network Pipeline Schematic */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs">
        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <Activity className="w-4 h-4 text-indigo-600" />
          <span>Autoregressive Transformer Layer Flow</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="font-bold text-slate-900 mb-1">1. Embedding Layer</div>
            <p className="text-[11px] text-slate-500 mb-2">
              Sum of Token Embedding `WTE` and Learned Positional Embedding `WPE`.
            </p>
            <div className="font-mono text-[10px] text-slate-400">
              Input: [seq_len] → [seq_len, {config.dModel}]
            </div>
          </div>

          <div className="p-3 bg-indigo-50/60 border border-indigo-200 rounded-lg">
            <div className="font-bold text-indigo-950 mb-1">2. Multi-Head Attention</div>
            <p className="text-[11px] text-indigo-900/80 mb-2">
              LayerNorm 1 + Q, K, V projections + LoRA adapters + Causal Mask + Softmax.
            </p>
            <div className="font-mono text-[10px] text-indigo-700">
              {config.nHeads} parallel attention heads
            </div>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="font-bold text-slate-900 mb-1">3. Feed-Forward MLP</div>
            <p className="text-[11px] text-slate-500 mb-2">
              LayerNorm 2 + Linear(d→{config.dFfn}) + GELU Activation + Linear({config.dFfn}→d).
            </p>
            <div className="font-mono text-[10px] text-slate-400">
              Residual stream addition
            </div>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="font-bold text-slate-900 mb-1">4. Head & Logits</div>
            <p className="text-[11px] text-slate-500 mb-2">
              Final LayerNorm `LN_f` + Projection to {config.vocabSize} vocabulary logits.
            </p>
            <div className="font-mono text-[10px] text-slate-400">
              Output: Softmax distribution
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
