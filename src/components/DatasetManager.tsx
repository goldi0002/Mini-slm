/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { 
  Database, 
  Plus, 
  Trash2, 
  Check, 
  User, 
  Bot, 
  Copy,
  Sparkles,
  Download,
  Upload,
  HardDrive
} from 'lucide-react';
import { ConversationTurn, DatasetPreset } from '../types';
import { generateExpandedChatCorpus } from '../slm/datasets';
import { defaultTokenizer } from '../slm/tokenizer';

interface DatasetManagerProps {
  /** Live dataset state, owned by App so the trainer sees every edit. */
  datasets: DatasetPreset[];
  onDatasetsChange: (datasets: DatasetPreset[]) => void;
  activeDatasetId: string;
  setActiveDatasetId: (id: string) => void;
  onNavigateToTrain: () => void;
}

export const DatasetManager: React.FC<DatasetManagerProps> = ({
  datasets,
  onDatasetsChange,
  activeDatasetId,
  setActiveDatasetId,
  onNavigateToTrain,
}) => {
  const [newUserMsg, setNewUserMsg] = useState('');
  const [newAssistantMsg, setNewAssistantMsg] = useState('');
  const [newCategory, setNewCategory] = useState('Daily Conversation');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [importNotice, setImportNotice] = useState<string | null>(null);

  const currentDataset =
    datasets.find((d) => d.id === activeDatasetId) || datasets[0];

  const handleAddTurn = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newUserMsg.trim() || !newAssistantMsg.trim()) return;

    const newTurn: ConversationTurn = {
      id: `turn-${Date.now()}`,
      user: newUserMsg.trim(),
      assistant: newAssistantMsg.trim(),
      category: newCategory.trim() || 'Conversation',
    };

    onDatasetsChange(
      datasets.map((d) =>
        d.id === currentDataset.id
          ? { ...d, turns: [...d.turns, newTurn] }
          : d
      )
    );

    setNewUserMsg('');
    setNewAssistantMsg('');
  };

  const handleDeleteTurn = (turnId: string) => {
    onDatasetsChange(
      datasets.map((d) =>
        d.id === currentDataset.id
          ? { ...d, turns: d.turns.filter((t) => t.id !== turnId) }
          : d
      )
    );
  };

  const handleCopyTurn = (turn: ConversationTurn) => {
    navigator.clipboard.writeText(`User: ${turn.user}\nAssistant: ${turn.assistant}`);
    setCopiedId(turn.id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  // Expand current dataset into a big dataset of 60 turns
  const handleExpandToBigDataset = () => {
    const expanded = generateExpandedChatCorpus(currentDataset, currentDataset.turns.length + 30);
    onDatasetsChange(
      datasets.map((d) =>
        d.id === currentDataset.id
          ? { ...d, turns: expanded }
          : d
      )
    );
    setImportNotice(`Dataset expanded to ${expanded.length} conversational turns.`);
    setTimeout(() => setImportNotice(null), 3000);
  };

  // Export dataset as JSON
  const handleExportJSON = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentDataset.turns, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", `${currentDataset.id}-chat-dataset.json`);
    dlAnchorElem.click();
  };

  // Handle file import for custom datasets
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        let importedTurns: ConversationTurn[] = [];

        // Check if JSON or JSONL
        if (text.trim().startsWith('[')) {
          const parsed = JSON.parse(text);
          importedTurns = parsed.map((item: any, idx: number) => ({
            id: `imported-${Date.now()}-${idx}`,
            user: item.user || item.prompt || item.input || '',
            assistant: item.assistant || item.response || item.output || '',
            category: item.category || 'Imported'
          })).filter((t: any) => t.user && t.assistant);
        } else {
          // Parse line-by-line JSONL
          const lines = text.split('\n');
          for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            try {
              const item = JSON.parse(line);
              if (item.user && item.assistant) {
                importedTurns.push({
                  id: `imported-${Date.now()}-${i}`,
                  user: item.user,
                  assistant: item.assistant,
                  category: item.category || 'Imported'
                });
              }
            } catch {
              // Ignore single malformed line
            }
          }
        }

        if (importedTurns.length > 0) {
          onDatasetsChange(
            datasets.map((d) =>
              d.id === currentDataset.id
                ? { ...d, turns: [...d.turns, ...importedTurns] }
                : d
            )
          );
          setImportNotice(`Successfully imported ${importedTurns.length} conversational turns.`);
          setTimeout(() => setImportNotice(null), 3000);
        }
      } catch (err) {
        setImportNotice('Failed to parse file. Ensure it is a valid JSON or JSONL format.');
        setTimeout(() => setImportNotice(null), 3000);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 space-y-4">
      {/* Header Banner */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 bg-emerald-50 text-emerald-600 rounded-lg border border-emerald-200">
              <Database className="w-5 h-5" />
            </span>
            <h2 className="text-base font-bold text-slate-900">
              Conversational Chat Datasets
            </h2>
            <span className="text-[11px] bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-md border border-emerald-200 font-medium flex items-center gap-1">
              <HardDrive className="w-3 h-3" />
              Streaming Low-Memory
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Browse, augment, and expand paired User & Assistant dialogues for fine-tuning your local language model.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onNavigateToTrain}
            className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-xs font-semibold shadow-2xs transition-colors cursor-pointer"
          >
            Train on This Dataset
          </button>
        </div>
      </div>

      {importNotice && (
        <div className="bg-indigo-50 border border-indigo-200 text-indigo-800 text-xs px-4 py-2.5 rounded-xl flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-indigo-600" />
          <span>{importNotice}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column: Preset Switcher & Add Form */}
        <div className="space-y-3">
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-2">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                Conversational Personas
              </h3>
            </div>

            {datasets.map((dataset) => {
              const isSelected = dataset.id === currentDataset.id;
              return (
                <button
                  key={dataset.id}
                  onClick={() => setActiveDatasetId(dataset.id)}
                  className={`w-full text-left p-3 rounded-lg border text-xs transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-indigo-50/70 border-indigo-300 ring-1 ring-indigo-500'
                      : 'bg-white border-slate-200 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-slate-900">{dataset.name}</span>
                    <span className="text-[10px] bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded font-medium">
                      {dataset.turns.length} turns
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 line-clamp-2">
                    {dataset.description}
                  </p>
                </button>
              );
            })}
          </div>

          {/* Expand to Big Dataset Action */}
          <div className="bg-gradient-to-br from-indigo-50 to-white border border-indigo-200 rounded-xl p-4 shadow-2xs space-y-2">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-600" />
              <h4 className="text-xs font-bold text-indigo-950">Scale to Big Dataset</h4>
            </div>
            <p className="text-[11px] text-slate-600 leading-relaxed">
              Expand this dataset with +30 synthetic natural conversation pairs to test learning on big datasets with zero memory bloat.
            </p>
            <div className="flex gap-2 pt-1">
              <button
                onClick={handleExpandToBigDataset}
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-2 px-3 rounded-lg text-xs font-semibold shadow-2xs transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Expand with 30+ Chat Turns</span>
              </button>
            </div>
          </div>

          {/* Add New Conversation Pair Form */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-1.5">
              <Plus className="w-3.5 h-3.5 text-indigo-600" />
              <span>Add Conversation Turn</span>
            </h3>

            <form onSubmit={handleAddTurn} className="space-y-3 text-xs">
              <div>
                <label className="block text-slate-600 font-medium mb-1">User Message</label>
                <input
                  type="text"
                  value={newUserMsg}
                  onChange={(e) => setNewUserMsg(e.target.value)}
                  placeholder="e.g. how can I stay calm during a busy day"
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-slate-900 focus:outline-hidden focus:ring-1 focus:ring-indigo-500"
                />
              </div>

              <div>
                <label className="block text-slate-600 font-medium mb-1">Assistant Reply</label>
                <textarea
                  rows={3}
                  value={newAssistantMsg}
                  onChange={(e) => setNewAssistantMsg(e.target.value)}
                  placeholder="e.g. take a slow deep breath and focus on one task at a time ."
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 text-slate-900 focus:outline-hidden focus:ring-1 focus:ring-indigo-500 resize-none"
                />
              </div>

              <div className="flex gap-2">
                <input
                  type="text"
                  value={newCategory}
                  onChange={(e) => setNewCategory(e.target.value)}
                  placeholder="Category (e.g. Advice)"
                  className="w-1/2 bg-slate-50 border border-slate-200 rounded-lg p-2 text-slate-900 focus:outline-hidden focus:ring-1 focus:ring-indigo-500"
                />
                <button
                  type="submit"
                  disabled={!newUserMsg.trim() || !newAssistantMsg.trim()}
                  className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-semibold transition-colors disabled:opacity-50 cursor-pointer"
                >
                  Add Pair
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Right Columns: Active Turns List */}
        <div className="lg:col-span-2 space-y-3">
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs">
            <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-100">
              <div>
                <h3 className="text-sm font-bold text-slate-900">
                  {currentDataset.name}
                </h3>
                <p className="text-xs text-slate-500">{currentDataset.description}</p>
              </div>
              
              <div className="flex items-center gap-2">
                <button
                  onClick={handleExportJSON}
                  title="Export dialogue pairs to JSON"
                  className="p-1.5 text-slate-500 hover:text-indigo-600 hover:bg-slate-100 rounded-lg transition-colors border border-slate-200 flex items-center gap-1 text-[11px] font-medium cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Export JSON</span>
                </button>

                <label
                  title="Import JSON/JSONL chat dataset"
                  className="p-1.5 text-slate-500 hover:text-indigo-600 hover:bg-slate-100 rounded-lg transition-colors border border-slate-200 flex items-center gap-1 text-[11px] font-medium cursor-pointer"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Import JSONL</span>
                  <input
                    type="file"
                    accept=".json,.jsonl,.txt"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </label>

                <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded">
                  {currentDataset.turns.length} examples
                </span>
              </div>
            </div>

            <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
              {currentDataset.turns.map((turn, idx) => {
                const userTokens = defaultTokenizer.encode(turn.user, false, false).length;
                const assistantTokens = defaultTokenizer.encode(turn.assistant, false, false).length;

                return (
                  <div
                    key={turn.id}
                    className="p-3.5 rounded-xl border border-slate-200 bg-slate-50/50 space-y-2 text-xs"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[10px] text-slate-400">#{idx + 1}</span>
                        {turn.category && (
                          <span className="text-[10px] bg-slate-200/70 text-slate-700 font-medium px-2 py-0.5 rounded-full">
                            {turn.category}
                          </span>
                        )}
                        <span className="text-[10px] text-slate-400 font-mono">
                          {userTokens + assistantTokens} tokens
                        </span>
                      </div>

                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => handleCopyTurn(turn)}
                          title="Copy dialogue pair"
                          className="p-1 text-slate-400 hover:text-slate-600 rounded hover:bg-slate-200 transition-colors cursor-pointer"
                        >
                          {copiedId === turn.id ? (
                            <Check className="w-3.5 h-3.5 text-emerald-600" />
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                        </button>
                        <button
                          onClick={() => handleDeleteTurn(turn.id)}
                          title="Delete turn"
                          className="p-1 text-slate-400 hover:text-rose-600 rounded hover:bg-rose-50 transition-colors cursor-pointer"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>

                    {/* User message */}
                    <div className="flex items-start gap-2 bg-white border border-slate-200/80 rounded-lg p-2.5">
                      <div className="w-5 h-5 rounded bg-slate-800 text-white flex items-center justify-center shrink-0 mt-0.5">
                        <User className="w-3 h-3" />
                      </div>
                      <div className="flex-1 text-slate-800 font-medium">
                        {turn.user}
                      </div>
                    </div>

                    {/* Assistant message */}
                    <div className="flex items-start gap-2 bg-indigo-50/40 border border-indigo-100 rounded-lg p-2.5">
                      <div className="w-5 h-5 rounded bg-indigo-600 text-white flex items-center justify-center shrink-0 mt-0.5">
                        <Bot className="w-3 h-3" />
                      </div>
                      <div className="flex-1 text-indigo-950 font-normal leading-relaxed">
                        {turn.assistant}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
