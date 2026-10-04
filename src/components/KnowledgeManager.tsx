/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useState } from 'react';
import { BookOpen, FileText, Trash2, Upload, Plus, Database, Search } from 'lucide-react';
import { LocalKnowledgeBase, KnowledgeDocument } from '../slm/knowledge';

interface KnowledgeManagerProps {
  knowledgeBase: LocalKnowledgeBase;
  onChanged: () => void;
}

export const KnowledgeManager: React.FC<KnowledgeManagerProps> = ({ knowledgeBase, onChanged }) => {
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [name, setName] = useState('');
  const [content, setContent] = useState('');
  const [query, setQuery] = useState('');
  const [preview, setPreview] = useState<{ name: string; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');

  const refresh = () => setDocuments(knowledgeBase.listDocuments());

  useEffect(() => {
    refresh();
  }, [knowledgeBase]);

  const addKnowledge = async () => {
    if (!content.trim() || busy) return;
    setBusy(true);
    setStatus('');
    try {
      await knowledgeBase.addDocument(name.trim() || 'Untitled knowledge', content);
      setName('');
      setContent('');
      refresh();
      onChanged();
      setStatus('Saved locally. The knowledge is available to chat immediately.');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Could not save knowledge.');
    } finally {
      setBusy(false);
    }
  };

  const importFile = async (file: File) => {
    setBusy(true);
    setStatus('');
    try {
      const text = await file.text();
      const looksJson = file.name.toLowerCase().endsWith('.json') || file.name.toLowerCase().endsWith('.jsonl');
      let imported = text;
      if (looksJson) {
        try {
          const parsed = JSON.parse(text);
          imported = typeof parsed === 'string' ? parsed : JSON.stringify(parsed, null, 2);
        } catch {
          imported = text;
        }
      }
      await knowledgeBase.addDocument(file.name, imported);
      refresh();
      onChanged();
      setStatus(`Imported “${file.name}” locally.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Could not import the file.');
    } finally {
      setBusy(false);
    }
  };

  const remove = async (doc: KnowledgeDocument) => {
    await knowledgeBase.deleteDocument(doc.id);
    if (preview?.name === doc.name) setPreview(null);
    refresh();
    onChanged();
  };

  const filtered = documents.filter((doc) => {
    const q = query.trim().toLowerCase();
    return !q || doc.name.toLowerCase().includes(q) || doc.content.toLowerCase().includes(q);
  });

  return (
    <section className="w-full min-h-full p-4 sm:p-5">
      <div className="max-w-6xl mx-auto space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-indigo-600 text-[10px] font-extrabold tracking-[0.14em]">
              <BookOpen className="w-3.5 h-3.5" />
              LOCAL KNOWLEDGE
            </div>
            <h2 className="mt-1 text-xl sm:text-2xl font-bold text-slate-900">Teach the assistant what your project knows</h2>
            <p className="mt-1 max-w-2xl text-xs sm:text-sm leading-6 text-slate-500">
              Add product docs, notes, FAQs, specifications, or any other text. MiniSLM retrieves the relevant pieces during chat instead of retraining the model.
            </p>
          </div>
          <div className="flex items-center gap-2 text-[11px] text-slate-500">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full border border-emerald-200 bg-emerald-50 text-emerald-700 font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              Browser-local
            </span>
            <span>{documents.length} source{documents.length === 1 ? '' : 's'} · {knowledgeBase.chunkCount()} chunks</span>
          </div>
        </div>

        <div className="grid lg:grid-cols-[minmax(0,1.1fr)_minmax(320px,.9fr)] gap-4">
          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-4">
            <div className="flex items-center justify-between gap-3 mb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900">Add knowledge</h3>
                <p className="text-[11px] text-slate-500">Stored in IndexedDB on this device.</p>
              </div>
              <label className="inline-flex items-center gap-1.5 px-2.5 py-2 rounded-lg border border-slate-200 bg-slate-50 hover:bg-slate-100 text-[11px] font-semibold text-slate-700 cursor-pointer">
                <Upload className="w-3.5 h-3.5" />
                Import file
                <input
                  type="file"
                  accept=".txt,.md,.json,.jsonl,.csv"
                  className="hidden"
                  disabled={busy}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void importFile(file);
                    e.currentTarget.value = '';
                  }}
                />
              </label>
            </div>

            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Source name, e.g. Product Manual"
              className="w-full mb-2 px-3 py-2.5 rounded-lg border border-slate-200 bg-slate-50 text-xs text-slate-900 outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-400"
            />
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder={'Paste your domain knowledge here...\n\nExample:\nThe X1 router supports Wi-Fi 6 and WPA3. To reset it, hold the reset button for 10 seconds.'}
              className="w-full min-h-[300px] resize-y px-3 py-3 rounded-lg border border-slate-200 bg-slate-50 text-xs leading-6 text-slate-900 outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-400"
            />
            <div className="mt-3 flex items-center justify-between gap-3">
              <span className="text-[10px] text-slate-400">{content.length.toLocaleString()} characters</span>
              <button
                onClick={() => void addKnowledge()}
                disabled={!content.trim() || busy}
                className="inline-flex items-center gap-1.5 px-3.5 py-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-white text-[11px] font-bold disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Plus className="w-3.5 h-3.5" />
                {busy ? 'Saving…' : 'Add to knowledge'}
              </button>
            </div>
            {status && <p className="mt-3 text-[11px] text-emerald-700">{status}</p>}
          </div>

          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-4 min-h-[420px]">
            <div className="flex items-center gap-2 mb-3">
              <Database className="w-4 h-4 text-indigo-600" />
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-bold text-slate-900">Knowledge sources</h3>
                <p className="text-[11px] text-slate-500">Retrieved automatically when a question matches.</p>
              </div>
            </div>

            <div className="relative mb-3">
              <Search className="absolute left-2.5 top-2.5 w-3.5 h-3.5 text-slate-400" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter sources"
                className="w-full pl-8 pr-3 py-2 rounded-lg border border-slate-200 bg-slate-50 text-xs outline-none focus:ring-2 focus:ring-indigo-100"
              />
            </div>

            {filtered.length === 0 ? (
              <div className="h-64 grid place-items-center text-center text-slate-400">
                <div>
                  <FileText className="w-7 h-7 mx-auto mb-2 opacity-50" />
                  <p className="text-xs font-semibold text-slate-500">No knowledge sources yet</p>
                  <p className="mt-1 text-[10px]">Add text or import a .txt, .md, .json, .jsonl, or .csv file.</p>
                </div>
              </div>
            ) : (
              <div className="space-y-2 max-h-[470px] overflow-y-auto pr-1">
                {filtered.map((doc) => (
                  <div key={doc.id} className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
                    <div className="flex items-start gap-2">
                      <div className="w-8 h-8 shrink-0 grid place-items-center rounded-lg bg-indigo-50 text-indigo-600">
                        <FileText className="w-4 h-4" />
                      </div>
                      <button
                        onClick={() => setPreview(preview?.name === doc.name ? null : { name: doc.name, text: doc.content })}
                        className="min-w-0 flex-1 text-left"
                      >
                        <div className="truncate text-xs font-bold text-slate-800">{doc.name}</div>
                        <div className="mt-0.5 text-[10px] text-slate-400">{doc.chunkCount} chunks · {doc.content.length.toLocaleString()} chars</div>
                      </button>
                      <button
                        onClick={() => void remove(doc)}
                        title="Remove knowledge source"
                        className="p-1.5 rounded-md text-slate-400 hover:text-rose-600 hover:bg-rose-50"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    {preview?.name === doc.name && (
                      <pre className="mt-3 max-h-52 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-white border border-slate-200 p-3 text-[10px] leading-5 text-slate-600">
                        {preview.text}
                      </pre>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="grid sm:grid-cols-3 gap-3">
          {[
            ['1', 'Add', 'Paste or import your domain information.'],
            ['2', 'Retrieve', 'Relevant chunks are selected for each question.'],
            ['3', 'Answer', 'The local model answers using that context.'],
          ].map(([n, title, body]) => (
            <div key={n} className="rounded-xl border border-slate-200 bg-white p-3">
              <div className="text-[10px] font-black text-indigo-600">STEP {n}</div>
              <div className="mt-1 text-xs font-bold text-slate-800">{title}</div>
              <div className="mt-1 text-[10px] leading-5 text-slate-500">{body}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
