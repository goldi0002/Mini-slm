/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Sparkles, 
  Sliders, 
  Trash2, 
  Columns, 
  MessageSquare, 
  Bot, 
  User, 
  Info,
  ChevronDown,
  ChevronUp,
  RefreshCw
} from 'lucide-react';
import { ChatMessage, GenerationOptions, GeneratedTokenInfo } from '../types';
import { SmallLanguageModel } from '../slm/transformer';
import { TokenInspectorModal } from './TokenInspectorModal';

interface ChatPlaygroundProps {
  model: SmallLanguageModel;
  isFinetuned: boolean;
  activeDatasetName?: string;
  /** Builds a fresh, never-fine-tuned model used by the comparison view. */
  createBaseModel: () => SmallLanguageModel;
}

export const ChatPlayground: React.FC<ChatPlaygroundProps> = ({
  model,
  isFinetuned,
  activeDatasetName,
  createBaseModel,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome-1',
      role: 'assistant',
      content: 'Hello! I am your conversational AI assistant. How can I help you today?',
      timestamp: Date.now(),
    },
  ]);

  const [compareMessages, setCompareMessages] = useState<{
    base: ChatMessage[];
    finetuned: ChatMessage[];
  }>({
    base: [
      {
        id: 'base-init',
        role: 'assistant',
        content: 'I am running the baseline pre-trained model weights.',
        timestamp: Date.now(),
        modelSource: 'base',
      },
    ],
    finetuned: [
      {
        id: 'ft-init',
        role: 'assistant',
        content: 'I am running with active fine-tuning adaptation.',
        timestamp: Date.now(),
        modelSource: 'finetuned',
      },
    ],
  });

  const [inputPrompt, setInputPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [comparisonMode, setComparisonMode] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [inspectedToken, setInspectedToken] = useState<GeneratedTokenInfo | null>(null);

  // Generation Hyperparameters
  const [options, setOptions] = useState<GenerationOptions>({
    temperature: 0.7,
    topK: 25,
    topP: 0.85,
    repetitionPenalty: 1.15,
    maxNewTokens: 26,
  });

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Lazily-built base checkpoint used only by the comparison view. A separate
  // instance is what makes "Base" actually mean "never fine-tuned".
  const baseModelRef = useRef<{ key: string; model: SmallLanguageModel } | null>(null);

  const getBaseModel = (): SmallLanguageModel => {
    const key = `${model.config.id}:${model.config.vocabSize}`;
    if (baseModelRef.current?.key !== key) {
      baseModelRef.current = { key, model: createBaseModel() };
    }
    return baseModelRef.current.model;
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, compareMessages, isGenerating]);

  // Handle message submission
  const handleSendMessage = async (textToSend?: string) => {
    const text = (textToSend || inputPrompt).trim();
    if (!text || isGenerating) return;

    setInputPrompt('');
    setIsGenerating(true);

    try {
      await runGeneration(text);
    } finally {
      // Always release the UI, even if a generation step throws.
      setIsGenerating(false);
    }
  };

  /**
   * Runs one generation turn. Split out of the submit handler so the handler
   * can clear the busy flag on every path.
   */
  const runGeneration = async (text: string) => {
    const userMsgId = `user-${Date.now()}`;
    const userMsg: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    if (comparisonMode) {
      // Side-by-side mode: generate for both Base and Fine-tuned
      setCompareMessages((prev) => ({
        base: [...prev.base, { ...userMsg, modelSource: 'base' }],
        finetuned: [...prev.finetuned, { ...userMsg, modelSource: 'finetuned' }],
      }));

      // 1. Generate Fine-tuned response
      const ftId = `ft-${Date.now()}`;
      const ftMsg: ChatMessage = {
        id: ftId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        tokens: [],
        modelSource: 'finetuned',
      };

      setCompareMessages((prev) => ({
        ...prev,
        finetuned: [...prev.finetuned, ftMsg],
      }));

      const prompt = model.tokenizer.formatConversationPrompt(text);
      const ftTokens: GeneratedTokenInfo[] = [];

      for await (const tokenInfo of model.generateStream(prompt, options, true)) {
        ftTokens.push(tokenInfo);
        const decoded = model.tokenizer.decode(
          ftTokens.map((t) => t.id),
          true
        );
        setCompareMessages((prev) => ({
          ...prev,
          finetuned: prev.finetuned.map((m) =>
            m.id === ftId ? { ...m, content: decoded, tokens: [...ftTokens] } : m
          ),
        }));
      }

      // 2. Generate the Base model response from an independent, never-trained
      // instance. Toggling LoRA off on the live model was not a base-model
      // comparison: the statistical memory layer is shared state that
      // fine-tuning mutates, and it supplies most of the sampling mass.
      const baseModel = getBaseModel();
      const basePrompt = baseModel.tokenizer.formatConversationPrompt(text);
      const baseId = `base-${Date.now()}`;
      const baseMsg: ChatMessage = {
        id: baseId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        tokens: [],
        modelSource: 'base',
      };

      setCompareMessages((prev) => ({
        ...prev,
        base: [...prev.base, baseMsg],
      }));

      const baseTokens: GeneratedTokenInfo[] = [];
      for await (const tokenInfo of baseModel.generateStream(basePrompt, options, true)) {
        baseTokens.push(tokenInfo);
        const decoded = baseModel.tokenizer.decode(
          baseTokens.map((t) => t.id),
          true
        );
        setCompareMessages((prev) => ({
          ...prev,
          base: prev.base.map((m) =>
            m.id === baseId ? { ...m, content: decoded, tokens: [...baseTokens] } : m
          ),
        }));
      }
    } else {
      // Standard single chat
      setMessages((prev) => [...prev, userMsg]);

      const assistantId = `assistant-${Date.now()}`;
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        tokens: [],
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // Format prompt with conversational history
      const prompt = model.tokenizer.formatConversationPrompt(
        text,
        messages.slice(-4).map((m) => ({ role: m.role, content: m.content }))
      );

      const collectedTokens: GeneratedTokenInfo[] = [];
      for await (const tokenInfo of model.generateStream(prompt, options, isFinetuned)) {
        collectedTokens.push(tokenInfo);
        const decoded = model.tokenizer.decode(
          collectedTokens.map((t) => t.id),
          true
        );

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: decoded, tokens: [...collectedTokens] }
              : m
          )
        );
      }
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: `clear-${Date.now()}`,
        role: 'assistant',
        content: 'Chat history cleared. How can I help you today?',
        timestamp: Date.now(),
      },
    ]);
    setCompareMessages({
      base: [
        {
          id: `base-c-${Date.now()}`,
          role: 'assistant',
          content: 'Base model ready for comparison.',
          timestamp: Date.now(),
          modelSource: 'base',
        },
      ],
      finetuned: [
        {
          id: `ft-c-${Date.now()}`,
          role: 'assistant',
          content: 'Fine-tuned model ready for comparison.',
          timestamp: Date.now(),
          modelSource: 'finetuned',
        },
      ],
    });
  };

  const samplePrompts = [
    'hello how are you doing today ?',
    'who are you and how can you help me ?',
    'I feel overwhelmed with my daily tasks',
    'can you give me advice on staying focused ?',
    'what makes a good morning routine ?',
    'thank you so much for chatting with me !',
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-8.5rem)] max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
      {/* Top Controls Bar */}
      <div className="bg-white border border-slate-200 rounded-xl p-3 mb-3 shadow-2xs flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setComparisonMode(!comparisonMode)}
            id="toggle-comparison-btn"
            className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition-all flex items-center gap-1.5 ${
              comparisonMode
                ? 'bg-indigo-600 text-white border-indigo-600 shadow-2xs'
                : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
            }`}
          >
            <Columns className="w-3.5 h-3.5" />
            <span>{comparisonMode ? 'Side-by-Side Active' : 'Compare Base vs Fine-Tuned'}</span>
          </button>

          {isFinetuned && (
            <span className="text-xs px-2.5 py-1 bg-emerald-50 text-emerald-700 rounded-lg border border-emerald-200 font-medium flex items-center gap-1">
              <Sparkles className="w-3 h-3" />
              {activeDatasetName ? `Tuned on: ${activeDatasetName}` : 'Fine-tuned Active'}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            id="toggle-settings-btn"
            className={`text-xs font-medium px-2.5 py-1.5 rounded-lg border transition-colors flex items-center gap-1.5 ${
              showSettings
                ? 'bg-slate-100 text-slate-800 border-slate-300'
                : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Sampling Parameters</span>
            {showSettings ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>

          <button
            onClick={clearChat}
            id="clear-chat-btn"
            title="Clear current dialogue history"
            className="text-xs font-medium text-slate-500 hover:text-slate-800 p-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Sampling Parameters Drawer */}
      {showSettings && (
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 mb-3 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 text-xs animate-in slide-in-from-top duration-150">
          <div>
            <div className="flex justify-between font-medium text-slate-700 mb-1">
              <span>Temperature</span>
              <span className="font-mono text-indigo-600">{options.temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.5"
              step="0.05"
              value={options.temperature}
              onChange={(e) => setOptions({ ...options, temperature: parseFloat(e.target.value) })}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <p className="text-[10px] text-slate-400 mt-0.5">Controls randomness/creativity</p>
          </div>

          <div>
            <div className="flex justify-between font-medium text-slate-700 mb-1">
              <span>Top-K</span>
              <span className="font-mono text-indigo-600">{options.topK}</span>
            </div>
            <input
              type="range"
              min="1"
              max="50"
              step="1"
              value={options.topK}
              onChange={(e) => setOptions({ ...options, topK: parseInt(e.target.value) })}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <p className="text-[10px] text-slate-400 mt-0.5">Limits top candidate logits</p>
          </div>

          <div>
            <div className="flex justify-between font-medium text-slate-700 mb-1">
              <span>Top-P (Nucleus)</span>
              <span className="font-mono text-indigo-600">{options.topP.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={options.topP}
              onChange={(e) => setOptions({ ...options, topP: parseFloat(e.target.value) })}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <p className="text-[10px] text-slate-400 mt-0.5">Cumulative probability cutoff</p>
          </div>

          <div>
            <div className="flex justify-between font-medium text-slate-700 mb-1">
              <span>Repetition Penalty</span>
              <span className="font-mono text-indigo-600">{options.repetitionPenalty.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="1.0"
              max="2.0"
              step="0.05"
              value={options.repetitionPenalty}
              onChange={(e) => setOptions({ ...options, repetitionPenalty: parseFloat(e.target.value) })}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <p className="text-[10px] text-slate-400 mt-0.5">Penalizes repeating phrases</p>
          </div>

          <div>
            <div className="flex justify-between font-medium text-slate-700 mb-1">
              <span>Max Tokens</span>
              <span className="font-mono text-indigo-600">{options.maxNewTokens}</span>
            </div>
            <input
              type="range"
              min="10"
              max="48"
              step="2"
              value={options.maxNewTokens}
              onChange={(e) => setOptions({ ...options, maxNewTokens: parseInt(e.target.value) })}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <p className="text-[10px] text-slate-400 mt-0.5">Tokens per generation turn</p>
          </div>
        </div>
      )}

      {/* Main Conversation Container */}
      <div className="flex-1 bg-white border border-slate-200 rounded-xl overflow-y-auto p-4 space-y-4 shadow-2xs">
        {comparisonMode ? (
          /* Side-by-Side Comparison Layout */
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-full">
            {/* Column 1: Base Pretrained Model */}
            <div className="border border-slate-200 rounded-xl p-3 flex flex-col bg-slate-50/50">
              <div className="flex items-center justify-between pb-2 border-b border-slate-200 mb-3">
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-slate-400" />
                  <span className="text-xs font-bold text-slate-700">Base Pretrained Model</span>
                </div>
                <span className="text-[11px] text-slate-400 font-mono">w/o Fine-tuning</span>
              </div>
              <div className="flex-1 space-y-3 overflow-y-auto pr-1">
                {compareMessages.base.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex gap-2.5 text-xs ${
                      msg.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    {msg.role === 'assistant' && (
                      <div className="w-6 h-6 rounded-md bg-slate-200 text-slate-600 flex items-center justify-center shrink-0 mt-0.5">
                        <Bot className="w-3.5 h-3.5" />
                      </div>
                    )}
                    <div
                      className={`rounded-xl px-3 py-2 max-w-[85%] leading-relaxed ${
                        msg.role === 'user'
                          ? 'bg-indigo-600 text-white'
                          : 'bg-white border border-slate-200 text-slate-800'
                      }`}
                    >
                      {msg.content || (isGenerating ? 'Thinking...' : '')}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Column 2: Fine-Tuned Model */}
            <div className="border border-emerald-200 rounded-xl p-3 flex flex-col bg-emerald-50/20">
              <div className="flex items-center justify-between pb-2 border-b border-emerald-100 mb-3">
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-xs font-bold text-emerald-900">Fine-Tuned Model</span>
                </div>
                <span className="text-[11px] text-emerald-600 font-mono">Adapted LoRA Weights</span>
              </div>
              <div className="flex-1 space-y-3 overflow-y-auto pr-1">
                {compareMessages.finetuned.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex gap-2.5 text-xs ${
                      msg.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    {msg.role === 'assistant' && (
                      <div className="w-6 h-6 rounded-md bg-emerald-600 text-white flex items-center justify-center shrink-0 mt-0.5">
                        <Sparkles className="w-3.5 h-3.5" />
                      </div>
                    )}
                    <div
                      className={`rounded-xl px-3 py-2 max-w-[85%] leading-relaxed ${
                        msg.role === 'user'
                          ? 'bg-indigo-600 text-white'
                          : 'bg-white border border-emerald-200 text-slate-800'
                      }`}
                    >
                      {msg.content || (isGenerating ? 'Thinking...' : '')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          /* Standard Single Conversation View */
          <div className="space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-3 text-xs sm:text-sm ${
                  msg.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-xl bg-indigo-100 text-indigo-700 flex items-center justify-center shrink-0 mt-0.5 shadow-2xs">
                    <Bot className="w-4 h-4" />
                  </div>
                )}

                <div
                  className={`rounded-2xl px-4 py-3 max-w-[85%] sm:max-w-[75%] leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-indigo-600 text-white shadow-2xs'
                      : 'bg-slate-50 border border-slate-200 text-slate-800'
                  }`}
                >
                  {/* Assistant response with clickable interactive token chips */}
                  {msg.role === 'assistant' && msg.tokens && msg.tokens.length > 0 ? (
                    <div>
                      <div className="flex flex-wrap gap-1 items-baseline">
                        {msg.tokens.map((tok, i) => (
                          <button
                            key={i}
                            onClick={() => setInspectedToken(tok)}
                            title={`Token: "${tok.token}", ID: ${tok.id}, Prob: ${(tok.prob * 100).toFixed(1)}%. Click to inspect logits.`}
                            className="inline-block px-1 py-0.5 rounded hover:bg-indigo-100/70 text-slate-800 font-sans hover:text-indigo-900 transition-colors cursor-pointer border border-transparent hover:border-indigo-200"
                          >
                            {tok.token === ' ' ? '␣' : tok.token}
                          </button>
                        ))}
                      </div>
                      <div className="mt-2 pt-2 border-t border-slate-200/60 flex items-center justify-between text-[10px] text-slate-400">
                        <span>{msg.tokens.length} tokens generated</span>
                        <span className="flex items-center gap-1 text-indigo-600 font-medium cursor-pointer">
                          <Info className="w-3 h-3" /> Click any token to view logits
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div>
                      {msg.content}
                      {isGenerating && msg.role === 'assistant' && msg.content === '' && (
                        <div className="flex items-center gap-1.5 text-slate-400 py-1">
                          <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
                          <span className="text-xs">Computing logits across layers...</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-xl bg-slate-800 text-white flex items-center justify-center shrink-0 mt-0.5 shadow-2xs">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
        )}
      </div>

      {/* Suggested Starter Prompts */}
      <div className="my-2 flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
        <span className="text-slate-400 text-[11px] shrink-0">Try prompt:</span>
        {samplePrompts.map((p, idx) => (
          <button
            key={idx}
            onClick={() => handleSendMessage(p)}
            disabled={isGenerating}
            className="shrink-0 bg-white hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 text-slate-700 hover:text-indigo-700 px-2.5 py-1 rounded-full text-xs transition-colors disabled:opacity-50"
          >
            {p}
          </button>
        ))}
      </div>

      {/* Input Message Box */}
      <div className="bg-white border border-slate-200 rounded-xl p-2 shadow-2xs">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            id="chat-input"
            value={inputPrompt}
            onChange={(e) => setInputPrompt(e.target.value)}
            placeholder="Type a conversational message to the local model..."
            disabled={isGenerating}
            className="flex-1 bg-slate-50 border-0 focus:outline-hidden focus:ring-1 focus:ring-indigo-500 rounded-lg px-3.5 py-2.5 text-xs sm:text-sm text-slate-900 placeholder:text-slate-400"
          />
          <button
            type="submit"
            id="send-message-btn"
            disabled={!inputPrompt.trim() || isGenerating}
            className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2.5 rounded-lg text-xs sm:text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5 shadow-2xs"
          >
            {isGenerating ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            <span className="hidden sm:inline">Send</span>
          </button>
        </form>
      </div>

      {/* Token Inspector Modal */}
      <TokenInspectorModal
        tokenInfo={inspectedToken}
        onClose={() => setInspectedToken(null)}
      />
    </div>
  );
};
