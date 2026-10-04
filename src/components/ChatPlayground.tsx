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
import { defaultTokenizer } from '../slm/tokenizer';
import { LocalKnowledgeBase } from '../slm/knowledge';
import { TokenInspectorModal } from './TokenInspectorModal';

/** Shown when a generation turn produced no tokens at all. */
const EMPTY_REPLY_NOTICE =
  'I could not produce a reply for that one — try rephrasing it, or fine-tune me on a dataset that covers it.';

/** Never leave a blank assistant bubble: say so instead. */
function finishReply<T extends ChatMessage>(list: T[], id: string): T[] {
  return list.map((m) =>
    m.id === id && m.content.trim() === '' ? { ...m, content: EMPTY_REPLY_NOTICE } : m
  );
}

interface ChatPlaygroundProps {
  model: SmallLanguageModel;
  isFinetuned: boolean;
  activeDatasetName?: string;
  /** Builds a fresh, never-fine-tuned model used by the comparison view. */
  createBaseModel: () => SmallLanguageModel;
  knowledgeBase: LocalKnowledgeBase;
  knowledgeReady: boolean;
  knowledgeVersion?: number;
}

export const ChatPlayground: React.FC<ChatPlaygroundProps> = ({
  model,
  isFinetuned,
  activeDatasetName,
  createBaseModel,
  knowledgeBase,
  knowledgeReady,
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
  // Id of the message whose token stream is expanded for inspection.
  const [tokenStreamFor, setTokenStreamFor] = useState<string | null>(null);

  // Generation Hyperparameters
  const [options, setOptions] = useState<GenerationOptions>({
    // Conservative defaults keep the small local model on-topic. Users can
    // increase temperature from Sampling when they want more variety.
    temperature: 0.2,
    topK: 8,
    topP: 0.9,
    repetitionPenalty: 1.12,
    maxNewTokens: 32,
  });

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Lazily-built base checkpoint used only by the comparison view. A separate
  // instance is what makes "Base" actually mean "never fine-tuned".
  const baseModelRef = useRef<{ key: string; model: SmallLanguageModel } | null>(null);

  const buildGroundedPrompt = (text: string, history: Array<{ role: string; content: string }>, tokenizer = model.tokenizer) => {
    const grounded = knowledgeBase.buildContext(text);
    const userContent = grounded.context ? `${text}\n\n${grounded.context}` : text;
    return {
      prompt: tokenizer.formatConversationPrompt(userContent, history),
      sources: [...new Set(grounded.hits.map((hit) => hit.documentName))],
    };
  };

  const getBaseModel = (): SmallLanguageModel => {
    const key = `${model.config.id}:${model.config.vocabSize}:${defaultTokenizer.vocabSize}`;
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
    } catch (error) {
      console.error('Local generation failed:', error);
      // Never leave an empty assistant bubble behind when the runtime fails.
      setMessages((prev) => prev.map((m) =>
        m.role === 'assistant' && m.content.trim() === ''
          ? { ...m, content: 'The local model could not generate a reply. Please try again.' }
          : m
      ));
      setCompareMessages((prev) => ({
        base: prev.base.map((m) =>
          m.role === 'assistant' && m.content.trim() === ''
            ? { ...m, content: 'The local model could not generate a reply. Please try again.' }
            : m
        ),
        finetuned: prev.finetuned.map((m) =>
          m.role === 'assistant' && m.content.trim() === ''
            ? { ...m, content: 'The local model could not generate a reply. Please try again.' }
            : m
        ),
      }));
    } finally {
      setIsGenerating(false);
    }
  };

  /**
   * The turns handed back to the model as conversation history.
   *
   * Only real conversation belongs there: the synthetic welcome line and any
   * assistant placeholder that never produced tokens are not turns the model
   * actually had. The engine windows whatever is left to fit the context.
   */
  const chatHistory = () =>
    messages
      .filter((m) => m.content.trim().length > 0)
      .filter((m) => m.role === 'user' || (m.tokens?.length ?? 0) > 0)
      .slice(-4)
      .map((m) => ({ role: m.role, content: m.content }));

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
    const grounded = buildGroundedPrompt(text, chatHistory());

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
        knowledgeSources: grounded.sources,
      };

      setCompareMessages((prev) => ({
        ...prev,
        finetuned: [...prev.finetuned, ftMsg],
      }));

      const prompt = grounded.prompt;
      const ftTokens: GeneratedTokenInfo[] = [];

      for await (const tokenInfo of model.generateChatStream(prompt, options, true, grounded.sources.length === 0)) {
        ftTokens.push(tokenInfo);
        const decoded = ftTokens.map((t) => t.token).join('');
        setCompareMessages((prev) => ({
          ...prev,
          finetuned: prev.finetuned.map((m) =>
            m.id === ftId ? { ...m, content: decoded, tokens: [...ftTokens] } : m
          ),
        }));
      }

      setCompareMessages((prev) => ({ ...prev, finetuned: finishReply(prev.finetuned, ftId) }));

      // 2. Generate the Base model response from an independent, never-trained
      // instance. Toggling LoRA off on the live model was not a base-model
      // comparison: the statistical memory layer is shared state that
      // fine-tuning mutates, and it supplies most of the sampling mass.
      const baseModel = getBaseModel();
      const baseGrounded = buildGroundedPrompt(text, chatHistory(), baseModel.tokenizer);
      const basePrompt = baseGrounded.prompt;
      const baseId = `base-${Date.now()}`;
      const baseMsg: ChatMessage = {
        id: baseId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        tokens: [],
        modelSource: 'base',
        knowledgeSources: baseGrounded.sources,
      };

      setCompareMessages((prev) => ({
        ...prev,
        base: [...prev.base, baseMsg],
      }));

      const baseTokens: GeneratedTokenInfo[] = [];
      for await (const tokenInfo of baseModel.generateChatStream(basePrompt, options, true, baseGrounded.sources.length === 0)) {
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
      setCompareMessages((prev) => ({ ...prev, base: finishReply(prev.base, baseId) }));
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
        knowledgeSources: grounded.sources,
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // Format prompt with conversational history and any retrieved local knowledge.
      // The engine drops old turns when the context window cannot hold everything.
      const prompt = grounded.prompt;

      const collectedTokens: GeneratedTokenInfo[] = [];
      for await (const tokenInfo of model.generateChatStream(prompt, options, isFinetuned, grounded.sources.length === 0)) {
        collectedTokens.push(tokenInfo);
        const decoded = collectedTokens.map((t) => t.token).join('');

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: decoded, tokens: [...collectedTokens] }
              : m
          )
        );
      }
      // Covers the case where the model spent a whole turn without emitting a
      // token: the bubble must still say something.
      setMessages((prev) => finishReply(prev, assistantId));
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

  const isWelcomeState = messages.length === 1 && messages[0].role === 'assistant';

  return (
    <section className="chat-page" aria-label="Chat playground">
      <div className="chat-toolbar">
        <div className="chat-toolbar__context">
          <div className="chat-toolbar__model">
            <span className="chat-toolbar__model-dot" />
            <div className="min-w-0">
              <strong>{comparisonMode ? 'Model comparison' : 'Conversation'}</strong>
              <span>{comparisonMode ? 'Base vs fine-tuned' : (isFinetuned ? 'Fine-tuned model' : 'Base model')}</span>
              {knowledgeReady && knowledgeBase.documentCount() > 0 && !comparisonMode && (
                <span className="chat-toolbar__knowledge">Knowledge active · {knowledgeBase.documentCount()} source{knowledgeBase.documentCount() === 1 ? '' : 's'}</span>
              )}
            </div>
          </div>

          {isFinetuned && !comparisonMode && (
            <span className="chat-toolbar__status">
              <Sparkles className="h-3.5 w-3.5" />
              {activeDatasetName ? activeDatasetName : 'Fine-tuned'}
            </span>
          )}
        </div>

        <div className="chat-toolbar__actions">
          <button
            onClick={() => setComparisonMode(!comparisonMode)}
            id="toggle-comparison-btn"
            aria-pressed={comparisonMode}
            className={'chat-toolbar__button ' + (comparisonMode ? 'chat-toolbar__button--active' : '')}
          >
            <Columns className="h-3.5 w-3.5" />
            <span>Compare</span>
          </button>

          <button
            onClick={() => setShowSettings(!showSettings)}
            id="toggle-settings-btn"
            aria-expanded={showSettings}
            className="chat-toolbar__button"
          >
            <Sliders className="h-3.5 w-3.5" />
            <span>Sampling</span>
            {showSettings ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          </button>

          <button
            onClick={clearChat}
            id="clear-chat-btn"
            title="Clear current dialogue history"
            aria-label="Clear chat"
            className="chat-toolbar__icon-button"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {showSettings && (
        <div className="sampling-panel">
          <div className="sampling-panel__header">
            <div>
              <strong>Generation settings</strong>
              <span>Adjust how the local model samples its next token.</span>
            </div>
            <button
              onClick={() => setShowSettings(false)}
              className="sampling-panel__close"
              aria-label="Close sampling settings"
            >
              <ChevronUp className="h-4 w-4" />
            </button>
          </div>
          <div className="sampling-panel__grid">
            {[
              ['Temperature', options.temperature, '0.1', '1.5', '0.05', (v: number) => v.toFixed(2), 'Randomness'],
              ['Top-K', options.topK, '1', '50', '1', (v: number) => String(v), 'Candidate limit'],
              ['Top-P', options.topP, '0.1', '1.0', '0.05', (v: number) => v.toFixed(2), 'Probability cutoff'],
              ['Repetition penalty', options.repetitionPenalty, '1.0', '2.0', '0.05', (v: number) => v.toFixed(2), 'Repeat control'],
              ['Max tokens', options.maxNewTokens, '10', '48', '2', (v: number) => String(v), 'Reply length'],
            ].map(([label, value, min, max, step, format, help], index) => (
              <label key={String(label)} className="sampling-control">
                <span className="sampling-control__label">
                  <strong>{String(label)}</strong>
                  <b>{(format as (v:number)=>string)(Number(value))}</b>
                </span>
                <input
                  type="range"
                  min={String(min)}
                  max={String(max)}
                  step={String(step)}
                  value={Number(value)}
                  onChange={(e) => {
                    const next = Number(e.target.value);
                    if (index === 0) setOptions({ ...options, temperature: next });
                    if (index === 1) setOptions({ ...options, topK: next });
                    if (index === 2) setOptions({ ...options, topP: next });
                    if (index === 3) setOptions({ ...options, repetitionPenalty: next });
                    if (index === 4) setOptions({ ...options, maxNewTokens: next });
                  }}
                />
                <small>{String(help)}</small>
              </label>
            ))}
          </div>
        </div>
      )}

      <div className="chat-surface">
        {comparisonMode ? (
          <div className="chat-scroll">
            <div className="compare-grid">
              <div className="compare-panel">
                <div className="compare-panel__header">
                  <div>
                    <span className="compare-panel__eyebrow">BASE</span>
                    <strong>Pretrained model</strong>
                  </div>
                  <span className="compare-panel__dot compare-panel__dot--base" />
                </div>
                <div className="compare-panel__body">
                  {compareMessages.base.map((msg) => (
                    <div key={msg.id} className={'compare-message ' + (msg.role === 'user' ? 'compare-message--user' : '')}>
                      {msg.role === 'assistant' && (
                        <div className="chat-avatar chat-avatar--assistant chat-avatar--small">
                          <Bot className="h-3.5 w-3.5" />
                        </div>
                      )}
                      <div className={'compare-bubble ' + (msg.role === 'user' ? 'compare-bubble--user' : '')}>
                        {msg.content || (isGenerating ? 'Thinking…' : '')}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="compare-panel compare-panel--tuned">
                <div className="compare-panel__header">
                  <div>
                    <span className="compare-panel__eyebrow">FINE-TUNED</span>
                    <strong>Adapted model</strong>
                  </div>
                  <span className="compare-panel__dot compare-panel__dot--tuned" />
                </div>
                <div className="compare-panel__body">
                  {compareMessages.finetuned.map((msg) => (
                    <div key={msg.id} className={'compare-message ' + (msg.role === 'user' ? 'compare-message--user' : '')}>
                      {msg.role === 'assistant' && (
                        <div className="chat-avatar chat-avatar--tuned chat-avatar--small">
                          <Sparkles className="h-3.5 w-3.5" />
                        </div>
                      )}
                      <div className={'compare-bubble ' + (msg.role === 'user' ? 'compare-bubble--user' : '')}>
                        {msg.content || (isGenerating ? 'Thinking…' : '')}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="chat-scroll">
            {isWelcomeState ? (
              <div className="chat-welcome">
                <div className="chat-welcome__icon">
                  <MessageSquare className="h-5 w-5" />
                </div>
                <div className="chat-welcome__eyebrow">LOCAL CONVERSATION</div>
                <h2>Chat with your local model</h2>
                <p>{messages[0].content}</p>
                <div className="chat-welcome__details">
                  <span><span className="chat-welcome__status-dot" /> Runs in your browser</span>
                  <span>{isFinetuned ? 'Fine-tuned weights active' : 'Base weights active'}</span>
                </div>
                <div className="chat-welcome__prompts">
                  {samplePrompts.slice(0, 4).map((prompt, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSendMessage(prompt)}
                      disabled={isGenerating}
                      title={prompt}
                    >
                      <span>{prompt}</span>
                      <Send className="h-3.5 w-3.5" />
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="chat-thread">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={'chat-message-row ' + (msg.role === 'user' ? 'chat-message-row--user' : '')}
                  >
                    {msg.role === 'assistant' && (
                      <div className="chat-avatar chat-avatar--assistant">
                        <Bot className="h-4 w-4" />
                      </div>
                    )}

                    <div className={'chat-bubble ' + (msg.role === 'user' ? 'chat-bubble--user' : 'chat-bubble--assistant')}>
                      {msg.role === 'assistant' && msg.tokens && msg.tokens.length > 0 ? (
                        <div>
                          <div>{msg.content}</div>
                          {msg.knowledgeSources && msg.knowledgeSources.length > 0 && (
                            <div className="chat-knowledge-meta">Grounded in: {msg.knowledgeSources.join(', ')}</div>
                          )}
                          <div className="chat-token-meta">
                            <div className="flex items-center justify-between gap-3">
                              <span>{msg.tokens.length} tokens generated</span>
                              <button
                                onClick={() => setTokenStreamFor((prev) => (prev === msg.id ? null : msg.id))}
                                className="inline-flex items-center gap-1 font-semibold text-indigo-600 hover:text-indigo-800"
                              >
                                <Info className="h-3 w-3" />
                                {tokenStreamFor === msg.id ? 'Hide' : 'Inspect'}
                              </button>
                            </div>
                            {tokenStreamFor === msg.id && (
                              <div className="mt-2 flex flex-wrap gap-1 rounded-lg border border-slate-200 bg-white p-2">
                                {msg.tokens.map((tok, i) => (
                                  <button
                                    key={i}
                                    onClick={() => setInspectedToken(tok)}
                                    title={'Token: ' + JSON.stringify(tok.token) + ', ID: ' + tok.id + ', Prob: ' + (tok.prob * 100).toFixed(1) + '%'}
                                    className="rounded border border-transparent px-1 py-0.5 font-sans text-slate-800 hover:border-indigo-200 hover:bg-indigo-50"
                                  >
                                    {tok.token === ' ' ? '␣' : tok.token}
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      ) : (
                        <div>
                          {msg.content}
                          {isGenerating && msg.role === 'assistant' && msg.content === '' && (
                            <div className="flex items-center gap-2 py-1 text-slate-400">
                              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-500" />
                              <span className="text-xs">Generating locally…</span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {msg.role === 'user' && (
                      <div className="chat-avatar chat-avatar--user">
                        <User className="h-4 w-4" />
                      </div>
                    )}
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
            )}
          </div>
        )}
      </div>

      {!isWelcomeState && (
        <div className="chat-suggestions" aria-label="Suggested prompts">
          <span className="chat-suggestions__label">Suggestions</span>
          {samplePrompts.map((prompt, idx) => (
            <button
              key={idx}
              onClick={() => handleSendMessage(prompt)}
              disabled={isGenerating}
              className="chat-suggestion disabled:cursor-not-allowed disabled:opacity-50"
              title={prompt}
            >
              {prompt}
            </button>
          ))}
        </div>
      )}

      <div className="chat-composer">
        <form
          className="flex w-full items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
        >
          <div className="chat-composer__leading" aria-hidden="true">
            <Bot className="h-4 w-4" />
          </div>
          <input
            type="text"
            id="chat-input"
            value={inputPrompt}
            onChange={(e) => setInputPrompt(e.target.value)}
            placeholder="Message your local model…"
            disabled={isGenerating}
            autoComplete="off"
            className="chat-composer__input"
          />
          <button
            type="submit"
            id="send-message-btn"
            disabled={!inputPrompt.trim() || isGenerating}
            aria-label="Send message"
            className="chat-composer__send"
          >
            {isGenerating ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </form>
      </div>

      <TokenInspectorModal
        tokenInfo={inspectedToken}
        onClose={() => setInspectedToken(null)}
      />
    </section>
  );
};
