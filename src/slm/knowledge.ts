/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Local, browser-only retrieval store for MiniSLM.
 *
 * This deliberately does not retrain the transformer. Domain knowledge is kept
 * outside the model and retrieved at chat time, so adding a document is cheap,
 * reversible, persistent, and does not destroy learned model weights.
 */

export interface KnowledgeDocument {
  id: string;
  name: string;
  content: string;
  createdAt: number;
  updatedAt: number;
  chunkCount: number;
}

export interface KnowledgeChunk {
  id: string;
  documentId: string;
  documentName: string;
  text: string;
  index: number;
}

export interface KnowledgeHit extends KnowledgeChunk {
  score: number;
}

const DB_NAME = 'minislm-local-knowledge';
const DB_VERSION = 1;
const STORE_NAME = 'documents';
const MAX_CHUNK_CHARS = 700;
const CHUNK_OVERLAP_CHARS = 100;

const STOP_WORDS = new Set([
  'a','an','and','are','as','at','be','but','by','can','could','do','does','for',
  'from','had','has','have','he','her','here','hers','him','his','how','i','if',
  'in','into','is','it','its','me','more','my','of','on','or','our','she','so',
  'that','the','their','them','there','these','they','this','to','up','us','was',
  'we','were','what','when','where','which','who','why','will','with','you','your'
]);

function terms(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .split(/\s+/)
    .filter((word) => word.length > 1 && !STOP_WORDS.has(word));
}

function splitText(text: string): string[] {
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
  if (!normalized) return [];

  const paragraphs = normalized
    .split(/\n\s*\n+/)
    .map((p) => p.trim())
    .filter(Boolean);

  const pieces: string[] = [];
  for (const paragraph of paragraphs.length ? paragraphs : [normalized]) {
    if (paragraph.length <= MAX_CHUNK_CHARS) {
      pieces.push(paragraph);
      continue;
    }

    const sentences = paragraph.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [paragraph];
    let current = '';
    for (const sentence of sentences.map((s) => s.trim()).filter(Boolean)) {
      if (!current) {
        current = sentence;
      } else if ((current + ' ' + sentence).length <= MAX_CHUNK_CHARS) {
        current += ' ' + sentence;
      } else {
        pieces.push(current);
        const overlap = current.slice(Math.max(0, current.length - CHUNK_OVERLAP_CHARS));
        current = overlap + ' ' + sentence;
      }
    }
    if (current.trim()) pieces.push(current.trim());
  }

  return pieces;
}

export class LocalKnowledgeBase {
  private documents = new Map<string, KnowledgeDocument>();
  private chunks: KnowledgeChunk[] = [];
  private ready = false;

  public async load(): Promise<void> {
    if (this.ready) return;
    if (typeof indexedDB === 'undefined') {
      this.ready = true;
      return;
    }

    try {
      const db = await this.openDb();
      const rows = await new Promise<KnowledgeDocument[]>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const request = tx.objectStore(STORE_NAME).getAll();
        request.onsuccess = () => resolve((request.result ?? []) as KnowledgeDocument[]);
        request.onerror = () => reject(request.error);
      });
      for (const doc of rows) this.documents.set(doc.id, doc);
      this.rebuildIndex();
    } catch {
      // The chat remains usable if IndexedDB is unavailable or blocked.
    } finally {
      this.ready = true;
    }
  }

  public isReady(): boolean {
    return this.ready;
  }

  public listDocuments(): KnowledgeDocument[] {
    return [...this.documents.values()].sort((a, b) => b.updatedAt - a.updatedAt);
  }

  public documentCount(): number {
    return this.documents.size;
  }

  public chunkCount(): number {
    return this.chunks.length;
  }

  public async addDocument(name: string, content: string): Promise<KnowledgeDocument> {
    const cleanName = name.trim() || 'Untitled note';
    const cleanContent = content.trim();
    if (!cleanContent) throw new Error('Knowledge cannot be empty.');

    const now = Date.now();
    const doc: KnowledgeDocument = {
      id: globalThis.crypto?.randomUUID?.() ?? `doc-${now}-${Math.random().toString(36).slice(2)}`,
      name: cleanName,
      content: cleanContent,
      createdAt: now,
      updatedAt: now,
      chunkCount: splitText(cleanContent).length,
    };

    this.documents.set(doc.id, doc);
    this.rebuildIndex();
    await this.persist(doc);
    return doc;
  }

  public async deleteDocument(id: string): Promise<void> {
    this.documents.delete(id);
    this.rebuildIndex();

    if (typeof indexedDB === 'undefined') return;
    try {
      const db = await this.openDb();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).delete(id);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch {
      // Keep the in-memory copy consistent even if persistence is unavailable.
    }
  }

  public clear(): Promise<void> {
    this.documents.clear();
    this.rebuildIndex();
    if (typeof indexedDB === 'undefined') return Promise.resolve();

    return this.openDb().then((db) =>
      new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).clear();
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      })
    ).catch(() => undefined);
  }

  public search(query: string, limit = 4): KnowledgeHit[] {
    const queryTerms = terms(query);
    if (!queryTerms.length || !this.chunks.length) return [];

    const querySet = new Set(queryTerms);
    const documentFrequency = new Map<string, number>();
    for (const chunk of this.chunks) {
      const unique = new Set(terms(chunk.text));
      for (const term of unique) documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
    }

    const totalChunks = this.chunks.length;
    const scored: KnowledgeHit[] = [];
    const queryLower = query.toLowerCase().trim();

    for (const chunk of this.chunks) {
      const chunkTerms = terms(chunk.text);
      const counts = new Map<string, number>();
      for (const term of chunkTerms) counts.set(term, (counts.get(term) ?? 0) + 1);

      let score = 0;
      for (const term of querySet) {
        const tf = counts.get(term) ?? 0;
        if (!tf) continue;
        const idf = Math.log((totalChunks + 1) / ((documentFrequency.get(term) ?? 0) + 1)) + 1;
        score += (1 + Math.log(tf)) * idf;
      }

      if (queryLower.length > 8 && chunk.text.toLowerCase().includes(queryLower)) score += 4;
      if (querySet.size > 0) {
        const overlap = queryTerms.filter((term) => counts.has(term)).length / querySet.size;
        score += overlap * 2.5;
      }

      if (score > 0) scored.push({ ...chunk, score });
    }

    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  public buildContext(query: string, maxChars = 3200): { context: string; hits: KnowledgeHit[] } {
    const hits = this.search(query, 5);
    if (!hits.length) return { context: '', hits: [] };

    let used = 0;
    const selected: KnowledgeHit[] = [];
    const blocks: string[] = [];

    for (const hit of hits) {
      const block = `[Source: ${hit.documentName}]\n${hit.text}`;
      if (used + block.length > maxChars && selected.length > 0) continue;
      selected.push(hit);
      blocks.push(block);
      used += block.length;
    }

    return {
      hits: selected,
      context: [
        'Relevant local knowledge is provided below.',
        'Use it when it directly answers the user. Do not invent facts that are not supported by it.',
        '',
        blocks.join('\n\n'),
      ].join('\n'),
    };
  }

  private rebuildIndex(): void {
    this.chunks = [];
    for (const doc of this.documents.values()) {
      splitText(doc.content).forEach((text, index) => {
        this.chunks.push({
          id: `${doc.id}:${index}`,
          documentId: doc.id,
          documentName: doc.name,
          text,
          index,
        });
      });
    }
  }

  private async persist(doc: KnowledgeDocument): Promise<void> {
    if (typeof indexedDB === 'undefined') return;
    try {
      const db = await this.openDb();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).put(doc);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch {
      // IndexedDB is an enhancement, not a requirement for the current session.
    }
  }

  private openDb(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
}
