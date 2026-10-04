/**
 * Browser-local pretrained language model runtime.
 *
 * The weights are downloaded from Hugging Face on first use and then cached by
 * Transformers.js in the browser. WebGPU is preferred when available; the
 * runtime falls back to WASM/CPU automatically.
 */
import { pipeline, TextStreamer } from '@huggingface/transformers';
import type { GenerationOptions, GeneratedTokenInfo } from '../types';

export const PRETRAINED_MODEL_ID = 'onnx-community/SmolLM2-135M-Instruct-ONNX';

type Generator = ((messages: Array<{ role: string; content: string }>, options?: Record<string, unknown>) => Promise<unknown>) & {\n  tokenizer: { decode: (ids: bigint[] | number[], options?: Record<string, unknown>) => string };\n};

let generatorPromise: Promise<Generator> | null = null;

function getDevice(): 'webgpu' | 'wasm' {
  return typeof navigator !== 'undefined' && 'gpu' in navigator ? 'webgpu' : 'wasm';
}

async function getGenerator(): Promise<Generator> {
  if (!generatorPromise) {
    const device = getDevice();
    generatorPromise = pipeline('text-generation', PRETRAINED_MODEL_ID, {
      device,
      dtype: device === 'webgpu' ? 'q4f16' : 'q4',
    }) as unknown as Promise<Generator>;
  }
  return generatorPromise;
}

function parseConversation(prompt: string): Array<{ role: string; content: string }> {
  const turns: Array<{ role: string; content: string }> = [];
  const re = /<user>\s*([\s\S]*?)(?=\s*<assistant>|$)|<assistant>\s*([\s\S]*?)(?=\s*<user>|$)/gi;
  let match: RegExpExecArray | null;
  while ((match = re.exec(prompt))) {
    if (match[1] !== undefined) {
      turns.push({ role: 'user', content: match[1].trim() });
    } else if (match[2] !== undefined) {
      turns.push({ role: 'assistant', content: match[2].trim() });
    }
  }

  if (!turns.length) {
    turns.push({ role: 'user', content: prompt.trim() });
  }

  return turns.filter((turn) => turn.content.length > 0);
}

function generatedText(output: unknown): string {
  const first = Array.isArray(output) ? output[0] : output;
  if (typeof first === 'string') return first;

  const generated = (first as { generated_text?: unknown } | undefined)?.generated_text;
  if (typeof generated === 'string') return generated;

  if (Array.isArray(generated)) {
    const last = generated[generated.length - 1];
    if (last && typeof last === 'object' && typeof (last as { content?: unknown }).content === 'string') {
      return (last as { content: string }).content;
    }
  }

  return '';
}

function cleanReply(text: string): string {
  return text
    .replace(/^<assistant>\s*/i, '')
    .split(/<user>|<assistant>/i)[0]
    .trim();
}

export async function generatePretrainedReply(
  prompt: string,
  options: GenerationOptions,
): Promise<string> {
  const generator = await getGenerator();
  const messages = parseConversation(prompt);
  const output = await generator(messages, {
    max_new_tokens: options.maxNewTokens,
    temperature: Math.max(0.1, options.temperature),
    top_k: options.topK,
    top_p: options.topP,
    repetition_penalty: options.repetitionPenalty,
    do_sample: true,
  });

  return cleanReply(generatedText(output));
}

export async function* generatePretrainedStream(
  prompt: string,
  options: GenerationOptions,
): AsyncGenerator<GeneratedTokenInfo> {
  const reply = await generatePretrainedReply(prompt, options);
  const ids = defaultTokenizer.encode(reply);

  for (const id of ids) {
    const token = defaultTokenizer.decode([id], true);
    yield {
      token,
      id,
      prob: 1,
      topCandidates: [{ token, id, prob: 1 }],
    };
    await new Promise((resolve) => setTimeout(resolve, 8));
  }
}
