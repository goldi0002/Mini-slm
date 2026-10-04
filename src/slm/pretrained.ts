/**
 * Browser-local pretrained language model runtime.
 *
 * The weights are downloaded from Hugging Face on first use and then cached by
 * Transformers.js in the browser. WebGPU is preferred when available, with a
 * real WASM/CPU fallback for browsers where WebGPU initialization or execution
 * is unavailable.
 */
import { pipeline, TextStreamer } from '@huggingface/transformers';
import type { GenerationOptions, GeneratedTokenInfo } from '../types';

export const PRETRAINED_MODEL_ID = 'onnx-community/SmolLM2-135M-Instruct-ONNX';

type Generator = ((messages: Array<{ role: string; content: string }>, options?: Record<string, unknown>) => Promise<unknown>) & {
  tokenizer: { decode: (ids: bigint[] | number[], options?: Record<string, unknown>) => string };
};

let generatorPromise: Promise<Generator> | null = null;
let generatorDevice: 'webgpu' | 'wasm' | null = null;

async function hasUsableWebGPU(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;

  try {
    const gpu = (navigator as Navigator & {
      gpu?: { requestAdapter: () => Promise<unknown | null> };
    }).gpu;
    return Boolean(gpu && await gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function createGenerator(device: 'webgpu' | 'wasm'): Promise<Generator> {
  return pipeline('text-generation', PRETRAINED_MODEL_ID, {
    device,
    dtype: device === 'webgpu' ? 'q4f16' : 'q4',
  }) as unknown as Promise<Generator>;
}

async function getGenerator(): Promise<Generator> {
  if (!generatorPromise) {
    generatorPromise = (async () => {
      const webgpuAvailable = await hasUsableWebGPU();

      if (webgpuAvailable) {
        try {
          const generator = await createGenerator('webgpu');
          generatorDevice = 'webgpu';
          return generator;
        } catch (error) {
          // WebGPU can exist but still fail during ONNX model/kernel setup on
          // a particular mobile GPU. Fall through to the compatible WASM path.
          console.warn('WebGPU pretrained model initialization failed; falling back to WASM.', error);
        }
      }

      const generator = await createGenerator('wasm');
      generatorDevice = 'wasm';
      return generator;
    })().catch((error) => {
      // Do not permanently cache a rejected initialization promise. A later
      // attempt should be able to retry after a transient download/runtime
      // failure.
      generatorPromise = null;
      generatorDevice = null;
      throw error;
    });
  }

  return generatorPromise;
}

async function getWasmGenerator(): Promise<Generator> {
  if (generatorDevice === 'wasm' && generatorPromise) return generatorPromise;

  generatorPromise = createGenerator('wasm')
    .then((generator) => {
      generatorDevice = 'wasm';
      return generator;
    })
    .catch((error) => {
      generatorPromise = null;
      generatorDevice = null;
      throw error;
    });

  return generatorPromise;
}

function parseConversation(prompt: string): Array<{ role: string; content: string }> {
  const turns: Array<{ role: string; content: string }> = [];
  const re = /<user>\s*([\s\S]*?)(?=\s*<assistant>|$)|<assistant>\s*([\s\S]*?)(?=\s*<user>|$)/gi;
  let match: RegExpExecArray | null;
  while ((match = re.exec(prompt))) {
    if (match[1] !== undefined) turns.push({ role: 'user', content: match[1].trim() });
    else if (match[2] !== undefined) turns.push({ role: 'assistant', content: match[2].trim() });
  }
  if (!turns.length) turns.push({ role: 'user', content: prompt.trim() });
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
  return text.replace(/^<assistant>\s*/i, '').split(/<user>|<assistant>/i)[0].trim();
}

export async function generatePretrainedReply(prompt: string, options: GenerationOptions): Promise<string> {
  const run = async (generator: Generator) => generator(parseConversation(prompt), {
    max_new_tokens: Math.min(options.maxNewTokens, 32),
    temperature: Math.max(0.1, options.temperature),
    top_k: options.topK,
    top_p: options.topP,
    repetition_penalty: options.repetitionPenalty,
    do_sample: true,
  });

  let generator = await getGenerator();

  try {
    const output = await run(generator);
    return cleanReply(generatedText(output));
  } catch (error) {
    // Some Android GPUs pass adapter/pipeline initialization but fail on the
    // first actual inference. Retry once on the portable WASM backend.
    if (generatorDevice !== 'webgpu') throw error;
    console.warn('WebGPU pretrained inference failed; retrying with WASM.', error);
    generator = await getWasmGenerator();
    const output = await run(generator);
    return cleanReply(generatedText(output));
  }
}

class StreamGenerationError extends Error {
  constructor(
    message: string,
    public readonly emittedTokens: boolean,
    public readonly cause: unknown,
  ) {
    super(message);
    this.name = 'StreamGenerationError';
  }
}

async function* streamWithGenerator(
  generator: Generator,
  prompt: string,
  options: GenerationOptions,
): AsyncGenerator<GeneratedTokenInfo> {
  const queue: GeneratedTokenInfo[] = [];
  const waiters: Array<() => void> = [];
  let finished = false;
  let failure: unknown = null;
  let emittedTokens = false;

  const pushToken = (ids: bigint[]) => {
    for (const rawId of ids) {
      const id = Number(rawId);
      const token = generator.tokenizer.decode([rawId], {
        skip_special_tokens: true,
        clean_up_tokenization_spaces: false,
      });
      if (!token) continue;
      queue.push({ token, id, prob: 1, topCandidates: [{ token, id, prob: 1 }] });
      waiters.shift()?.();
    }
  };

  const streamer = new TextStreamer(generator.tokenizer as never, {
    skip_prompt: true,
    skip_special_tokens: true,
    token_callback_function: pushToken,
  });

  void generator(parseConversation(prompt), {
    max_new_tokens: Math.min(options.maxNewTokens, 32),
    temperature: Math.max(0.1, options.temperature),
    top_k: options.topK,
    top_p: options.topP,
    repetition_penalty: options.repetitionPenalty,
    do_sample: true,
    streamer,
  }).then(() => {
    finished = true;
    waiters.splice(0).forEach((resolve) => resolve());
  }).catch((error) => {
    failure = error;
    finished = true;
    waiters.splice(0).forEach((resolve) => resolve());
  });

  while (!finished || queue.length) {
    if (!queue.length) {
      await new Promise<void>((resolve) => waiters.push(resolve));
      if (failure) throw failure;
      continue;
    }
    const token = queue.shift();
    if (token) {
      emittedTokens = true;
      yield token;
    }
  }
  if (failure) {
    throw new StreamGenerationError(
      'Pretrained streaming inference failed.',
      emittedTokens,
      failure,
    );
  }
}

export async function* generatePretrainedStream(prompt: string, options: GenerationOptions): AsyncGenerator<GeneratedTokenInfo> {
  let generator = await getGenerator();

  try {
    yield* streamWithGenerator(generator, prompt, options);
  } catch (error) {
    // Only retry if WebGPU failed before producing any visible token. Retrying
    // after partial output would duplicate the beginning of the response.
    if (
      generatorDevice !== 'webgpu' ||
      (error instanceof StreamGenerationError && error.emittedTokens)
    ) {
      throw error instanceof StreamGenerationError ? error.cause : error;
    }
    console.warn('WebGPU pretrained streaming failed; retrying with WASM.', error);
    generator = await getWasmGenerator();
    yield* streamWithGenerator(generator, prompt, options);
  }
}
