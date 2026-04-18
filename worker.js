import {
  pipeline,
  TextStreamer,
  DynamicCache,
  InterruptableStoppingCriteria,
} from "https://esm.sh/@huggingface/transformers@4.1.0";

const MODELS = {
  "bonsai-1.7b": {
    id: "onnx-community/Bonsai-1.7B-ONNX",
    dtype: "q1",
    // 1-bit model — greedy decoding, strong repetition controls
    params: { max_new_tokens: 1024, do_sample: false, repetition_penalty: 1.2, no_repeat_ngram_size: 3 },
  },
  "bonsai-4b": {
    id: "onnx-community/Bonsai-4B-ONNX",
    dtype: "q4",
    // 4-bit model — same family, same approach
    params: { max_new_tokens: 1024, do_sample: false, repetition_penalty: 1.2, no_repeat_ngram_size: 3 },
  },
  "deepseek-r1": {
    id: "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
    dtype: "q4f16",
    // Reasoning model — needs sampling + higher token budget; no_repeat_ngram_size breaks CoT;
    // system prompt should be avoided (handled in index.html)
    params: { max_new_tokens: 8192, do_sample: true, temperature: 0.6, top_p: 0.95 },
  },
  "qwen3-0.6b": {
    id: "onnx-community/Qwen3-0.6B-Instruct-ONNX",
    dtype: "q4",
    // Thinking mode on by default — greedy decoding causes loops; no_repeat_ngram_size breaks CoT
    params: { max_new_tokens: 1024, do_sample: true, temperature: 0.6, top_p: 0.95, top_k: 20 },
  },
};

class PipelineRegistry {
  static cache = new Map();

  static getInstance(modelKey, progress_callback = null) {
    const spec = MODELS[modelKey];
    if (!spec) throw new Error(`Unknown model: ${modelKey}`);
    if (!this.cache.has(modelKey)) {
      this.cache.set(
        modelKey,
        pipeline("text-generation", spec.id, {
          device: "webgpu",
          dtype: spec.dtype,
          progress_callback,
        }),
      );
    }
    return this.cache.get(modelKey);
  }
}

const stopping_criteria = new InterruptableStoppingCriteria();
let past_key_values_cache = null;
let current_model_key = null;

function disposePastKeyValues() {
  past_key_values_cache?.dispose?.();
  past_key_values_cache = null;
}

async function check() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("WebGPU is not supported in this browser.");
    self.postMessage({ status: "check_ok" });
  } catch (e) {
    self.postMessage({ status: "error", data: e.message });
  }
}

async function load(modelKey) {
  if (current_model_key && current_model_key !== modelKey) {
    disposePastKeyValues();
  }
  current_model_key = modelKey;

  self.postMessage({ status: "loading", data: "Fetching model weights…" });

  const generator = await PipelineRegistry.getInstance(modelKey, (info) => {
    if (info.status === "progress_total") {
      self.postMessage({
        status: "progress",
        loaded: Number(info.loaded ?? 0),
        total: Number(info.total ?? 0),
      });
    }
  });

  self.postMessage({ status: "loading", data: "Warming up model…" });
  const inputs = generator.tokenizer("a");
  await generator.model.generate({ ...inputs, max_new_tokens: 1 });

  self.postMessage({ status: "ready" });
}

async function generate(messages) {
  const generator = await PipelineRegistry.getInstance(current_model_key);

  let startTime;
  let numTokens = 0;

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (output) => {
      const tps = numTokens > 1
        ? (numTokens / (performance.now() - startTime)) * 1000
        : null;
      self.postMessage({ status: "update", output, tps, numTokens });
    },
    token_callback_function: () => {
      startTime ??= performance.now();
      numTokens++;
    },
  });

  self.postMessage({ status: "start" });
  past_key_values_cache ??= new DynamicCache();

  try {
    const { params } = MODELS[current_model_key];
    const output = await generator(messages, {
      ...params,
      streamer,
      stopping_criteria,
      past_key_values: past_key_values_cache,
    });

    self.postMessage({
      status: "complete",
      output: output[0].generated_text.at(-1).content,
    });
  } catch (e) {
    self.postMessage({ status: "error", data: e.message });
  }
}

self.addEventListener("message", async ({ data: { type, data } }) => {
  switch (type) {
    case "check":     check(); break;
    case "load":      load(data); break;
    case "generate":
      stopping_criteria.reset();
      generate(data);
      break;
    case "interrupt":
      stopping_criteria.interrupt();
      break;
    case "reset":
      disposePastKeyValues();
      stopping_criteria.reset();
      break;
  }
});
