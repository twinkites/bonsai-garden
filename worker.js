import {
  pipeline,
  AutoProcessor,
  AutoModelForVision2Seq,
  RawImage,
  TextStreamer,
  DynamicCache,
  InterruptableStoppingCriteria,
} from "https://esm.sh/@huggingface/transformers@4.1.0";

const MODELS = {
  "smollm2-135m": {
    id: "HuggingFaceTB/SmolLM2-135M-Instruct",
    dtype: "q4f16",
    // Sampling avoids language drift / Chinese switching caused by greedy decoding
    // + no_repeat_ngram_size forcing the model off English tokens
    params: { max_new_tokens: 256, do_sample: true, temperature: 0.7, top_p: 0.9, repetition_penalty: 1.1 },
  },
  "bonsai-1.7b": {
    id: "onnx-community/Bonsai-1.7B-ONNX",
    dtype: "q1",
    params: { max_new_tokens: 1024, do_sample: false, repetition_penalty: 1.2, no_repeat_ngram_size: 3 },
  },
  "bonsai-4b": {
    id: "onnx-community/Bonsai-4B-ONNX",
    dtype: "q4",
    params: { max_new_tokens: 1024, do_sample: false, repetition_penalty: 1.2, no_repeat_ngram_size: 3 },
  },
  "deepseek-r1": {
    id: "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
    dtype: "q4f16",
    params: { max_new_tokens: 8192, do_sample: true, temperature: 0.6, top_p: 0.95 },
  },
  "qwen3-0.6b": {
    id: "onnx-community/Qwen3-0.6B-Instruct-ONNX",
    dtype: "q4",
    params: { max_new_tokens: 1024, do_sample: true, temperature: 0.6, top_p: 0.95, top_k: 20 },
  },
  "smolvlm-256m": {
    id: "HuggingFaceTB/SmolVLM-256M-Instruct",
    vision: true,
    // Three-component dtype map required by AutoModelForVision2Seq
    dtype: { embed_tokens: "fp32", vision_encoder: "q4f16", decoder_model_merged: "q4f16" },
    params: { max_new_tokens: 256, do_sample: false },
  },
};

// ── Text pipeline ─────────────────────────────────────────────────────────────
class PipelineRegistry {
  static cache = new Map();

  static getOrCreate(key, id = null, dtype = null, progress_callback = null) {
    if (!this.cache.has(key)) {
      this.cache.set(
        key,
        pipeline("text-generation", id, { device: "webgpu", dtype, progress_callback }),
      );
    }
    return this.cache.get(key);
  }
}

// ── Vision pipeline ───────────────────────────────────────────────────────────
let visionProcessor = null;
let visionModel     = null;
let visionModelKey  = null;

// ── Shared state ──────────────────────────────────────────────────────────────
const stopping_criteria = new InterruptableStoppingCriteria();
let past_key_values_cache = null;
let current_model_key = null;
let current_params    = null;

function disposePastKeyValues() {
  past_key_values_cache?.dispose?.();
  past_key_values_cache = null;
}

// ── WebGPU check ──────────────────────────────────────────────────────────────
async function check() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("WebGPU is not supported in this browser.");
    self.postMessage({ status: "check_ok" });
  } catch (e) {
    self.postMessage({ status: "error", data: e.message });
  }
}

// ── Text model load ───────────────────────────────────────────────────────────
// spec: { type: "preset", key } | { type: "custom", id, dtype, max_new_tokens }
async function load(spec) {
  let modelKey, modelId, dtype, params;

  if (spec.type === "preset") {
    modelKey = spec.key;
    const entry = MODELS[modelKey];
    if (!entry) { self.postMessage({ status: "error", data: `Unknown model: ${modelKey}` }); return; }
    modelId = entry.id;
    dtype   = entry.dtype;
    params  = entry.params;
  } else {
    modelKey = "__custom__";
    modelId  = spec.id;
    dtype    = spec.dtype;
    params   = { max_new_tokens: spec.max_new_tokens ?? 1024, do_sample: true, temperature: 0.7, top_p: 0.9 };
  }

  if (current_model_key && current_model_key !== modelKey) disposePastKeyValues();
  current_model_key = modelKey;
  current_params    = params;

  self.postMessage({ status: "loading", data: "Fetching model weights…" });

  const generator = await PipelineRegistry.getOrCreate(modelKey, modelId, dtype, (info) => {
    if (info.status === "progress_total") {
      self.postMessage({ status: "progress", loaded: Number(info.loaded ?? 0), total: Number(info.total ?? 0) });
    }
  });

  self.postMessage({ status: "loading", data: "Warming up model…" });
  const inputs = generator.tokenizer("a");
  await generator.model.generate({ ...inputs, max_new_tokens: 1 });
  self.postMessage({ status: "ready" });
}

// ── Vision model load ─────────────────────────────────────────────────────────
async function loadVision(modelKey) {
  const spec = MODELS[modelKey];

  // Already loaded — skip
  if (visionModelKey === modelKey && visionProcessor && visionModel) {
    self.postMessage({ status: "ready" });
    return;
  }

  visionProcessor = null;
  visionModel     = null;
  visionModelKey  = null;
  disposePastKeyValues();
  current_model_key = modelKey;
  current_params    = spec.params;

  const progressCb = (info) => {
    if (info.total > 0 && (info.status === "progress" || info.status === "progress_total")) {
      self.postMessage({ status: "progress", loaded: Number(info.loaded ?? 0), total: Number(info.total ?? 0) });
    }
  };

  self.postMessage({ status: "loading", data: "Loading vision processor…" });
  visionProcessor = await AutoProcessor.from_pretrained(spec.id, { progress_callback: progressCb });

  self.postMessage({ status: "loading", data: "Loading vision model…" });
  visionModel = await AutoModelForVision2Seq.from_pretrained(spec.id, {
    dtype: spec.dtype,
    device: "webgpu",
    progress_callback: progressCb,
  });

  visionModelKey = modelKey;
  self.postMessage({ status: "ready" });
}

// ── Text generation ───────────────────────────────────────────────────────────
async function generate(messages) {
  const generator = await PipelineRegistry.getOrCreate(current_model_key);

  let startTime, numTokens = 0;
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (output) => {
      const tps = numTokens > 1 ? (numTokens / (performance.now() - startTime)) * 1000 : null;
      self.postMessage({ status: "update", output, tps, numTokens });
    },
    token_callback_function: () => { startTime ??= performance.now(); numTokens++; },
  });

  self.postMessage({ status: "start" });
  past_key_values_cache ??= new DynamicCache();

  try {
    const output = await generator(messages, {
      ...current_params, streamer, stopping_criteria, past_key_values: past_key_values_cache,
    });
    self.postMessage({ status: "complete", output: output[0].generated_text.at(-1).content });
  } catch (e) {
    self.postMessage({ status: "error", data: e.message });
  }
}

// ── Vision generation ─────────────────────────────────────────────────────────
// data: { messages: [{role, content}], imageDataUrl: string|null }
async function generateVision({ messages, imageDataUrl }) {
  self.postMessage({ status: "start" });

  try {
    // Build VLM message array — only the final user turn gets the image placeholder
    const vlmMessages = messages.map((msg, i) => {
      const isLastUser = msg.role === "user" && i === messages.length - 1;
      if (isLastUser && imageDataUrl) {
        return { role: "user", content: [{ type: "image" }, { type: "text", text: msg.content }] };
      }
      return { role: msg.role, content: msg.content };
    });

    const text = visionProcessor.apply_chat_template(vlmMessages, {
      tokenize: false,
      add_generation_prompt: true,
    });

    const images = imageDataUrl ? [await RawImage.fromURL(imageDataUrl)] : null;
    const inputs = await visionProcessor(text, images, { return_tensors: "pt" });

    let startTime, numTokens = 0, accumulated = "";
    const streamer = new TextStreamer(visionProcessor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (output) => {
        accumulated += output;
        const tps = numTokens > 1 ? (numTokens / (performance.now() - startTime)) * 1000 : null;
        self.postMessage({ status: "update", output, tps, numTokens });
      },
      token_callback_function: () => { startTime ??= performance.now(); numTokens++; },
    });

    await visionModel.generate({ ...inputs, ...current_params, streamer, stopping_criteria });
    self.postMessage({ status: "complete", output: accumulated });
  } catch (e) {
    self.postMessage({ status: "error", data: e.message });
  }
}

// ── Message handler ───────────────────────────────────────────────────────────
self.addEventListener("message", async ({ data: { type, data } }) => {
  switch (type) {
    case "check": check(); break;

    case "load": {
      const key = data.type === "preset" ? data.key : "__custom__";
      if (MODELS[key]?.vision) loadVision(key);
      else load(data);
      break;
    }

    case "generate": {
      stopping_criteria.reset();
      // data is always { messages, imageDataUrl } — vision and text both use this shape
      if (MODELS[current_model_key]?.vision) generateVision(data);
      else generate(data.messages);
      break;
    }

    case "interrupt": stopping_criteria.interrupt(); break;

    case "reset":
      disposePastKeyValues();
      stopping_criteria.reset();
      break;
  }
});
