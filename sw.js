const CACHE = "bonsai-garden-v3";
const ASSETS = [
  "./",
  "./index.html",
  "./worker.js",
  "./bonsai.png",
  "./1.svg",
  "./manifest.json",
  "./vendor/marked.min.js",
  "./vendor/purify.min.js",
  "./vendor/highlight.min.js",
  "./vendor/github-dark.min.css",
  "./vendor/github.min.css",
  "./vendor/transformers.js",
  // Not precached: the ort-wasm-simd-threaded*.wasm/.mjs runtime files (~13-24MB each).
  // transformers.js caches whichever variant it actually needs in its own
  // CacheStorage entry the first time a model is loaded, same as model weights.
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);

  // Never intercept model weight / ONNX runtime downloads — transformers.js
  // caches those itself; let it manage them natively instead of double-caching.
  if (
    url.hostname.includes("huggingface.co") || url.hostname.includes("hf.co") ||
    url.pathname.startsWith("/vendor/ort-wasm-simd-threaded")
  ) {
    return;
  }

  e.respondWith(
    caches.match(e.request).then(cached => cached ?? fetch(e.request))
  );
});
