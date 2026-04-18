const CACHE = "bonsai-garden-v1";
const ASSETS = [
  "./",
  "./index.html",
  "./worker.js",
  "./bonsai.png",
  "./1.svg",
  "./manifest.json",
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

  // Never intercept model weight downloads — let the browser cache those natively
  if (url.hostname.includes("huggingface.co") || url.hostname.includes("hf.co") || url.hostname === "esm.sh") {
    return;
  }

  e.respondWith(
    caches.match(e.request).then(cached => cached ?? fetch(e.request))
  );
});
