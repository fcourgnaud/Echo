/* Écho — service worker : permet d'installer l'app et de jouer hors ligne.
   Il va d'abord chercher la version en ligne (tes modifs sur GitHub
   apparaissent donc tout de suite), et utilise la copie gardée
   en mémoire seulement si le téléphone n'a pas de réseau. */
const CACHE = 'echo-v8';
const FICHIERS = ['./', './index.html', './theme.css', './manifest.webmanifest', './icon-192.png', './icon-512.png'];
const JSQR = 'https://cdn.jsdelivr.net/npm/jsqr@1.4.0/dist/jsQR.js';

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(async c => {
    await c.addAll(FICHIERS);
    try { await c.add(JSQR); } catch (err) {}
  }));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(noms =>
    Promise.all(noms.filter(n => n !== CACHE).map(n => caches.delete(n)))));
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    fetch(e.request).then(r => {
      if (r.ok) { const copie = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copie)); }
      return r;
    }).catch(() => caches.match(e.request, { ignoreSearch: true }))
  );
});
