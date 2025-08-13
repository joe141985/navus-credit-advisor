/**
 * NAVUS PWA Service Worker
 * Handles caching, offline functionality, and background sync
 */

const CACHE_NAME = 'navus-v1.0.0';
const API_CACHE_NAME = 'navus-api-v1.0.0';

// Assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/login.html',
  '/manifest.json',
  
  // Icons
  '/favicon-16x16.png',
  '/favicon-32x32.png',
  '/icon-48x48.png',
  '/icon-72x72.png',
  '/icon-96x96.png',
  '/icon-144x144.png',
  '/apple-touch-icon.png',
  '/icon-192x192.png',
  '/icon-512x512.png',
  
  // Auth icons
  '/icons8-google-48.svg',
  '/twitch-icon.svg',
  
  // Static assets from build
  '/static/css/main.196b0ba9.css',
  '/static/js/main.2d77702e.js',
];

// API endpoints to cache
const API_ENDPOINTS = [
  '/cards',
  '/auth/status'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('💾 NAVUS SW: Installing service worker...');
  
  event.waitUntil(
    Promise.all([
      // Cache static assets
      caches.open(CACHE_NAME).then((cache) => {
        console.log('💾 NAVUS SW: Caching static assets');
        return cache.addAll(STATIC_ASSETS.filter(url => url !== '/'));
      }),
      
      // Cache API responses
      caches.open(API_CACHE_NAME).then((cache) => {
        console.log('💾 NAVUS SW: Pre-caching API endpoints');
        return Promise.all(
          API_ENDPOINTS.map(endpoint => {
            return fetch(`https://web-production-685ca.up.railway.app${endpoint}`)
              .then(response => {
                if (response.ok) {
                  return cache.put(endpoint, response.clone());
                }
              })
              .catch(err => {
                console.log(`💾 NAVUS SW: Could not pre-cache ${endpoint}:`, err.message);
              });
          })
        );
      })
    ]).then(() => {
      console.log('✅ NAVUS SW: Service worker installed successfully');
      // Force activation of new service worker
      return self.skipWaiting();
    })
  );
});

// Activate event - cleanup old caches
self.addEventListener('activate', (event) => {
  console.log('🔄 NAVUS SW: Activating service worker...');
  
  event.waitUntil(
    Promise.all([
      // Cleanup old caches
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== CACHE_NAME && cacheName !== API_CACHE_NAME) {
              console.log('🗑️ NAVUS SW: Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      }),
      
      // Take control of all pages
      self.clients.claim()
    ]).then(() => {
      console.log('✅ NAVUS SW: Service worker activated successfully');
    })
  );
});

// Fetch event - implement caching strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Handle different types of requests with different strategies
  if (request.method !== 'GET') {
    // Don't cache non-GET requests
    return;
  }
  
  // API requests - Network First with fallback to cache
  if (url.origin === 'https://web-production-685ca.up.railway.app') {
    event.respondWith(handleApiRequest(request));
    return;
  }
  
  // Static assets - Cache First
  if (isStaticAsset(request.url)) {
    event.respondWith(handleStaticAsset(request));
    return;
  }
  
  // HTML pages - Network First with cache fallback
  if (request.destination === 'document') {
    event.respondWith(handlePageRequest(request));
    return;
  }
  
  // Default: Network First
  event.respondWith(
    fetch(request).catch(() => {
      return caches.match(request);
    })
  );
});

/**
 * Handle API requests - Network First with fallback to cache
 */
async function handleApiRequest(request) {
  const cache = await caches.open(API_CACHE_NAME);
  
  try {
    // Try network first
    const response = await fetch(request);
    
    if (response.ok) {
      // Cache successful responses
      cache.put(request, response.clone());
      return response;
    }
    
    // If response not ok, try cache
    const cached = await cache.match(request);
    return cached || response;
    
  } catch (error) {
    console.log('🔄 NAVUS SW: Network failed, trying cache for:', request.url);
    
    // Network failed, try cache
    const cached = await cache.match(request);
    
    if (cached) {
      return cached;
    }
    
    // Return offline fallback for chat endpoint
    if (request.url.includes('/chat')) {
      return new Response(JSON.stringify({
        response: "I'm currently offline, but I'll be back soon! Your message has been saved and I'll respond when connectivity is restored.",
        suggested_questions: [
          "What's the best travel credit card in Canada?",
          "Help me compare cashback vs rewards cards",
          "How do I build credit as a newcomer to Canada?"
        ],
        confidence: 0.5,
        offline: true
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    throw error;
  }
}

/**
 * Handle static assets - Cache First
 */
async function handleStaticAsset(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  
  if (cached) {
    return cached;
  }
  
  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.log('💾 NAVUS SW: Failed to fetch static asset:', request.url);
    throw error;
  }
}

/**
 * Handle page requests - Network First with cache fallback
 */
async function handlePageRequest(request) {
  const cache = await caches.open(CACHE_NAME);
  
  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.log('🔄 NAVUS SW: Network failed for page, trying cache:', request.url);
    const cached = await cache.match(request);
    
    if (cached) {
      return cached;
    }
    
    // Fallback to index.html for SPA routing
    return cache.match('/index.html') || cache.match('/');
  }
}

/**
 * Check if request is for a static asset
 */
function isStaticAsset(url) {
  return url.includes('/static/') ||
         url.includes('.png') ||
         url.includes('.svg') ||
         url.includes('.ico') ||
         url.includes('.css') ||
         url.includes('.js') ||
         url.includes('.woff') ||
         url.includes('.woff2');
}

// Background sync for offline chat messages
self.addEventListener('sync', (event) => {
  console.log('🔄 NAVUS SW: Background sync triggered:', event.tag);
  
  if (event.tag === 'chat-sync') {
    event.waitUntil(syncOfflineMessages());
  }
});

/**
 * Sync offline messages when connectivity is restored
 */
async function syncOfflineMessages() {
  try {
    // Get offline messages from IndexedDB or localStorage
    const clients = await self.clients.matchAll();
    
    clients.forEach(client => {
      client.postMessage({
        type: 'SYNC_OFFLINE_MESSAGES'
      });
    });
    
    console.log('✅ NAVUS SW: Offline messages sync completed');
  } catch (error) {
    console.error('❌ NAVUS SW: Failed to sync offline messages:', error);
  }
}

// Push notifications (future enhancement)
self.addEventListener('push', (event) => {
  console.log('🔔 NAVUS SW: Push notification received:', event.data?.text());
  
  if (event.data) {
    const data = event.data.json();
    
    event.waitUntil(
      self.registration.showNotification(data.title || 'NAVUS', {
        body: data.body || 'You have a new message',
        icon: '/icon-192x192.png',
        badge: '/icon-96x96.png',
        data: data,
        actions: [
          {
            action: 'open',
            title: 'Open NAVUS'
          },
          {
            action: 'dismiss',
            title: 'Dismiss'
          }
        ]
      })
    );
  }
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  console.log('🔔 NAVUS SW: Notification clicked:', event.action);
  
  event.notification.close();
  
  if (event.action === 'open' || !event.action) {
    event.waitUntil(
      self.clients.openWindow('/')
    );
  }
});

console.log('🚀 NAVUS PWA Service Worker loaded successfully!');