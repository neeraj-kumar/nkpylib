/* Offline service worker to enable offline access (or access without local network) for apps.
 *
 * This uses a url parameter "nkow_param" to reference a localforage key from which to read the
 * list of resources to use. In your main code, call `registerServiceWorker(resources)`, where
 * `resources` is a list of string urls (or rel-paths).
 *
 * Adapted from MDN's example at:
 * https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API/Using_Service_Workers
 *
 */

console.log('Service worker script loaded');

const CACHE_NAME = 'v1';

const FALLBACK_URL = 'nkow-fallback-url.json';

// we import localforage to allow easy access to IndexedDB, where we will read our list of resources
// to cache, as described in https://stackoverflow.com/questions/40387983/
self.importScripts('localforage.min.js');

// Adds the given list of resources to the cache
const addResourcesToCache = async (resources) => {
  const cache = await caches.open(CACHE_NAME);
  await cache.addAll(resources);
};

// Puts the given request-response pair into the cache
const putInCache = async (request, response) => {
  const cache = await caches.open(CACHE_NAME);
  await cache.put(request, response);
};

// Tries to get the resource from the cache first, then the network,
// and finally falls back to the given fallbackUrl.
//
// This also caches the resource if it's fetched from the network.
//
// If nothing works, we return a status 408 (Request Timeout) response.
const cacheFirst = async ({request, preloadResponsePromise, fallbackUrl}) => {
  // First try to get the resource from the cache
  const responseFromCache = await caches.match(request);
  if (responseFromCache) {
    return responseFromCache;
  }

  // Next try to use the preloaded response, if it's there
  // NOTE: Chrome throws errors regarding preloadResponse, see:
  // https://bugs.chromium.org/p/chromium/issues/detail?id=1420515
  // https://github.com/mdn/dom-examples/issues/145
  // To avoid those errors, remove or comment out this block of preloadResponse
  // code along with enableNavigationPreload() and the "activate" listener.
  const preloadResponse = await preloadResponsePromise;
  if (preloadResponse) {
    console.info('Using preload response', preloadResponse);
    putInCache(request, preloadResponse.clone());
    return preloadResponse;
  }

  // Next try to get the resource from the network
  try {
    const responseFromNetwork = await fetch(request.clone());
    // response may be used only once
    // we need to save clone to put one copy in cache
    // and serve second one
    putInCache(request, responseFromNetwork.clone());
    return responseFromNetwork;
  } catch (error) {
    const fallbackResponse = await caches.match(fallbackUrl);
    if (fallbackResponse) {
      return fallbackResponse;
    }
    // when even the fallback response is not available,
    // there is nothing we can do, but we must always
    // return a Response object
    return new Response(`Network error happened trying to fetch ${request}`, {
      status: 408,
      headers: { 'Content-Type': 'text/plain' },
    });
  }
};

// Enables navigation preloads if the browser supports it
const enableNavigationPreload = async () => {
  if (self.registration.navigationPreload) {
    // Enable navigation preloads!
    await self.registration.navigationPreload.enable();
  }
};

// Activate the service worker and enable navigation preloads
self.addEventListener('activate', (event) => {
  event.waitUntil(enableNavigationPreload());
});

/* Install the service worker and adds resources to the cache.
 *
 * The list of resources (string urls) is read via localforage, under the key:
 *  `nk-offline-worker-${nkow_param}`, where `nkow_param` is a search param for this url that you
 *  specify. (i.e., using .../offline-worker.js?nkow_param=blah)
 */
self.addEventListener('install', (event) => {
  console.log('Service worker installing...');
  // fetch the nkow param
  const nkow_param = new URL(self.location).searchParams.get('nkow_param') || '';
  const key = `nk-offline-work-${nkow_param}`;
  console.log(`Reading list of resources to cache for service worker from ${key}`);
  localForage.getItem(key, (err, resources) => {
    if (err == null) {
      console.log(`Error reading nkow list of resources from ${key}, doing nothing`);
    } else {
      event.waitUntil(
        addResourcesToCache(resources)
      );
    }
  });
});

/* Add the fetch listener to the service worker.
 *
 * This uses the global FALLBACK_URL.
 */
self.addEventListener('fetch', (event) => {
  event.respondWith(
    cacheFirst({
      request: event.request,
      preloadResponsePromise: event.preloadResponse,
      fallbackUrl: FALLBACK_URL,
    })
  );
});
