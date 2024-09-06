/* Registers a service worker with given list of `resources`.
 *
 * This uses localforage to store the list of resources in an indexedDB (that the service worker can
 * also access) with a unique key based on a hash of the list contents. It then registers the
 * service worker at given `path` with a search param of nkow_param={unique_key}, which lets the
 * worker know where to read the list of resources from.
 */
const registerServiceWorker = async (resources, path='/static/offline-worker.js') => {
  console.log('registering service worker with resources:', resources, path);
  try {
    const id = 'test';
    path += `?nkow_param=${id}`;
    const key = `nk-offline-worker-${id}`;
    console.log(`Storing list of resources for service worker:`, id, key, resources, path);
    await localforage.setItem(key, resources);
    const registration = await navigator.serviceWorker.register(path, {scope: '/'});
    if (registration.installing) {
      console.log(`Service worker ${path} installing`);
    } else if (registration.waiting) {
      console.log(`Service worker ${path} installed`);
    } else if (registration.active) {
      console.log(`Service worker ${path} active`);
    }
  } catch (error) {
    console.error(`Registration of ${path} failed with ${error}`);
  }
};

/* Does a 'sane' comparison of 2 values, returning -1, 0, or 1.
 *
 * You almost always want this when sorting an array (as the 3rd argument), because the default uses
 * string comparison (!)
 *
 * For simple usage: call this as `myArray.sort(sane_sort)`
 *
 * You can optionally pass in a `key`, which is one of:
 * - undefined/null: compare `a` and `b` directly
 * - string: key in object
 * - int: index in array
 * - function: applied to a and b to get values
 *
 * For usage with a key, call this as `myArray.sort((a, b) => sane_sort(a, b, key))`
 *
 * This also compares arrays sanely, by first comparing lengths, then each element.
 *
 */
function sane_sort(a, b, key=null) {
  if (key !== undefined && key !== null) {
    //console.log('trying to sort', a, b, key, key instanceof Function);
    a = key instanceof Function ? key(a) : a[key];
    b = key instanceof Function ? key(b) : b[key];
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    const length_diff = a.length - b.length;
    if (length_diff !== 0) {
      return length_diff;
    }
    for (let i = 0; i < a.length; i++) {
      if (a[i] < b[i]) return -1;
      if (b[i] < a[i]) return 1;
    }
    return 0;
  }
  if (a < b) return -1;
  if (b < a) return 1;
  return 0;
}

// Equivalent to python's strftime
function strftime(date, fmt) {
  const pad = (n) => n.toString().padStart(2, '0');
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                  'Sep', 'Oct', 'Nov', 'Dec'];
  const repl = {
    'a': days[date.getDay()],
    'A': days[date.getDay()],
    'b': months[date.getMonth()],
    'B': months[date.getMonth()],
    'd': pad(date.getDate()),
    'H': pad(date.getHours()),
    'I': pad(date.getHours() % 12),
    'm': pad(date.getMonth() + 1),
    'M': pad(date.getMinutes()),
    'p': date.getHours() < 12 ? 'AM' : 'PM',
    'S': pad(date.getSeconds()),
    'w': date.getDay(),
    'y': date.getFullYear().toString().slice(2),
    'Y': date.getFullYear(),
    '%': '%',
  };
  let out = '';
  let i = 0;
  while (i < fmt.length) {
    if (fmt[i] === '%') {
      const c = fmt[i+1];
      if (repl[c]) {
        out += repl[c];
        i += 2;
      } else {
        out += c;
        i += 1;
      }
    } else {
      out += fmt[i];
      i += 1;
    }
  }
  return out;
}

// makes a link to an object, with the child as the text
const makeLink = (obj, child, idx, kw) => {
  return (
    <a href={obj.url} target="_blank" key={idx} {...kw}>{child}</a>
  );
}

// hash a string to a number
const hashString = (str) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

// convert a number to a unique rgb color as hex
const numberToColor = (num) => {
  let r = (num & 0xFF0000) >> 16;
  let g = (num & 0x00FF00) >> 8;
  let b = num & 0x0000FF;
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}
