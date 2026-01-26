(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? module.exports = factory() :
    typeof define === 'function' && define.amd ? define(factory) :
    (global = global || self, global.intersectionObserverAdmin = factory());
}(this, function () { 'use strict';

    class Registry {
        constructor() {
            this.registry = new WeakMap();
        }
        elementExists(elem) {
            return this.registry.has(elem);
        }
        getElement(elem) {
            return this.registry.get(elem);
        }
        /**
         * administrator for lookup in the future
         *
         * @method add
         * @param {HTMLElement | Window} element - the item to add to root element registry
         * @param {IOption} options
         * @param {IOption.root} [root] - contains optional root e.g. window, container div, etc
         * @param {IOption.watcher} [observer] - optional
         * @public
         */
        addElement(element, options) {
            if (!element) {
                return;
            }
            this.registry.set(element, options || {});
        }
        /**
         * @method remove
         * @param {HTMLElement|Window} target
         * @public
         */
        removeElement(target) {
            this.registry.delete(target);
        }
        /**
         * reset weak map
         *
         * @method destroy
         * @public
         */
        destroyRegistry() {
            this.registry = new WeakMap();
        }
    }

    const noop = () => { };
    var CallbackType;
    (function (CallbackType) {
        CallbackType["enter"] = "enter";
        CallbackType["exit"] = "exit";
    })(CallbackType || (CallbackType = {}));
    class Notifications {
        constructor() {
            this.registry = new Registry();
        }
        /**
         * Adds an EventListener as a callback for an event key.
         * @param type 'enter' or 'exit'
         * @param key The key of the event
         * @param callback The callback function to invoke when the event occurs
         */
        addCallback(type, element, callback) {
            let entry;
            if (type === CallbackType.enter) {
                entry = { [CallbackType.enter]: callback };
            }
            else {
                entry = { [CallbackType.exit]: callback };
            }
            this.registry.addElement(element, Object.assign({}, this.registry.getElement(element), entry));
        }
        removeElementNotification(element) {
            this.registry.removeElement(element);
        }
        elementNotificationExists(element) {
            return Boolean(this.registry.elementExists(element));
        }
        /**
         * @hidden
         * Executes registered callbacks for key.
         * @param type
         * @param element
         * @param data
         */
        dispatchCallback(type, element, data) {
            if (type === CallbackType.enter) {
                const { enter = noop } = this.registry.getElement(element);
                enter(data);
            }
            else {
                // no element in WeakMap possible because element may be removed from DOM by the time we get here
                const found = this.registry.getElement(element);
                if (found && found.exit) {
                    found.exit(data);
                }
            }
        }
    }

    class IntersectionObserverAdmin extends Notifications {
        constructor() {
            super();
            this.elementRegistry = new Registry();
        }
        /**
         * Adds element to observe via IntersectionObserver and stores element + relevant callbacks and observer options in static
         * administrator for lookup in the future
         *
         * @method observe
         * @param {HTMLElement | Window} element
         * @param {Object} options
         * @public
         */
        observe(element, options = {}) {
            if (!element) {
                return;
            }
            this.elementRegistry.addElement(element, Object.assign({}, options));
            this.setupObserver(element, Object.assign({}, options));
        }
        /**
         * Unobserve target element and remove element from static admin
         *
         * @method unobserve
         * @param {HTMLElement|Window} target
         * @param {Object} options
         * @public
         */
        unobserve(target, options) {
            const matchingRootEntry = this.findMatchingRootEntry(options);
            if (matchingRootEntry) {
                this.clearRootEntry(target, matchingRootEntry);
            }
            else {
                this.removeElement(target);
                this.clearDefaultRoot(target);
            }
        }
        /**
         * register event to handle when intersection observer detects enter
         *
         * @method addEnterCallback
         * @public
         */
        addEnterCallback(element, callback) {
            this.addCallback(CallbackType.enter, element, callback);
        }
        /**
         * register event to handle when intersection observer detects exit
         *
         * @method addExitCallback
         * @public
         */
        addExitCallback(element, callback) {
            this.addCallback(CallbackType.exit, element, callback);
        }
        /**
         * retrieve registered callback and call with data
         *
         * @method dispatchEnterCallback
         * @public
         */
        dispatchEnterCallback(element, entry) {
            this.dispatchCallback(CallbackType.enter, element, entry);
        }
        /**
         * retrieve registered callback and call with data on exit
         *
         * @method dispatchExitCallback
         * @public
         */
        dispatchExitCallback(element, entry) {
            this.dispatchCallback(CallbackType.exit, element, entry);
        }
        /**
         * cleanup data structures and unobserve elements
         *
         * @method destroy
         * @public
         */
        destroy() {
            this.elementRegistry.destroyRegistry();
        }
        /**
         * cleanup removes provided elements from both registries
         *
         * @method removeElement
         * @public
         *
         */
        removeElement(element) {
            this.removeElementNotification(element);
            this.elementRegistry.removeElement(element);
        }
        /**
         * checks whether element exists in either registry
         *
         * @method elementExists
         * @public
         *
         */
        elementExists(element) {
            return Boolean(this.elementNotificationExists(element) ||
                this.elementRegistry.elementExists(element));
        }
        /**
         * use function composition to curry options
         *
         * @method setupOnIntersection
         * @param {Object} options
         */
        setupOnIntersection(options) {
            return (ioEntries) => {
                return this.onIntersection(options, ioEntries);
            };
        }
        setupObserver(element, options) {
            const { root = window } = options;
            // First - find shared root element (window or target HTMLElement)
            // this root is responsible for coordinating it's set of elements
            const potentialRootMatch = this.findRootFromRegistry(root);
            // Second - if there is a matching root, see if an existing entry with the same options
            // regardless of sort order. This is a bit of work
            let matchingEntryForRoot;
            if (potentialRootMatch) {
                matchingEntryForRoot = this.determineMatchingElements(options, potentialRootMatch);
            }
            // next add found entry to elements and call observer if applicable
            if (matchingEntryForRoot) {
                const { elements, intersectionObserver } = matchingEntryForRoot;
                elements.push(element);
                if (intersectionObserver) {
                    intersectionObserver.observe(element);
                }
            }
            else {
                // otherwise start observing this element if applicable
                // watcher is an instance that has an observe method
                const intersectionObserver = this.newObserver(element, options);
                const observerEntry = {
                    elements: [element],
                    intersectionObserver,
                    options
                };
                // and add entry to WeakMap under a root element
                // with watcher so we can use it later on
                const stringifiedOptions = this.stringifyOptions(options);
                if (potentialRootMatch) {
                    // if share same root and need to add new entry to root match
                    // not functional but :shrug
                    potentialRootMatch[stringifiedOptions] = observerEntry;
                }
                else if (!this.elementRegistry.elementExists(root)) {
                    // no root exists, so add to WeakMap
                    this.elementRegistry.addElement(root, {
                        [stringifiedOptions]: observerEntry
                    });
                }
            }
        }
        newObserver(element, options) {
            // No matching entry for root in static admin, thus create new IntersectionObserver instance
            const { root, rootMargin, threshold } = options;
            const newIO = new IntersectionObserver(this.setupOnIntersection(options).bind(this), { root, rootMargin, threshold });
            newIO.observe(element);
            return newIO;
        }
        /**
         * IntersectionObserver callback when element is intersecting viewport
         * either when `isIntersecting` changes or `intersectionRadio` crosses on of the
         * configured `threshold`s.
         * Exit callback occurs eagerly (when element is initially out of scope)
         * See https://stackoverflow.com/questions/53214116/intersectionobserver-callback-firing-immediately-on-page-load/53385264#53385264
         *
         * @method onIntersection
         * @param {Object} options
         * @param {Array} ioEntries
         * @private
         */
        onIntersection(options, ioEntries) {
            ioEntries.forEach(entry => {
                const { isIntersecting, intersectionRatio } = entry;
                let threshold = options.threshold || 0;
                if (Array.isArray(threshold)) {
                    threshold = threshold[threshold.length - 1];
                }
                // then find entry's callback in static administration
                const matchingRootEntry = this.findMatchingRootEntry(options);
                // first determine if entry intersecting
                if (isIntersecting || intersectionRatio > threshold) {
                    if (matchingRootEntry) {
                        matchingRootEntry.elements.some((element) => {
                            if (element && element === entry.target) {
                                this.dispatchEnterCallback(element, entry);
                                return true;
                            }
                            return false;
                        });
                    }
                }
                else {
                    if (matchingRootEntry) {
                        matchingRootEntry.elements.some((element) => {
                            if (element && element === entry.target) {
                                this.dispatchExitCallback(element, entry);
                                return true;
                            }
                            return false;
                        });
                    }
                }
            });
        }
        /**
         * { root: { stringifiedOptions: { observer, elements: []...] } }
         * @method findRootFromRegistry
         * @param {HTMLElement|Window} root
         * @private
         * @return {Object} of elements that share same root
         */
        findRootFromRegistry(root) {
            if (this.elementRegistry) {
                return this.elementRegistry.getElement(root);
            }
        }
        /**
         * We don't care about options key order because we already added
         * to the static administrator
         *
         * @method findMatchingRootEntry
         * @param {Object} options
         * @return {Object} entry with elements and other options
         */
        findMatchingRootEntry(options) {
            const { root = window } = options;
            const matchingRoot = this.findRootFromRegistry(root);
            if (matchingRoot) {
                const stringifiedOptions = this.stringifyOptions(options);
                return matchingRoot[stringifiedOptions];
            }
        }
        /**
         * Determine if existing elements for a given root based on passed in options
         * regardless of sort order of keys
         *
         * @method determineMatchingElements
         * @param {Object} options
         * @param {Object} potentialRootMatch e.g. { stringifiedOptions: { elements: [], ... }, stringifiedOptions: { elements: [], ... }}
         * @private
         * @return {Object} containing array of elements and other meta
         */
        determineMatchingElements(options, potentialRootMatch) {
            const matchingStringifiedOptions = Object.keys(potentialRootMatch).filter(key => {
                const { options: comparableOptions } = potentialRootMatch[key];
                return this.areOptionsSame(options, comparableOptions);
            })[0];
            return potentialRootMatch[matchingStringifiedOptions];
        }
        /**
         * recursive method to test primitive string, number, null, etc and complex
         * object equality.
         *
         * @method areOptionsSame
         * @param {any} a
         * @param {any} b
         * @private
         * @return {boolean}
         */
        areOptionsSame(a, b) {
            if (a === b) {
                return true;
            }
            // simple comparison
            const type1 = Object.prototype.toString.call(a);
            const type2 = Object.prototype.toString.call(b);
            if (type1 !== type2) {
                return false;
            }
            else if (type1 !== '[object Object]' && type2 !== '[object Object]') {
                return a === b;
            }
            if (a && b && typeof a === 'object' && typeof b === 'object') {
                // complex comparison for only type of [object Object]
                for (const key in a) {
                    if (Object.prototype.hasOwnProperty.call(a, key)) {
                        // recursion to check nested
                        if (this.areOptionsSame(a[key], b[key]) === false) {
                            return false;
                        }
                    }
                }
            }
            // if nothing failed
            return true;
        }
        /**
         * Stringify options for use as a key.
         * Excludes options.root so that the resulting key is stable
         *
         * @param {Object} options
         * @private
         * @return {String}
         */
        stringifyOptions(options) {
            const { root } = options;
            const replacer = (key, value) => {
                if (key === 'root' && root) {
                    const classList = Array.prototype.slice.call(root.classList);
                    const classToken = classList.reduce((acc, item) => {
                        return (acc += item);
                    }, '');
                    const id = root.id;
                    return `${id}-${classToken}`;
                }
                return value;
            };
            return JSON.stringify(options, replacer);
        }
        clearRootEntry(target, rootState) {
            const { intersectionObserver } = rootState;
            intersectionObserver.unobserve(target);
            if (rootState.elements) {
                rootState.elements = rootState.elements.filter((el) => el !== target);
            }
            this.removeElement(target);
            this.clearDefaultRoot(target);
        }
        clearDefaultRoot(target) {
            const windowRoot = this.elementRegistry.getElement(window);
            if (windowRoot && windowRoot.elements) {
                windowRoot.elements = windowRoot.elements.filter((el) => el !== target);
            }
        }
    }

    return IntersectionObserverAdmin;

}));
