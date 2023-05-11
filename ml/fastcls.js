'use strict';

const T = React.createElement;

let MAX_CLS = 50;

const DRAG_FMT = 'data/drag';

let CFG = null;

function set_config(cfg){
    CFG = Object.assign({}, cfg);
    if (CFG.max_cls){
        MAX_CLS = CFG.max_cls;
    }
}

function shuffle(array, seed) {                // <-- ADDED ARGUMENT
  var m = array.length, t, i;

  // While there remain elements to shuffle…
  while (m) {

    // Pick a remaining element…
    i = Math.floor(random(seed) * m--);        // <-- MODIFIED LINE

    // And swap it with the current element.
    t = array[m];
    array[m] = array[i];
    array[i] = t;
    ++seed                                     // <-- ADDED LINE
  }

  return array;
}

function random(seed) {
  var x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

// generate unique UUID
// from: http://stackoverflow.com/questions/105034/create-guid-uuid-in-javascript
function guid() {
    const s4 = function(){
        return Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
    }
    return s4() + s4() + '-' + s4() + '-' + s4() + '-' + s4() + '-' + s4() + s4() + s4();
}

// return true if object is an array
function is_array(obj){
    return Array.isArray(obj);
}


/* Does a 'sane' comparison of 2 values, returning -1, 0, or 1.
 *
 * You almost always want this when sorting an array (as the 3rd argument), because the default uses
 * string comparison (!)
 *
 * You can optionally pass in a `key`, which is one of:
 * - undefined/null: compare `a` and `b` directly
 * - string: key in object
 * - int: index in array
 * - function: applied to a and b to get values
 *
 * This also compares arrays sanely, by first comparing lengths, then each element.
 *
 */
function sane_sort(a, b, key){
    if (key !== undefined && key !== null){
        //console.log('trying to sort', a, b, key);
        a = (key instanceof Function) ? key(a) : a[key];
        b = (key instanceof Function) ? key(b) : b[key];
    }
    if (is_array(a) && is_array(b)){
        const length_diff = a.length - b.length;
        if (length_diff !== 0){
            return length_diff;
        }
        for (let i = 0; i < a.length; i++){
            if (a[i] < b[i]) return -1;
            if (b[i] < a[i]) return 1;
        }
        return 0;
    }
    if (a < b) return -1;
    if (b < a) return 1;
    return 0;
}

// Simple debouncer
function debounce(func, timeout = 1000){
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => { func.apply(this, args); }, timeout);
  };
}

// creates the filters ux
function make_filters_ux(){
    const ret = [];
    ret.push(T('label', {}, 'Filter:'));
    console.log('in make filters ux', this.state);
    Object.entries(this.state.filter_btns).forEach(([field, value]) => {
        console.log('mapping with ', field, value);
          ret.push(T('select', {
                    key: field,
                    value,
                    onChange: (ev) => this.filter_ids(null, field, ev.target.value),
                },
              T('option', {value: ''}, `<${field}>`),
              CFG.filter_options[field].map(name => T('option', {key: name, value: name}, name)),
          ));
    });
    ret.push(T('input', {
        type: 'text',
        size: 40,
        value: this.state.filter_text,
        onChange: (ev) => this.filter_ids(ev.target.value)
    }));
    return ret;
}

// creates the search index from given dict of `items`
function createSearchIndex(items){
    const builder = new lunr.Builder();
    let fields = null;
    if (CFG.search_fields){
        fields = CFG.search_fields.slice();
    } else {
        for (const [id, item] of Object.entries(items)) {
            const cur = Object.keys(item);
            if (fields === null){
                fields = cur;
            } else {
                fields = fields.filter(f => cur.includes(f));
            }
        }
    }
    console.log('Got search fields', fields);
    fields.forEach(field => builder.field(field));
    for (const [id, item] of Object.entries(items)) {
        item.id = id;
        builder.add(item);
    }
    return builder.build();
}


// Runs a text search given a set of input ids and `search_index`, returning subset of ids
function text_search(q, ids, search_index){
    if (!search_index){
        console.log('Warning, no search index, searching manually');
        ids = ids.filter(id => id.match(q));
    } else {
        const results = search_index.query((query) => {
            const {LEADING, TRAILING} = lunr.Query.wildcard;
            q.split(/(\s+)/).forEach(s => {
                if (s){
                    query = query.term(s, {wildcard: LEADING|TRAILING});
                }
            });
            return query;
        });
        const valid_ids = {};
        results.forEach(r => valid_ids[r.ref] = 1);
        ids = ids.filter(id => valid_ids[id]);
    }
    return ids;
}


function filter_items(state, items, search_index, filter_text=null, filter_btn_key=null, filter_btn_value=null){
    console.log('in filter items with ', state, filter_text, filter_btn_key, filter_btn_value);
    if (filter_text == null){
        filter_text = state.filter_text;
    }
    let filter_btns = state.filter_btns;
    if (filter_btn_key != null){
        // duplicate this so that we update our state accordingly
        filter_btns = Object.assign({}, filter_btns);
        filter_btns[filter_btn_key] = filter_btn_value;
    }
    let ids = Object.keys(items)
    // apply filter buttons
    ids = ids.filter(id => {
        let keep = true;
        for (const [key, value] of Object.entries(filter_btns)) {
            if (value === '') continue;
            const item = items[id];
            if (item[key] !== value){
                keep = false;
                break;
            }
        }
        return keep;
    });
    // apply text filter
    ids = text_search(filter_text, ids, search_index);
    const ret = {
        filter_text,
        filter_btns,
        ids,
    };
    console.log('return filter items', ret);
    return ret;
}



// generic item renderer
function item_renderer(props){
    let {item, hover=null, hoverHandler=null, clickHandler=null, example_type='', source=''} = props;
    const {id} = item;
    if (0 && source.includes('cls')){
        console.log('trying to render item', item, id);
    }
    const onDragStart = (ev) => {
        const data = JSON.stringify(Object.assign({}, ev.currentTarget.dataset));
        ev.dataTransfer.setData(DRAG_FMT, data);
        console.log('drag start', ev.currentTarget, data, ev.dataTransfer.getData(DRAG_FMT));
    }
    const opts = {className:'item-div', 'data-id': id, 'data-source': source, key: id, draggable: true, onDragStart};
    if (clickHandler){
        opts.style = {position: 'relative'};
        switch (example_type) {
            case 'pos':
                // for pos examples, clicking removes from pos
                opts.onClick = () => clickHandler(id, true);
                break;
            case 'neg':
                // for neg examples, clicking removes from neg
                opts.onClick = () => clickHandler(id, false);
                break;
        }
        if (hoverHandler){
            opts.onMouseOver = () => hoverHandler(id);
            opts.onMouseOut = () => hoverHandler(null);
        }
    }
    // figure out what to render based on what fields we have
    let {link='', image_url='', thumb_url='', desc='', name=''} = item;
    let inner = '';
    const make_url = (url) => url.startsWith('http') ? url : 'static/'+url;
    // if we have an image or thumb url, then we want an image
    if (image_url || thumb_url){
        inner = T('img', {src: make_url(thumb_url || image_url)});
    } else if (name) {
        inner = T('span', {}, name);
    }
    // if we have a link of any sort, we want to wrap the inner in a link
    if (link || image_url || thumb_url){
        inner = T('a', {target: '_blank', href: make_url(link || image_url || thumb_url)}, inner);
    }
    let bottom = null;
    if (desc){
        bottom = T('div', {className: 'item-desc'}, desc);
    }
    return T('div', opts, inner, bottom);
}


// viewer for a collection -- this also has the logic for classification
class Collection extends React.Component {
    constructor(props) {
        super(props);
        console.log('initializing collection with', this.props);
        this.make_filters_ux = make_filters_ux.bind(this);
        this.state = {
            uuid: guid(),
            pos: [],
            neg: [],
            cls: [],
            filter_cls: [],
            drag_over: null,
            filter_text: '',
            filter_btns: Object.assign({}, CFG.filter_btns),
            search_index: null,
        };
    }

    // after prop/state changes
    componentDidUpdate(prevProps, prevState, snapshot){
        if (prevState.pos != this.state.pos || prevState.neg != this.state.neg){
            this.classify(Object.assign({}, this.state, this.props));
        }
    }

    // returns item objects from ids
    id2item = (id) => {
        let item = this.props.items[id];
        //console.log('for id ', id, item);
        return {id, ...item};
    }

    // classifies our items based on pos & neg in `obj`
    classify = (obj) => {
        const {pos, neg, type} = obj;
        console.log('classifying with ', pos, neg, type);
        if (pos.length == 0){
            this.setState({cls: []});
            return;
        }
        const data = {pos, neg, type};
        let url = `classify`;
        console.log('Classifying', obj, url);
        fetch(url, {method: 'post', body: JSON.stringify(data)})
            .then(ret => ret.json())
            .then(obj => {
                console.log('got back cls', obj, Object.keys(this.props.items));
                if (!obj.cls || !is_array(obj.cls)){
                    this.setState({cls: []});
                } else {
                    let {cls} = obj;
                    let max_num = MAX_CLS;
                    switch(this.props.type) {
                        case 'plus-minus':
                            // map to ids
                            cls = cls.map(([id, score]) => id);
                            // filter out pos and neg
                            cls = cls.filter(id => (pos.indexOf(id) < 0 && neg.indexOf(id) < 0));
                            break;
                        case 'rel':
                            console.log('in rel');
                            // assemble pairs into a single list
                            const ret = [];
                            cls.forEach(item => {
                                const [[a, b], score] = item;
                                //console.log('evaluating ', a, b, score, ret);
                                // skip if we've seen either item in the pair before
                                if (ret.includes(a) || ret.includes(b)) return;
                                ret.push(a, b, 'spacer');
                            });
                            max_num = MAX_CLS * 3;
                            cls = ret;
                            break;
                    }
                    // limit number
                    cls = cls.slice(0, max_num);
                    if (this.props.type === 'plus-minus'){
                        // build search index
                        const items = {};
                        cls.forEach(id => {
                            items[id] = this.id2item(id);
                        });
                        const search_index = createSearchIndex(items);
                        // rerun existing search (if any)
                        this.state.cls = cls;
                        this.state.search_index = search_index;
                        this.filter_ids();
                    } else {
                        this.setState({cls: cls, filter_cls: cls, search_index: null});
                    }
                }
            })
            .catch((err) => {
                console.log('got error when converting resp to json', err);
                this.setState({cls: []});
            })
    }

    // needed to make things droppable, but we don't actually need to do anything
    onDragOver = (ev, type) => {
        ev.preventDefault();
        this.setState({drag_over: type});
    }

    onDrop = (ev, type) => {
        ev.preventDefault();
        // Get the id of the target and add the moved element to the target's DOM
        const data = JSON.parse(ev.dataTransfer.getData(DRAG_FMT));
        //ev.target.appendChild(document.getElementById(data));
        console.log('on drop', data);
        const newState = {drag_over: false};
        if (data.source){
            const [uuid, key] = data.source.split('/');
            if (uuid == this.state.uuid){
                newState[key] = immer.produce(this.state[key], draft => draft.filter(el => el != data.id));
            }
        }
        const cur = this.state[type];
        if (cur && type != 'cls'){
            newState[type] = immer.produce(cur, draft => {
                if (!draft.includes(data.id)){
                    draft.push(data.id);
                }
                return draft;
            });
        }
        //console.log('got new state', newState);
        this.setState(newState);
    }

    // filters our items
    filter_ids = (filter_text=null, filter_btn_key=null, filter_btn_value=null) => {
        const items = {};
        this.state.cls.forEach(id => {items[id] = this.id2item(id)});
        const ret = filter_items(this.state, items, this.state.search_index, filter_text, filter_btn_key, filter_btn_value);
        this.setState({
            filter_text: ret.filter_text,
            filter_btns: ret.filter_btns,
            filter_cls: ret.ids,
        });
    }

    render(){
        const {title, desc, type, clickHandler=null} = this.props;
        const {drag_over} = this.state;
        const make_div = (key, bg, lst) => {
            return T('div', {
                            key,
                            className: `collection ${key} collection-${type} user ${(drag_over == key) ? 'dragover' : ''}`,
                            onDragOver: (ev) => this.onDragOver(ev, key),
                            onDrop: (ev) => this.onDrop(ev, key),
                            style: {backgroundColor: bg},
                        },
                    lst.map(this.id2item).map(item => item_renderer({
                        item,
                        clickHandler,
                        example_type: key,
                        source: `${this.state.uuid}/${key}`,
                    })));
        };
        let pos, neg = null;
        if (type == 'plus-minus' || type == 'rel'){
            pos = make_div('pos', 'lightblue', this.state.pos);
            neg = make_div('neg', 'lightpink', this.state.neg);
        }
        const cls = make_div('cls', 'white', this.state.filter_cls);
        const results = T('div', {style: {overflowX: 'auto', textAlign: 'center'}},
                     T('div', {className: 'cls-controls'}, this.make_filters_ux()),
                     cls);
        const els = [];
        switch(type) {
            case 'plus-minus':
                els.push(pos, results, neg);
                break;
            case 'rel':
                els.push(pos, neg, results);
                break;
        }
        return T('div', {className: `collection-div`, key: title},
                 T('h3', {}, `${title} (${type})`),
                 T('div', {}, desc),
                 T('div', {className:`collection-container`}, els));
    }
}

class AllItems extends React.Component {
    constructor(props) {
        super(props);
        console.log('calling all items constructor');
        this.make_filters_ux = make_filters_ux.bind(this);
        this.state = this.set_params(props);
    }

    shouldComponentUpdate(nextProps, nextState){
        if (nextProps.search_index !== this.props.search_index || nextProps.items !== this.props.items){
            console.log('there was a change', this.props, nextProps, this.props.items === nextProps.items, this.props.search_index === nextProps.search_index);
            this.setState(this.set_params(nextProps));
        }
        return true;
    }

    // sets various pagination-related params based on given `props`
    set_params = (props) => {
        const {page_size=100} = props;
        const ids = Object.keys(props.items)
        const ret = {
            page_size,
            page_num: 0,
            last_page: Math.max(0, Math.ceil(ids.length / page_size) - 1),
            hover: null,
            ids,
            filter_text: '',
            filter_btns: Object.assign({}, CFG.filter_btns),
        };
        console.log('setting params', ret, CFG);
        return ret;
    }

    hoverHandler = (id) => {
        this.setState({hover: id});
    }

    // increments our page number, or set it directly
    incr = (n_pages, is_relative) => {
        let {page_num, page_size, ids, last_page} = this.state;
        if (is_relative){
            page_num += n_pages;
        } else {
            page_num = n_pages;
        }
        page_num = Math.max(Math.min(page_num, last_page), 0);
        //console.log('incr', page_num, page_size, n_pages, last_page);
        this.setState({page_num});
    }

    // returns item objects from ids
    id2item = (id) => {
        return {id, ...this.props.items[id]};
    }

    // filters our items
    filter_ids = (filter_text=null, filter_btn_key=null, filter_btn_value=null) => {
        const ret = filter_items(this.state, this.props.items, this.props.search_index, filter_text, filter_btn_key, filter_btn_value);
        ret.page_num = 0;
        ret.last_page = Math.ceil(ret.ids.length / this.state.page_size) - 1;
        this.setState(ret);
    }

    render_nav(){
        const {items} = this.props;
        const {page_num, page_size, last_page} = this.state;
        let {ids} = this.state
        //console.log(page_num, page_size, page_num*page_size, (page_num+1)*page_size);
        const idx0 = page_num*page_size;
        const idx1 = (page_num+1)*page_size;
        ids = ids.slice(idx0, idx1);
        let pages = [...Array(10).keys()].map(i => i+page_num-5).filter(p => (p >= 0 && p<= last_page));
        pages.push(0, last_page);
        pages = [...new Set(pages)];
        pages.sort(sane_sort);
        const nav = T('div', {className: 'controls'},
            T('span', {}, `Items ${idx0+1} - ${idx1}`),
            T('div', {className: 'nav'},
              T('label', {}, 'Page nav: '),
              T('button', {onClick: () => this.incr(-1, true)}, 'Prev'),
              pages.map(p => {
                  if (p == page_num){
                      return T('span', {key: p}, p+1);
                  } else {
                      return T('button', {key: p, onClick: () => this.incr(p, false)}, p+1);
                  }
              }),
              T('button', {onClick: () => this.incr(1, true)}, 'Next'),
            ),
            T('div', {className: 'buttons'},
              T('button', {onClick: () => this.setState({ids: shuffle(this.state.ids, (new Date()).getTime())})}, 'Shuffle items'),
              this.make_filters_ux(),
            ),
        );
        return [nav, ids];
    }

    render(){
        const {clickHandler} = this.props;
        const {hover} = this.state;
        const [nav, ids] = this.render_nav();
        //console.log('rendering all images', ids)
        return T('div', {},
                 nav,
                 T('div', {className: 'all-div'},
                     ids.map(this.id2item).map(item => item_renderer({
                            item,
                            hover,
                            clickHandler,
                            hoverHandler: this.hoverHandler,
                        }))),
                 nav,
        );
    }
}

class Main extends React.Component {
    constructor(props) {
        super(props);
        set_config(props.cfg);
        // helper to randomly choose from array
        const choice = (arr) => arr[Math.floor(Math.random() * arr.length)];
        this.state = {items: [], collections: [], search_index: null};
    }

    // returns a list of ids based on our props
    ids(){
        return Object.keys(this.state.items);
    }

    // adds a new collection
    add_collection = (type='plus-minus') => {
        const {collections} = this.state;
        this.setState({collections: immer.produce(collections, draft => {
            draft.push({
                title: `Collection ${collections.length+1}`,
                desc: '',
                type,
                //type: 'plus-minus',
                //type: 'rel',
            });
            return draft;
        })});
    }

    // loads data
    load_data(){
        fetch('items').then(resp => resp.json()).then(items => {
            // update filter buttons and options based on this data
            const filter_buttons = CFG.filter_buttons || [];
            CFG.filter_options = {};
            CFG.filter_btns = {};
            filter_buttons.forEach(field => {
                CFG.filter_btns[field] = '';
                CFG.filter_options[field] = new Set();
            });
            // make a list of all filter options for each filter button field
            for (const [id, item] of Object.entries(items)) {
                filter_buttons.forEach(field => {
                    const value = item[field];
                    if (!(field in CFG.filter_options)){
                        CFG.filter_options[field] = new Set();
                    }
                    CFG.filter_options[field].add(value);
                });
            }
            // convert each filter option set to an array
            for (const [field, opts] of Object.entries(CFG.filter_options)) {
                CFG.filter_options[field] = Array.from(opts);
            }
            // set our state
            this.setState({items, search_index: createSearchIndex(items)});
        });
    }

    render(){
        const {items, search_index} = this.state;
        console.log('rendering main', Object.keys(items).length, this.state, this.props.cfg, new Date());
        const {collections} = this.state;
        const ret = T('div', {className: 'main'},
                   collections.map(c => T(Collection, {items, ...c})),
                   T('button', {onClick: () => this.add_collection('plus-minus')}, 'Add +/- collection'),
                   T('button', {onClick: () => this.add_collection('rel')}, 'Add rel collection'),
                   T('h3', {}, "All items"),
                   T(AllItems, {items, search_index}));
        console.log('done with main', new Date());
        return ret;
    }

    // after first mount, load data add a collection
    componentDidMount(){
        this.load_data();
    }

}


ReactDOM.render(T(Main, data), document.querySelector('#main'));
