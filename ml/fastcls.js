'use strict';

const T = React.createElement;

const MAX_CLS = 10;

const DRAG_FMT = 'data/drag';

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

// generic item renderer
function item_renderer(props){
    let {item, hover=null, hoverHandler=null, clickHandler=null, example_type='', source=''} = props;
    const {id} = item;
    //console.log('trying to render item', item, id);
    let path = item.paths[0];
    path = path.replace('/images/', '/thumbs/');
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
    return T('div', opts, T('img', {src:`static/${path}`}));
}


// viewer for a collection -- this also has the logic for classification
class Collection extends React.Component {
    constructor(props) {
        super(props);
        console.log('initializing collection with', this.props);
        this.state = {uuid: guid(), pos: [], neg: [], cls: [], drag_over: null};
    }

    // after prop/state changes
    componentDidUpdate(prevProps, prevState, snapshot){
        if (prevState.pos != this.state.pos || prevState.neg != this.state.neg){
            this.classify(this.state);
        }
    }

    // returns item objects from ids
    id2item = (id) => {
        return {id, ...this.props.items[id]};
    }

    // classifies our items based on pos & neg in `obj`
    classify = (obj) => {
        const {pos, neg} = obj;
        console.log('classifying with ', pos, neg);
        if (pos.length == 0){
            this.setState({cls: []});
            return;
        }
        let url = 'classify?';
        if (pos.length > 0){
            url += 'pos=' + pos.join(',');
        }
        if (neg.length > 0){
            url += '&neg=' + neg.join(',');
        }
        console.log('Classifying', obj, url);
        fetch(url)
            .then(ret => ret.json())
            .then(obj => {
                console.log('got back cls', obj);
                if (!obj.cls){
                    this.setState({cls: []});
                } else {
                    let {cls} = obj;
                    // map to ids
                    cls = cls.map(([id, score]) => id);
                    // filter out pos and neg
                    cls = cls.filter(id => (pos.indexOf(id) < 0 && neg.indexOf(id) < 0));
                    // limit number
                    cls = cls.slice(0, MAX_CLS);
                    this.setState({cls});
                }
            });
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

    render(){
        const {title, desc, show_plus, show_minus, clickHandler=null} = this.props;
        const {drag_over} = this.state;
        const user_cls = (show_plus && show_minus) ? 'user' : '';
        const make_div = (key, bg, lst) => {
            return T('div', {
                            key,
                            className: `collection ${key} ${user_cls} ${(drag_over == key) ? 'dragover' : ''}`,
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
        const pos = (show_plus) ? make_div('pos', 'lightblue', this.state.pos) : null;
        const cls = make_div('cls', 'white', this.state.cls);
        const neg = (show_minus) ? make_div('neg', 'lightpink', this.state.neg) : null;
        return T('div', {className: `collection-div`, key: title},
                 T('h3', {}, title),
                 T('div', {}, desc),
                 T('div', {className:`collection-container`}, pos, cls, neg),
        );
    }
}

class AllItems extends React.Component {
    constructor(props) {
        super(props);
        const {page_size=100} = this.props;
        const ids = Object.keys(this.props.items)
        this.state = {
            page_size,
            page_num: 0,
            last_page: Math.ceil(ids.length / page_size) - 1,
            hover: null,
            ids,
            filter_text: '',
        };
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
    filter = (ev) => {
        const filter_text = ev.target.value;
        if (filter_text == this.state.filter_text){
            return;
        }
        let ids = Object.keys(this.props.items).filter(id => {
            // check id
            if (id.match(filter_text)) return true;
            // check item attr values
            const item = this.props.items[id];
            for (const [key, value] in Object.entries(item)) {
                try {
                    if (value.match(filter_text)) return true;
                } catch (e) {} // ignore errors
            }
            // at this point, nothing matched, so return false
            return false;
        });
        this.setState({
            filter_text,
            page_num: 0,
            ids,
            last_page: Math.ceil(ids.length / this.state.page_size) - 1,
        });
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
              T('button', {onClick: () => this.setState({ids: shuffle(this.state.ids)})}, 'Shuffle items'),
              T('label', {}, 'Filter by regexp:'),
              T('input', {type: 'text', size: 40, value: this.state.filter_text, onChange: this.filter}),
            ),
        );
        return [nav, ids];
    }

    render(){
        const {clickHandler} = this.props;
        const {hover} = this.state;
        const [nav, ids] = this.render_nav();
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
        // helper to randomly choose from array
        const choice = (arr) => arr[Math.floor(Math.random() * arr.length)];
        this.state = {collections: []};
    }

    // returns a list of ids based on our props
    ids(){
        return Object.keys(this.props.items);
    }

    // adds a new collection
    add_collection = () => {
        const {collections} = this.state;
        this.setState({collections: immer.produce(collections, draft => {
            draft.push({
                title: `Collection ${collections.length+1}`,
                desc: '',
                show_plus: true,
                show_minus: true,
            });
            return draft;
        })});
    }

    render(){
        const {items} = this.props;
        console.log('rendering main', Object.keys(items).length, this.state, this.props.cfg);
        const {collections} = this.state;
        return T('div', {className: 'main'},
                   collections.map(c => T(Collection, {items, ...c})),
                   T('button', {onClick: this.add_collection}, 'Add collection'),
                   T('h3', {}, "All items"),
                   T(AllItems, {items}));
    }

    // after first mount, add a collection
    componentDidMount(){
        this.add_collection();
    }

}


ReactDOM.render(T(Main, data), document.querySelector('#main'));
