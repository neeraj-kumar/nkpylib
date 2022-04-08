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



// generic image renderer
function image_renderer(props){
    let {id, path, hover=null, hoverHandler=null, clickHandler=null, example_type='', source=''} = props;
    path = path.replace('/images/', '/thumbs/');
    const onDragStart = (ev) => {
        const data = JSON.stringify(Object.assign({}, ev.currentTarget.dataset));
        ev.dataTransfer.setData(DRAG_FMT, data);
        console.log('drag start', ev.currentTarget, data, ev.dataTransfer.getData(DRAG_FMT));
    }
    const opts = {className:'image-div', 'data-id': id, 'data-source': source, key: id, draggable: true, onDragStart};
    let inner = [];
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
    return T('div', opts, T('img', {src:`static/${path}`}), inner);
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

    // returns image objects from ids
    id2image = (id) => {
        return {id, path: this.props.images[id]};
    }

    // classifies our images based on pos & neg in `obj`
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
                    lst.map(this.id2image).map(im => image_renderer({
                        clickHandler,
                        example_type: key,
                        source: `${this.state.uuid}/${key}`,
                        ...im
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

class AllImages extends React.Component {
    constructor(props) {
        super(props);
        const {pageSize=100} = this.props;
        this.state = {start: 0, end: pageSize, hover: null};
    }

    hoverHandler = (id) => {
        this.setState({hover: id});
    }

    incr = (amount) => {
        let {start, end} = this.state;
        start += amount;
        end += amount;
        if (start < 0){
            start = 0;
        }
        if (end >= this.props.images.length){
            end = this.props.images.length - 1;
        }
        this.setState({start, end});
    }

    // returns image objects from ids
    id2image = (id) => {
        return {id, path: this.props.images[id]};
    }

    // returns a list of ids based on our props
    // Note that we sort this pseudo-randomly
    ids(){
        // props.images is a map from id to path
        const {images} = this.props;
        let ids = Object.keys(images);
        ids = shuffle(ids, 4);
        return ids;
    }

    render(){
        //console.log('rendering all-images', Object.keys(this.props.images).length, this.state);
        const {images, clickHandler, pageSize=50} = this.props;
        const {start, end, hover} = this.state;
        let ids = this.ids().slice(start, end);
        return T('div', {className: 'all-div'},
                 ids.map(this.id2image).map(im => image_renderer({
                        hover,
                        clickHandler,
                        hoverHandler: this.hoverHandler,
                        ...im,
                    })),
                 T('button', {onClick: () => this.incr(-pageSize)}, 'Prev'),
                 T('button', {onClick: () => this.incr(pageSize)}, 'Next'),
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
        return Object.keys(this.props.images);
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
        console.log('rendering main', Object.keys(this.props.images).length, this.state);
        const {images} = this.props;
        const {collections} = this.state;
        return T('div', {className: 'main'},
                   collections.map(c => T(Collection, {images, ...c})),
                   T('button', {onClick: this.add_collection}, 'Add collection'),
                   T('h3', {}, "All items"),
                   T(AllImages, {images}));
    }

    // after first mount, add a collection
    componentDidMount(){
        this.add_collection();
    }

}


ReactDOM.render(T(Main, data), document.querySelector('#main'));
