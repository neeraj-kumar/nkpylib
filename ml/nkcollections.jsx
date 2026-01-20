/* NK Collections React App
 *
 * Each collection item contains:
 * - id: unique int identifier
 * - source: the source platform (e.g., twitter, tumblr, etc)
 * - stype: this is just 'blog' for now, can ignore
 * - otype: ("object type") one of: post (an entire post), text, link, image, video
 * - url: URL of the object (if applicable)
 * - name: name of the object (if applicable)
 * - parent: id of the parent object (if applicable). For example, if a post contains some text and
 *   2 images, then the text and images objects will have the post's id as their parent.
 * - ts: timestamp of the item on the original platform (seconds since epoch)
 * - added_ts: timestamp of when the item was added to this Collection
 * - explored_ts: timestamp of when the item was last explored/expanded (if ever)
 * - seen_ts: timestamp of when the item was last seen by me (if ever)
 * - embed_ts: timestamp of when the item was last embedded (if ever)
 * - md: json metadata object (see below)
 *
 *
 * For different types of sources we have different metadata:
 *
 * - Twitter: We have posts, text, and image objects for now. The url on a post is the tweet url,
 *   while on images it's to the image thumbnail. Metadata:
 *   - Posts: handle, display_name, likes, replies, reposts, views, iso_ts
 *   - Text: text (the tweet content)
 *   - Images: media_key, ext (file extension), name (optional size info for url, like "360x360" or
 *   "small")
 *
 * - Tumblr: We have posts and various content blocks (text, image, video, link). The url on a post
 *   is the tumblr post URL, while content blocks have fragment URLs or direct media URLs. Metadata:
 *   - Posts: post_id, reblog_key, tags, n_notes, n_likes, n_reblogs, summary, original_type, reblogged_from
 *   - Text: text (the text content)
 *   - Images: w (width), h (height), media_key
 *   - Videos: w, h, media_key, provider, poster_url, poster_media_key
 *   - Links: display_url, title, description
 */

const DEBOUNCE_MS = 2000;

// Detect if we're on a mobile device
const IS_MOBILE = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
                  (window.innerWidth <= 768);

// Utility function for making API calls
const fetchEndpoint = async (endpoint, data = {}, options = {}) => {
  const {
    method = 'POST',
    headers = { 'Content-Type': 'application/json' },
    onError = (error) => console.error('Fetch error:', error),
    ...fetchOptions
  } = options;

  try {
    const response = await fetch(endpoint, {
      method,
      headers,
      body: method !== 'GET' ? JSON.stringify(data) : undefined,
      ...fetchOptions
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    onError(error);
    throw error;
  }
};

// API helper functions
const api = {
  get: (params) => fetchEndpoint('/get', params),
  classify: (pos) => fetchEndpoint('/classify', { pos }),
  action: (id, action) => fetchEndpoint('/action', { id, action }),
  sourceUrl: (url) => fetchEndpoint('/source', { url }),
};

const STYLES = `

.labeled {
  border: 1px solid #888;
  padding: 5px;
  margin-bottom: 10px;
}

.objects {
  display: grid;
  gap: 5px;
}

.labeled, .controls {
  display: flex;
  flex-wrap: wrap;
}

.control {
  margin-right: 10px;
}

.text-fields input {
  margin-right: 10px;
}

.filter-input, .search-input {
  display: none;
}

.object {
  border: 1px solid #ccc;
  padding: 5px;
  margin: 5px;
  text-align: center;
}

.object.post {
  border-color: blue;
}

.object.text {
  border-color: green;
}

.object.image {
  border-color: orange;
}

.text .content {
  max-width: 200px;
}

.object {
  max-width: calc((100vw - (var(--n-cols) + 1) * 10px) / var(--n-cols) - 10px);
  box-sizing: border-box;
}

.object img {
  max-width: 100%;
  height: auto;
}

.score {
  font-size: 0.8em;
  color: #555;
}

.button-bar {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 5px;
  padding: 5px;
  border-top: 1px solid #eee;
}

.button-bar .icon-button {
  cursor: pointer;
  font-size: 18px;
  padding: 5px;
  border-radius: 3px;
  user-select: none;
}

.button-bar .icon-button:hover {
  background-color: #f0f0f0;
}

.heart-icon {
  color: #ccc;
}

.heart-icon.liked {
  color: red;
}

.classify-icon {
  color: #666;
}

.classify-icon.selected {
  background-color: #e6f3ff;
  color: #0066cc;
}

.open-icon {
  color: #666;
}

/* Twitter-specific styles */
.source-twitter.otype-post {
  border-left: 4px solid #1da1f2;
}

.twitter-handle {
  font-weight: bold;
  color: #1da1f2;
}

.twitter-display-name {
  color: #333;
  margin-bottom: 5px;
}

.twitter-stats {
  font-size: 0.8em;
  color: #666;
  margin-top: 5px;
}

/* Tumblr-specific styles */
.source-tumblr.otype-post {
  border-left: 4px solid #00cf35;
}

.tumblr-tags {
  font-size: 0.8em;
  color: #666;
  margin-bottom: 5px;
}

.tumblr-stats {
  font-size: 0.8em;
  color: #666;
  margin-top: 5px;
}

.tumblr-content {
  margin: 10px 0;
}

.tumblr-text-block {
  margin: 5px 0;
  text-align: left;
}

.tumblr-image-block {
  margin: 5px 0;
}

.tumblr-video-block {
  margin: 5px 0;
}

.tumblr-link-block {
  margin: 5px 0;
  padding: 5px;
  border: 1px solid #ddd;
  border-radius: 3px;
  background-color: #f9f9f9;
}

.tumblr-link-description {
  font-size: 0.8em;
  color: #666;
  margin-top: 3px;
}

.tumblr-unknown-block {
  margin: 5px 0;
  padding: 5px;
  background-color: #ffe6e6;
  border: 1px solid #ffcccc;
  border-radius: 3px;
  font-size: 0.8em;
}
`;

// Source-specific content renderers for posts only
const TwitterPostContent = ({id, otype, url, md, score}) => {
  return (
    <div>
      <div className="twitter-handle">@{md.handle}</div>
      <div className="twitter-display-name">{md.display_name}</div>
      <div className="twitter-stats">
        {md.likes} ♥ {md.reposts} ↻ {md.replies} 💬 {md.views} 👁
      </div>
      <p className="score">ID: {id}</p>
      {score !== undefined && (
        <div className="score">Score: {score.toFixed(3)}</div>
      )}
    </div>
  );
};

const TumblrContentBlock = ({block}) => {
  const {type, data} = block;
  
  switch (type) {
    case 'text':
      return <div className="tumblr-text-block">{data.md.text}</div>;
    case 'image':
      return (
        <div className="tumblr-image-block">
          <img src={data.url} alt={`Image ${data.id}`} />
        </div>
      );
    case 'video':
      return (
        <div className="tumblr-video-block">
          <a href={data.url} target="_blank" rel="noreferrer">
            <img src={data.md.poster_url} alt={`Video ${data.id} poster`} />
          </a>
        </div>
      );
    case 'link':
      return (
        <div className="tumblr-link-block">
          <a href={data.url} target="_blank" rel="noreferrer">
            {data.md.title || data.md.display_url}
          </a>
          {data.md.description && (
            <div className="tumblr-link-description">{data.md.description}</div>
          )}
        </div>
      );
    default:
      return <div className="tumblr-unknown-block">Unknown block type: {type}</div>;
  }
};

const TumblrPostContent = ({id, otype, url, md, score, content_blocks}) => {
  return (
    <div>
      <div className="tumblr-tags">#{md.tags.slice(0, 3).join(' #')}</div>
      <div className="tumblr-stats">
        {md.n_notes} notes • {md.n_likes} ♥ • {md.n_reblogs} ↻
      </div>
      <div className="tumblr-content">
        {content_blocks.map((block, index) => (
          <TumblrContentBlock key={`${id}-${index}`} block={block} />
        ))}
      </div>
      <p className="score">ID: {id}</p>
      {score !== undefined && (
        <div className="score">Score: {score.toFixed(3)}</div>
      )}
    </div>
  );
};

const Obj = (props) => {
  const {id, otype, url, md, togglePos, score, rels, setLiked, source, pos} = props;
  //console.log('Obj', id, otype, score, props);
  const liked = Boolean(rels.like);
  const rendererName = `${source.charAt(0).toUpperCase() + source.slice(1)}PostContent`;
  const PostContentRenderer = window[rendererName]

  return (
    <div id={`id-${id}`} className={`object ${otype} source-${source} otype-${otype}`}>
      <div className="button-bar">
        <div
          className={`icon-button heart-icon ${liked ? 'liked' : ''}`}
          onClick={(e) => {
            e.stopPropagation();
            setLiked(id, !liked);
          }}
        >
          ♥
        </div>
        <div
          className={`icon-button classify-icon ${pos.includes(id) ? 'selected' : ''}`}
          onClick={(e) => {
            e.stopPropagation();
            togglePos(id);
          }}
        >
          🎯
        </div>
        <div
          className="icon-button open-icon"
          onClick={(e) => {
            e.stopPropagation();
            window.open(url, '_blank');
          }}
        >
          🔗
        </div>
      </div>
      {otype === 'post' && PostContentRenderer ? (
        <PostContentRenderer {...props} />
      ) : (
        <div>
          {otype === 'text' && (
            <div className="content">{md.text}</div>
          )}
          {otype === 'link' && (
            <div className="content"><a href={url} target="_blank" rel="noreferrer">{md.title || md.display_url}</a></div>
          )}
          {otype === 'image' && (
            <div className="content">
              <img src={url} alt={`Image ${id}`} />
            </div>
          )}
          {otype === 'video' && (
            <div className="content">
              <a href={url} target="_blank" rel="noreferrer">
                <img src={md.poster_url} alt={`Video ${id} poster`} />
              </a>
            </div>
          )}
          <p className="score">ID: {id}</p>
          {score !== undefined && (
            <div className="score">Score: {score.toFixed(3)}</div>
          )}
        </div>
      )}
    </div>
  );
}

const Controls = ({allOtypes, curOtypes, setCurOtypes, setCurIds,
  sourceStr, setSourceStr, doSource, filterStr, updateFilterStr, searchStr, updateSearchStr,
  nCols, setNCols, ...props}) => {
  // add a "return" key handler for the source input
  const keyHandler = (e) => {
    if (e.key === 'Enter') {
      doSource();
    }
  }
  return (
    <div className="controls">
      <div className="control text-fields">
        <input
          type="text"
          className="src-input"
          placeholder="Source..."
          value={sourceStr}
          onChange={(e) => setSourceStr(e.target.value)}
          onKeyDown={keyHandler}
          size="80"
        />
        <button onClick={() => doSource()}>Set Source</button>
        <input
          type="text"
          className="filter-input"
          placeholder="Filter..."
          value={filterStr}
          onChange={(e) => updateFilterStr(e.target.value)}
        />
        <input
          type="text"
          className="search-input"
          placeholder="Search..."
          value={searchStr}
          onChange={(e) => updateSearchStr(e.target.value)}
        />
        <label>Cols:</label>
        <input
          type="number"
          placeholder="Cols"
          value={nCols}
          onChange={(e) => setNCols(parseInt(e.target.value) || 1)}
          min="1"
          max="20"
          style={{width: '60px', marginLeft: '10px'}}
        />
      </div>
      <div className="control otype-filters">
      {allOtypes.map((otype) => (
        <label key={otype} style={{marginRight: '10px'}}>
          <input
            type="checkbox"
            checked={curOtypes.includes(otype)}
            onChange={(e) => {
              setCurOtypes((curOtypes) => {
                if (e.target.checked) {
                  return [...curOtypes, otype];
                } else {
                  return curOtypes.filter((x) => x !== otype);
                }
              });
            }}
          />
          {otype}
        </label>
      ))}
      </div>
      <div className="control randomize-btn">
        <button onClick={() => {
          // shuffle curIds
          setCurIds((curIds) => {
            const shuffled = [...curIds];
            for (let i = shuffled.length - 1; i > 0; i--) {
              const j = Math.floor(Math.random() * (i + 1));
              [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }
            return shuffled;
          });
        }}>Randomize</button>
      </div>
    </div>
  );
}


const App = () => {
  const [rowById, setRowById] = React.useState({});
  const [allOtypes, setAllOtypes] = React.useState([]);
  const [curOtypes, setCurOtypes] = React.useState(['post', 'image', 'text', 'link']);
  const [curIds, setCurIds] = React.useState([]);
  const [scores, setScores] = React.useState({});
  const [pos, setPos] = React.useState([]);
  const [filterStr, setFilterStr] = React.useState('');
  const [searchStr, setSearchStr] = React.useState('');
  const [sourceStr, setSourceStr] = React.useState('');
  const [nCols, setNCols] = React.useState(IS_MOBILE ? 2 : 6);

  // Refs to access current values in debounced callbacks
  const filterStrRef = React.useRef(filterStr);
  const searchStrRef = React.useRef(searchStr);

  // Debounce timers
  const searchTimeoutRef = React.useRef(null);
  const filterTimeoutRef = React.useRef(null);

  // Update refs when state changes
  React.useEffect(() => {
    filterStrRef.current = filterStr;
  }, [filterStr]);

  React.useEffect(() => {
    searchStrRef.current = searchStr;
  }, [searchStr]);

  // Generic debounced function factory
  const createDebouncedUpdater = React.useCallback((setter, timeoutRef, onTrigger, delay = DEBOUNCE_MS) => {
    return (value) => {
      setter(value);
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(() => onTrigger(value), delay);
    };
  }, []);

  const doSearch = React.useCallback((value) => {
    console.log('searching for', value, filterStrRef.current, searchStrRef.current);
    //TODO implement
  }, []);

  const doFilter = React.useCallback((value) => {
    console.log('filtering for', value, filterStrRef.current, searchStrRef.current);
    //TODO implement
  }, []);

  const updateSearchStr = createDebouncedUpdater(setSearchStr, searchTimeoutRef, doSearch);
  const updateFilterStr = createDebouncedUpdater(setFilterStr, filterTimeoutRef, doFilter);
  React.useEffect(() => {
    document.title = 'NK Collections';
    // insert styles
    const styleEl = document.createElement('style');
    styleEl.innerHTML = STYLES;
    document.head.appendChild(styleEl);
  }, []);

  const updateData = React.useCallback((data, resetData=false) => {
    console.log('got data', data);
    if (resetData) {
      setRowById({});
    }
    // use immer to update rowById
    setRowById((rowById) => immer.produce(rowById, (draft) => {
      Object.entries(data.rows).forEach(([id, row]) => {
        row.rels = row.rels || {};
        draft[id] = row;
      });
    }));
    setCurIds(Object.keys(data.rows));
    setAllOtypes(data.allOtypes);
  }, [setRowById, setCurIds, setAllOtypes]);

  // fetch data when otypes changes
  React.useEffect(() => {
    // fetch objects from the server
    api.get({
      otype: curOtypes,
      seen_ts: '>=' + (Math.floor(Date.now() / 1000) - 3600), // seen within the last hour
    }).then(updateData);
  }, [updateData]);

  // toggles the given id in the pos array
  const togglePos = React.useCallback((id) => {
    setPos((pos) => {
      if (pos.includes(id)) {
        return pos.filter((x) => x !== id);
      } else {
        return [...pos, id];
      }
    });
  });

  const setLiked = React.useCallback((id, likedState) => {
    console.log('setting liked for', id, likedState);
    // send to server
    api.action(id, likedState ? 'like' : 'unlike');
    // update rowById
    setRowById((rowById) => {
      return immer.produce(rowById, (draft) => {
        if (!draft[id]) return;
        if (likedState) {
          // set like to current ts (seconds since epoch)
          draft[id].rels.like = Math.floor(Date.now() / 1000);
        } else {
          // delete like from rels (if it exists)
          delete draft[id].rels.like;
        }
      });
    });
  }, [setRowById]);

  // function to call classification, whenever pos changes
  React.useEffect(() => {
    if (pos.length === 0) {
      // reset curIds to all ids and scores to empty
      setCurIds(Object.keys(rowById));
      setScores({});
      return;
    }
    console.log('calling classify for pos', pos);
    api.classify(pos).then((data) => {
      console.log('got classify data', data);
      // update curIds and scores
      if (data.curIds && data.scores){
        setCurIds(data.curIds);
        setScores(data.scores);
      }
    });
  }, [pos]);

  // the source string can be either a url or a JSON string of parameters
  const doSource = React.useCallback(() => {
    console.log('updating source with', sourceStr);
    const isUrl = sourceStr.startsWith('http');
    if (isUrl) { // if we got a URL, extract the params and do another fetch to /get
      api.sourceUrl(sourceStr).then((params) => {
        return api.get(params);
      }).then((data) => {
        updateData(data, true);
      });
    } else { // Parse as JSON and use as get parameters
      try {
        const params = JSON.parse(sourceStr);
        api.get(params).then((data) => {
          updateData(data, true);
        });
      } catch (error) {
        console.error('Invalid JSON in source string:', error);
      }
    }
  }, [sourceStr, updateData]);

  const funcs = {allOtypes, curOtypes, togglePos, setCurOtypes, setCurIds,
    sourceStr, setSourceStr, doSource, filterStr, updateFilterStr, searchStr, updateSearchStr,
    setLiked, nCols, setNCols, pos};
  console.log('rowById', rowById, curIds, pos, scores);
  const ids = curIds.filter(id => rowById[id] && curOtypes.includes(rowById[id].otype));

  return (
  <div>
    <h3>Collections</h3>
    <h4>Labeled</h4>
    <div className="labeled">
      {pos.map((id) => <Obj key={id} {...funcs} {...rowById[id]} />)}
    </div>
    <Controls {...funcs} />
    <div
      className="objects"
      style={{
        gridTemplateColumns: `repeat(${nCols}, 1fr)`,
        '--n-cols': nCols
      }}
    >
      {ids.map((id) => <Obj key={id} score={scores[id]} {...funcs} {...rowById[id]} />)}
    </div>
  </div>
  );
}

ReactDOM.createRoot(document.getElementById("main")).render(<App />);
