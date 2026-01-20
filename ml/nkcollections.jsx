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

.randomize-btn {
  display: none;
}

.objects {
  /* Masonry library will handle layout */
}

.object {
  display: block;
  break-inside: avoid;
  border: 1px solid #ccc;
  margin: 0px;
  text-align: center;
  max-width: 100%;
  box-sizing: border-box;
}

.flexobjects {
  display: flex;
  flex-wrap: wrap;
}

.flexobject {
  border: 1px solid #ccc;
  margin: 0px;
  text-align: center;
  flex: 0 0 calc((100% - (var(--n-cols) - 1) * 10px) / var(--n-cols));
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

.object.post {
}

.object.text {
}

.object.image {
}

.text .content {
  max-width: 200px;
}

.object {
  max-width: calc((100vw - (var(--n-cols) + 1) * 10px) / var(--n-cols) - 1px);
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
}

.button-bar .icon-button {
  cursor: pointer;
  font-size: 12px;
  padding: 2px;
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
}

.tumblr-tags {
  display: none;
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

/* Media carousel styles */
.media-carousel {
  margin: 0 0;
}

.media-nav {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  margin-top: 5px;
  padding: 5px;
}

.media-nav button {
  background: #f0f0f0;
  border: 1px solid #ccc;
  border-radius: 3px;
  padding: 5px 10px;
  cursor: pointer;
  font-size: 16px;
}

.media-nav button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.media-nav button:hover:not(:disabled) {
  background: #e0e0e0;
}

.media-nav span {
  font-size: 0.9em;
  color: #666;
  min-width: 40px;
  text-align: center;
}

/* Media navigation in button bar */
.media-nav-button {
  color: #666;
}

.media-counter {
  font-size: 0.8em;
  color: #666;
  padding: 0 5px;
  user-select: none;
  margin-top: 3px;
}
`;

// Source-specific content renderers for posts only
const TwitterPostContent = ({id, otype, url, md, score, simpleMode}) => {
  return (
    <div>
      <div className="twitter-handle">@{md.handle}</div>
      <div className="twitter-display-name">{md.display_name}</div>
      <div className="twitter-stats">
        {md.likes} ♥ {md.reposts} ↻ {md.replies} 💬 {md.views} 👁
      </div>
      {!simpleMode && <p className="score">ID: {id}</p>}
      {!simpleMode && score !== undefined && (
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
    case 'image':
    case 'video':
      // Media is now handled by MediaCarousel, so skip rendering here
      return null;
    default:
      return <div className="tumblr-unknown-block">Unknown block type: {type}</div>;
  }
};

const TumblrPostContent = (props) => {
  const {id, otype, url, md, score, simpleMode} = props;
  const content_blocks = props.content_blocks || [];
  // Filter out media blocks since they're handled by MediaCarousel
  const nonMediaBlocks = content_blocks.filter(block =>
    block.type !== 'image' && block.type !== 'video'
  ) || [];
  return (
    <div>
      <div className="tumblr-tags">#{md.tags.slice(0, 3).join(' #')}</div>
      {!simpleMode && (
        <div className="tumblr-stats">
          {md.n_notes} 📝 • {md.n_likes} ♥ • {md.n_reblogs} ↻
        </div>
      )}
      <div className="tumblr-content">
        {nonMediaBlocks.map((block, index) => (
          <TumblrContentBlock key={`${id}-${index}`} block={block} />
        ))}
      </div>
      {!simpleMode && <p className="score">ID: {id}</p>}
      {!simpleMode && score !== undefined && (
        <div className="score">Score: {score.toFixed(3)}</div>
      )}
    </div>
  );
};

const MediaCarousel = ({mediaBlocks, currentIndex, setCurrentIndex}) => {
  if (!mediaBlocks.length) return null;
  const currentMedia = mediaBlocks[currentIndex];
  const handleImageClick = (e) => {
    if (mediaBlocks.length <= 1) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const imageWidth = rect.width;
    const clickThreshold = imageWidth * 0.45; // pixels from edge to trigger navigation
    if (clickX <= clickThreshold) {
      // Clicked on left edge - go to previous
      e.preventDefault();
      e.stopPropagation();
      setCurrentIndex(currentIndex === 0 ? mediaBlocks.length - 1 : currentIndex - 1);
    } else if (clickX >= imageWidth - clickThreshold) {
      // Clicked on right edge - go to next
      e.preventDefault();
      e.stopPropagation();
      setCurrentIndex(currentIndex === mediaBlocks.length - 1 ? 0 : currentIndex + 1);
    }
  };
  const renderMedia = (block) => {
    const {type, data} = block;
    switch (type) {
      case 'image':
        return (
          <img
            src={data.url}
            alt={`Image ${data.id}`}
            onClick={handleImageClick}
            style={{cursor: mediaBlocks.length > 1 ? 'pointer' : 'default'}}
          />
        );
      case 'video':
        return (
          <a href={data.url} target="_blank" rel="noreferrer">
            <img
              src={data.md.poster_url}
              alt={`Video ${data.id} poster`}
              onClick={handleImageClick}
              style={{cursor: mediaBlocks.length > 1 ? 'pointer' : 'default'}}
            />
          </a>
        );
      default:
        return null;
    }
  };
  return (
    <div className="media-carousel">
      {renderMedia(currentMedia)}
    </div>
  );
};

const Obj = (props) => {
  const {id, otype, url, md, togglePos, score, rels, setLiked, source, pos, media_blocks} = props;
  //console.log('Obj', id, otype, score, props);
  const liked = Boolean(rels.like);
  const rendererName = `${source.charAt(0).toUpperCase() + source.slice(1)}PostContent`;
  const PostContentRenderer = window[rendererName]

  // Media carousel state
  const [currentMediaIndex, setCurrentMediaIndex] = React.useState(0);
  const hasMultipleMedia = media_blocks && media_blocks.length > 1;

  const mediaDivs = [
    (<div key="a"
      className="icon-button media-nav-button"
      onClick={(e) => {
        e.stopPropagation();
        setCurrentMediaIndex(currentMediaIndex === 0 ? media_blocks.length - 1 : currentMediaIndex - 1);
      }}
    >
      ←
    </div>),
    (<span className="media-counter" key="b">
      {currentMediaIndex + 1}/{media_blocks.length}
    </span>),
    (<div key="c"
      className="icon-button media-nav-button"
      onClick={(e) => {
        e.stopPropagation();
        setCurrentMediaIndex(currentMediaIndex === media_blocks.length - 1 ? 0 : currentMediaIndex + 1);
      }}
    >
      →
    </div>),
  ];

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
        {/* Media navigation controls - only show if multiple media */}
        {hasMultipleMedia && mediaDivs}
      </div>
      
      {/* Media carousel for posts with media */}
      {otype === 'post' && media_blocks && media_blocks.length > 0 && (
        <MediaCarousel 
          mediaBlocks={media_blocks} 
          currentIndex={currentMediaIndex}
          setCurrentIndex={setCurrentMediaIndex}
        />
      )}
      
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
          {!props.simpleMode && <p className="score">ID: {id}</p>}
          {!props.simpleMode && score !== undefined && (
            <div className="score">Score: {score.toFixed(3)}</div>
          )}
        </div>
      )}
    </div>
  );
}

const Controls = ({allOtypes, curOtypes, setCurOtypes, setCurIds,
  sourceStr, setSourceStr, doSource, filterStr, updateFilterStr, searchStr, updateSearchStr,
  nCols, setNCols, simpleMode, setSimpleMode, ...props}) => {
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
      <div className="control simple-mode">
        <label>
          <input
            type="checkbox"
            checked={simpleMode}
            onChange={(e) => setSimpleMode(e.target.checked)}
          />
          Simple
        </label>
      </div>
    </div>
  );
}


const App = () => {
  const [rowById, setRowById] = React.useState({});
  const [allOtypes, setAllOtypes] = React.useState([]);
  //const [curOtypes, setCurOtypes] = React.useState(['post', 'image', 'text', 'link']);
  const [curOtypes, setCurOtypes] = React.useState(['post']);
  const [curIds, setCurIds] = React.useState([]);
  const [scores, setScores] = React.useState({});
  const [pos, setPos] = React.useState([]);
  const [filterStr, setFilterStr] = React.useState('');
  const [searchStr, setSearchStr] = React.useState('');
  const [sourceStr, setSourceStr] = React.useState('');
  const [nCols, setNCols] = React.useState(IS_MOBILE ? 3 : 8);
  const [simpleMode, setSimpleMode] = React.useState(true);

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

  React.useEffect(() => {
    const grid = document.querySelector('.objects');
    if (grid && window.Masonry) {
      const containerWidth = grid.offsetWidth;
      const columnWidth = (containerWidth - (nCols - 1) * 10) / nCols;
      
      // Wait for images to load
      const images = grid.querySelectorAll('img');
      let loadedImages = 0;
      
      const initMasonry = () => {
        new window.Masonry(grid, {
          itemSelector: '.object',
          columnWidth: columnWidth,
          gutter: 10
        });
      };
      
      if (images.length === 0) {
        initMasonry();
      } else {
        images.forEach(img => {
          if (img.complete) {
            loadedImages++;
          } else {
            img.onload = () => {
              loadedImages++;
              if (loadedImages === images.length) {
                initMasonry();
              }
            };
          }
        });
        
        if (loadedImages === images.length) {
          initMasonry();
        }
      }
    }
  }, [ids, nCols]); // Re-run when items or columns change

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
      added_ts: '>=' + (Math.floor(Date.now() / 1000) - (24*3600)), // added within the last day
      assemble_posts: true,
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
    setLiked, nCols, setNCols, pos, simpleMode, setSimpleMode};
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
