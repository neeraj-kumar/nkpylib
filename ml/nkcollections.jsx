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
const MODES = ['multicol', 'cluster'];

// Detect if we're on a mobile device
const IS_MOBILE = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || (window.innerWidth <= 768);

// Create context for message handling
const MessageContext = React.createContext();

// Utility function for making API calls
const fetchEndpoint = async (endpoint, data = {}, options = {}) => {
  const {
    method = 'POST',
    headers = { 'Content-Type': 'application/json' },
    onError = (error) => console.error('Fetch error:', error),
    ...fetchOptions
  } = options;

  // Get setMessage from context
  const setMessage = React.useContext(MessageContext);

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
    if (setMessage) {
      setMessage(`API call failed: ${endpoint} - ${error.message}`);
    }
    onError(error);
    throw error;
  }
};

// API helper functions
const api = {
  get: (params) => fetchEndpoint('/get', params),
  classify: (pos) => fetchEndpoint('/classify', { pos }),
  classifyLikes: (type, otype) => fetchEndpoint('/classify', { type, otype }),
  action: (id, action) => fetchEndpoint('/action', { id, action }),
  sourceUrl: (url) => fetchEndpoint('/source', { url }),
  cluster: (clusters, ids) => fetchEndpoint('/cluster', { clusters, ids }),
};

const STYLES = `

.hidden {
  display: none;
}

.labeled {
  border: 1px solid #888;
  padding: 5px;
  margin-bottom: 10px;
}

.randomize-btn {
  display: none;
}

.flex-break {
  flex-basis: 100%;
  height: 0;
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
  width: 100%;
  box-sizing: border-box;
}

.object.single-col {
  max-width: 400px!important;
}

.gridobjects {
  display: grid;
  grid-auto-flow: dense;
  align-items: start;
  column-fill: balance;
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
  font-size:20px!important;
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

/* Twitter-specific styles */
.source-twitter {
  font-family: "ui-sans-serif", "system-ui", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

.object.source-twitter {
  padding: 0 5px;
  margin-top: 5px;
  /* causes issues: max-width: 550px; */
}


.twitter-header {
  display: flex;
  text-align: left;
}

.twitter-display-name {
  color: #333;
  margin-bottom: 5px;
  font-weight: 700;
  margin-right: 10px;
}

.twitter-handle {
  color: rgb(83, 100, 113);
}

.twitter-ts {
  display: none;
  font-size: 0.6em;
}

.twitter-stats {
  font-size: 0.8em;
  color: #666;
  margin-top: 5px;
  display: flex;
  justify-content: space-evenly;
}

.twitter-text-block {
  margin: 5px 0;
  text-align: left;
  font-size: 15px;
}

.twitter-content {
  margin: 10px 0;
}

.twitter-unknown-block {
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

/* Video link overlay icon */
.video-link {
  position: relative;
  display: inline-block;
}

.video-link::after {
  content: '▶';
  position: absolute;
  top: 5px;
  right: 5px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  font-size: 12px;
  padding: 2px 4px;
  border-radius: 3px;
  pointer-events: none;
}

/* Source input validation colors */
.src-input.url-input {
  border: 2px solid #007bff;
}

.src-input.valid-json {
  border: 2px solid #28a745;
}

.src-input.invalid-json {
  border: 2px solid #dc3545;
}

/* Cluster buttons */
.cluster-buttons {
  display: flex;
  gap: 2px;
  margin-left: 5px;
}

.cluster-button {
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 3px;
  padding: 2px;
  cursor: pointer;
  font-size: 11px;
  font-weight: bold;
  color: #495057;
  text-align: center;
  user-select: none;
  min-width: 20px;
  max-height: 16px;
}

.cluster-button:hover {
  background: #e9ecef;
  border-color: #adb5bd;
}

.cluster-button.active {
  background: #007bff;
  border-color: #0056b3;
  color: white;
}

.cluster-button.active:hover {
  background: #0056b3;
}

.cluster-button.manual {
  border: 2px solid #28a745;
}

.cluster-button.automatic {
  border: 1px dashed #dee2e6;
}

/* Keyboard active object highlighting */
.object.keyboard-active {
  box-shadow: 0 0 5px #007bff;
  border-color: #007bff;
}

/* Cluster assignment visual indicators */
.object.manual-cluster {
  border: 2px solid #000;
}

.object.automatic-cluster {
  border: 2px dotted #bbb;
  border-opacity: calc(0.4 + 0.6 * var(--cluster-score, 0.5));
}

.object.unlabeled-cluster {
  border: 2px solid #bbb;
  border-opacity: 0.6;
}

/* Cluster columns layout */
.cluster-columns {
  display: flex;
  gap: 10px;
  width: 100%;
}

.cluster-column {
  flex: 1;
  min-height: 200px;
  /*border: 2px dashed #ddd; */
  border-radius: 5px;
  padding: 10px;
  background-color: #fafafa;
}

.cluster-column h4 {
  margin: 0 0 10px 0;
  text-align: center;
  color: #666;
  font-size: 14px;
  font-weight: bold;
}

.cluster-column.has-items {
  border-color: #007bff;
  background-color: #f8f9ff;
}

.cluster-column .object {
  margin-bottom: 10px;
  max-width: 100%;
}
`;

// Source-specific content renderers for posts only
const TwitterContentBlock = ({block}) => {
  const {type, data} = block;
  switch (type) {
    case 'text':
      return <div className="twitter-text-block">{data.md.text}</div>;
    case 'image':
    case 'video':
      // Media is now handled by MediaCarousel, so skip rendering here
      return null;
    default:
      return <div className="twitter-unknown-block">Unknown block type: {type}</div>;
  }
};

const TwitterPostContent = (props) => {
  const {id, otype, url, ts, md, score, simpleMode, content_blocks} = props;
  const blocks = content_blocks || [];
  // Filter out media blocks since they're handled by MediaCarousel
  const nonMediaBlocks = blocks.filter(block =>
    block.type !== 'image' && block.type !== 'video'
  ) || [];
  const tsString = new Date(ts*1000).toLocaleString();
  return (
    <div>
      <div className="twitter-header" title={tsString}>
        <div className="twitter-display-name">{md.display_name}</div>
        <div className="twitter-handle">@{md.handle}</div>
        <div className="twitter-ts"> {tsString}</div>
      </div>
      <div className="twitter-content">
        {nonMediaBlocks.map((block, index) => (
          <TwitterContentBlock key={`${id}-${index}`} block={block} />
        ))}
      </div>
      <div className="twitter-stats">
        <span>💬{md.replies}</span>
        <span>↻{md.reposts}</span>
        <span>♥{md.likes}</span>
        <span>📊{md.views}</span>
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
  // Calculate max height based on image metadata
  const maxHeight = React.useMemo(() => {
    if (!mediaBlocks.length) return 0;
    // Get approximate container width (will be refined by actual container width)
    const containerWidth = window.innerWidth / 8; // Rough estimate based on default columns
    return Math.max(...mediaBlocks.map(block => {
      const {w, h} = block.data.md || {};
      if (w && h) {
        // Calculate scaled height maintaining aspect ratio
        return (h / w) * containerWidth;
      }
      return 200; // fallback height
    }));
  }, [mediaBlocks]);

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
          <a href={data.url} target="_blank" rel="noreferrer" className="video-link">
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
    <div className="media-carousel"
      style={{minHeight: maxHeight > 0 ? `${maxHeight}px` : 'auto'}}
    >
      {renderMedia(currentMedia)}
    </div>
  );
};

const Obj = (props) => {
  let {id, otype, url, md, togglePos, score, rels, setLiked, source, pos, media_blocks, mode, clusters, setCluster} = props;
  media_blocks = media_blocks || [];
  //console.log('Obj', id, otype, score, props);
  const liked = Boolean(rels.like);
  const rendererName = `${source.charAt(0).toUpperCase() + source.slice(1)}PostContent`;
  const PostContentRenderer = window[rendererName]


  // Media carousel state
  const [currentMediaIndex, setCurrentMediaIndex] = React.useState(0);
  const hasMultipleMedia = media_blocks && media_blocks.length > 1;
  // Hover state for keyboard shortcuts
  const [isHovered, setIsHovered] = React.useState(false);

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

  let classes = ['object', otype, `source-${source}`, `otype-${otype}`];
  let cClasses = ['icon-button', 'classify-icon', (pos.includes(id) ? 'selected' : '')];
  if (mode !== 'multicol') {
    classes.push('single-col');
  } else {
    cClasses.push('hidden');
  }
  if (isHovered && mode === 'cluster') {
    classes.push('keyboard-active');
  }

  // Current cluster for this object
  const currentCluster = (clusters[id]) ? clusters[id].num : null;
  const clusterScore = (clusters[id]) ? clusters[id].score : 0;
  const isManualCluster = (clusters[id]) ? clusters[id].score === 1000 : false;
  const isUnlabeled = (clusters[id]) ? clusters[id].score === 0 : false;
  const normalizedScore = isManualCluster ? 1.0 : clusterScore; // Manual clusters get full opacity
  // Keyboard event handler for cluster assignment
  React.useEffect(() => {
    if (!isHovered || mode !== 'cluster') return;
    const handleKeyPress = (e) => {
      const num = parseInt(e.key);
      if (num >= 0 && num <= 5) {
        e.preventDefault();
        setCluster(id, currentCluster === num ? null : num);
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isHovered, mode, id, currentCluster, setCluster]);

  // Add cluster assignment visual indicators
  if (mode === 'cluster') {
    if (clusters[id]) {
      if (isManualCluster) {
        classes.push('manual-cluster');
      } else if (isUnlabeled) {
        classes.push('unlabeled-cluster');
      } else {
        classes.push('automatic-cluster');
      }
    }
  }

  return (
    <div 
      id={`id-${id}`} 
      className={classes.join(' ')}
      style={mode === 'cluster' && clusters[id] ? {
        '--cluster-score': normalizedScore
      } : {}}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
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
          className={cClasses.join(' ')}
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
        {/* Cluster buttons - only show in cluster mode */}
        {mode === 'cluster' && (
          <div className="cluster-buttons" title={`Cluster ${currentCluster}: ${clusterScore}`}>
            {[1, 2, 3, 4, 5].map(clusterNum => (
              <div
                key={clusterNum}
                className={`cluster-button ${currentCluster === clusterNum ? 'active' : ''} ${isManualCluster && currentCluster === clusterNum ? 'manual' : 'automatic'}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setCluster(id, currentCluster === clusterNum ? null : clusterNum);
                }}
              >
                {clusterNum}
              </div>
            ))}
          </div>
        )}
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
              <a href={url} target="_blank" rel="noreferrer" className="video-link">
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
      {/* Media carousel for posts with media */}
      {otype === 'post' && media_blocks && media_blocks.length > 0 && (
        <MediaCarousel
          mediaBlocks={media_blocks}
          currentIndex={currentMediaIndex}
          setCurrentIndex={setCurrentMediaIndex}
        />
      )}

    </div>
  );
}

const Controls = ({allOtypes, curOtypes, setCurOtypes, setCurIds,
  sourceStr, setSourceStr, doSource, filterStr, updateFilterStr, searchStr, updateSearchStr,
  nCols, setNCols, simpleMode, setSimpleMode, mode, setMode, refreshMasonry, doLikeClassifier, message, ...props}) => {
  // add a "return" key handler for the source input
  const keyHandler = (e) => {
    if (e.key === 'Enter') {
      doSource();
    }
  }

  // Determine the CSS class for the source input based on its contents
  const getSourceInputClass = () => {
    if (!sourceStr) return 'src-input';
    if (sourceStr.startsWith('http')) {
      return 'src-input url-input';
    }
    try {
      JSON.parse(sourceStr);
      return 'src-input valid-json';
    } catch (error) {
      return 'src-input invalid-json';
    }
  };
  return (
    <div className="controls">
      <div className="control text-fields">
        <input
          type="text"
          className={getSourceInputClass()}
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
      <div className="control refresh-masonry">
        <button onClick={refreshMasonry}>Refresh Layout</button>
      </div>
      <div className="control like-classifier">
        <button onClick={doLikeClassifier}>Like Classifier</button>
      </div>
      <div className="control mode-select">
        <label>Mode:</label>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          style={{marginLeft: '5px'}}
        >
          {MODES.map((modeOption) => (
            <option key={modeOption} value={modeOption}>
              {modeOption}
            </option>
          ))}
        </select>
      </div>
      <div className="control flex-break"></div>
      <div className="control message-display">
        <span>{message}</span>
      </div>
    </div>
  );
}


const App = () => {
  const [rowById, setRowById] = React.useState({});
  const [allOtypes, setAllOtypes] = React.useState([]);
  //const [curOtypes, setCurOtypes] = React.useState(['post', 'image', 'text', 'link']);
  const [curOtypes, setCurOtypes] = React.useState(['image', 'post']);
  const [curIds, setCurIds] = React.useState([]);
  const [scores, setScores] = React.useState({});
  const [pos, setPos] = React.useState([]);
  const [filterStr, setFilterStr] = React.useState('');
  const [searchStr, setSearchStr] = React.useState('');
  const [sourceStr, setSourceStr] = React.useState('{"limit": 500, "assemble_posts": true, "embed_ts":">1", "otype": "image", "order": "-ts"}');
  //const [sourceStr, setSourceStr] = React.useState('{"source": "twitter", "limit": 500, "assemble_posts": true, "embed_ts":">1", "otype": "post", "order": "-ts"}');
  //const [sourceStr, setSourceStr] = React.useState(`{"added_ts": ">=${Math.floor(Date.now() / 1000) - (24*3600)}", "assemble_posts":true, "limit":500}`);
  const [nCols, setNCols] = React.useState(IS_MOBILE ? 1 : 6);
  const [simpleMode, setSimpleMode] = React.useState(true);
  const [mode, setMode] = React.useState(MODES[0]);
  const [clusters, setClusters] = React.useState({}); // {id: {num: 1, score: 0}}
  const [message, setMessage] = React.useState('Messages show up here');

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
    // call doSource initially
    if (sourceStr) {
      setTimeout(() => doSource(), 500);
    }
  }, []);

  React.useEffect(() => {
    // if mode changes away from multicol, set nCols to 1
    if (mode !== 'multicol') {
      setNCols(1);
    }
  }, [mode, setNCols]);

  React.useEffect(() => {
    const grid = document.querySelector('.objects');
    if (grid && window.Masonry) {
      if (mode !== 'multicol') {
        //TODO do something
      }
      setTimeout(() => { // small timeout to ensure layout is ready
        const containerWidth = grid.offsetWidth;
        const columnWidth = (containerWidth - (nCols - 1) * 10) / nCols;

        // Wait for images to load
        const images = grid.querySelectorAll('img');
        let loadedImages = 0;

        const initMasonry = () => {
          if (grid.masonry) {
            grid.masonry.destroy();
          }
          const masonryInstance = new window.Masonry(grid, {
            itemSelector: '.object',
            columnWidth: columnWidth,
            gutter: 10
          });
          // Store reference for manual refresh
          grid.masonry = masonryInstance;
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
      }, 500);
    }
  }, [curIds, nCols, mode]); // Re-run when items or columns change

  // Throttled scroll handler for infinite scroll
  /*
  React.useEffect(() => {
    let ticking = false;
    const handleScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          const isAtBottom = window.innerHeight + window.scrollY >= document.body.offsetHeight - 100;
          if (isAtBottom) {
            console.log('User scrolled to bottom - fetch more data');
            // TODO: Implement fetchMoreData function
          }
          ticking = false;
        });
        ticking = true;
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
*/

  const updateData = React.useCallback((data, resetData=false) => {
    console.log('got data', data);
    if (resetData) {
      setRowById({});
      setClusters({});
    }
    // use immer to update rowById
    setRowById((rowById) => immer.produce(rowById, (draft) => {
      Object.entries(data.rows).forEach(([id, row]) => {
        if (!row.rels) {
          row.rels = {};
        }
        draft[id] = row;
      });
    }));
    setCurIds(Object.keys(data.rows));
    setAllOtypes(data.allOtypes);

    // In clustering mode, initialize clusters for new objects
    if (mode === 'cluster') {
      setClusters((prevClusters) => {
        const newClusters = resetData ? {} : {...prevClusters};
        Object.entries(data.rows).forEach(([id, row]) => {
          if (!(id in newClusters)) {
            // Initialize unlabeled items with num=1, score=0
            newClusters[id] = {num: 1, score: 0};
          }
        });
        return newClusters;
      });
    }
  }, [setRowById, setCurIds, setAllOtypes, mode, setClusters]);

  // fetch data when otypes changes
  React.useEffect(() => {
    // fetch objects from the server
    api.get({
      otype: curOtypes,
      added_ts: '>=' + (Math.floor(Date.now() / 1000) - (24*3600)), // added within the last day
      assemble_posts: true,
      limit: 500,
    }).then(updateData).catch(() => {
      // Error already handled by fetchEndpoint
    });
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
    api.action(id, likedState ? 'like' : 'unlike').catch(() => {
      // Error already handled by fetchEndpoint
    });
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

  // sets an individual item's cluster
  const setCluster = React.useCallback((id, clusterNum) => {
    console.log('setting cluster for', id, clusterNum);
    setClusters((clusters) => {
      const newClusters = {...clusters};
      if (clusterNum === null) {
        // Reset to unlabeled
        newClusters[id] = {num: 1, score: 0};
      } else {
        // Set as manual with score 1000
        newClusters[id] = {num: clusterNum, score: 1000};
      }
      // Extract manually assigned clusters and call API
      const manualClusters = {};
      Object.entries(newClusters).forEach(([objId, data]) => {
        if (data.score === 1000) {
          manualClusters[objId] = data.num;
        }
      });
      // Call cluster endpoint with manual clusters and all object IDs
      if (Object.keys(manualClusters).length > 0) {
        api.cluster(manualClusters, curIds).then((response) => {
          console.log('Got cluster response:', response);
          // Update clusters with server response, preserving manual assignments
          if (response.clusters) {
            setClusters(prevClusters => {
              const updatedClusters = {...prevClusters};
              Object.entries(response.clusters).forEach(([objId, clusterData]) => {
                // Only update if not manually assigned (score !== 1000)
                if (updatedClusters[objId].score !== 1000) {
                  updatedClusters[objId] = {
                    num: clusterData.num,
                    score: clusterData.score
                  };
                }
              });
              return updatedClusters;
            });
          }
        }).catch(() => {
          // Error already handled by fetchEndpoint
        });
      }
      return newClusters;
    });
  }, [setClusters, curIds]);

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
      console.log('got classify resp', data);
      // update curIds and scores
      if (data.curIds && data.scores){
        setCurIds(data.curIds);
        setScores(data.scores);
      }
    }).catch(() => {
      // Error already handled by fetchEndpoint
    });
  }, [pos]);

  // the source string can be either a url or a JSON string of parameters
  const doSource = React.useCallback(() => {
    const isUrl = sourceStr.startsWith('http');
    if (isUrl) { // if we got a URL, extract the params and do another fetch to /get
      api.sourceUrl(sourceStr).then((params) => {
        // save the params, serialized, to sourceStr
        //setSourceStr(JSON.stringify(params));
        return api.get(params);
      }).then((data) => {
        updateData(data, true);
      }).catch(() => {
        // Error already handled by fetchEndpoint
      });
    } else { // Parse as JSON and use as get parameters
      try {
        const params = JSON.parse(sourceStr);
        api.get(params).then((data) => {
          updateData(data, true);
        }).catch(() => {
          // Error already handled by fetchEndpoint
        });
      } catch (error) {
        console.error('Invalid JSON in source string:', error);
        setMessage(`Invalid JSON in source string: ${error.message}`);
      }
    }
  }, [sourceStr, updateData, setMessage]);

  // Function to manually refresh Masonry layout
  const refreshMasonry = React.useCallback(() => {
    const grid = document.querySelector('.objects');
    if (grid && window.Masonry && grid.masonry) {
      setTimeout(() => {
        grid.masonry.reloadItems();
        grid.masonry.layout();
      }, 100);
    }
  }, []);

  const doLikeClassifier = React.useCallback(() => {
    console.log('calling like classifier');
    api.classifyLikes('likes', 'image').then((resp) => {
      console.log('got like classifier response', resp);
      if (resp.scores) {
        setScores(resp.scores);
        // sort cur ids by score desc, if we have it for that item
        const sortedIds = curIds.sort((a, b) => {
          const scoreA = resp.scores[a] || -10;
          const scoreB = resp.scores[b] || -10;
          return scoreB - scoreA;
        });
        console.log('cur ids vs new', curIds, sortedIds);
        setCurIds(sortedIds);
        refreshMasonry()
      }
    }).catch(() => {
      // Error already handled by fetchEndpoint
    });
  }, [setCurIds, setScores, curIds, refreshMasonry]);

  const funcs = {allOtypes, curOtypes, togglePos, setCurOtypes, setCurIds,
    sourceStr, setSourceStr, doSource, filterStr, updateFilterStr, searchStr, updateSearchStr,
    setLiked, nCols, setNCols, pos, simpleMode, setSimpleMode, mode, setMode, refreshMasonry,
    clusters, setCluster, doLikeClassifier, message};
  console.log('rowById', rowById, curIds, pos, scores);
  const ids = curIds.filter(id => rowById[id] && curOtypes.includes(rowById[id].otype));

  // Group objects by cluster for cluster mode
  const renderClusterColumns = () => {
    const clusterGroups = {1: [], 2: [], 3: [], 4: [], 5: []};
    ids.forEach(id => {
      const clusterNum = clusters[id].num || 1;
      clusterGroups[clusterNum].push(id);
    });

    return (
      <div className="cluster-columns">
        {[1, 2, 3, 4, 5].map(clusterNum => (
          <div key={clusterNum}
            className={`cluster-column ${clusterGroups[clusterNum].length > 0 ? 'has-items' : ''}`}
          >
            <h4>Cluster {clusterNum}</h4>
            {clusterGroups[clusterNum].map(id => (
              <Obj key={id} score={scores[id]} {...funcs} {...rowById[id]} />
            ))}
          </div>
        ))}
      </div>
    );
  };

  return (
    <MessageContext.Provider value={setMessage}>
      <div>
        <h3>Collections</h3>
        <h4>Labeled</h4>
        <div className="labeled">
          {pos.map((id) => <Obj key={id} {...funcs} {...rowById[id]} />)}
        </div>
        <Controls {...funcs} />
        {mode === 'cluster' ? (
          renderClusterColumns()
        ) : (
          <div
            className="objects"
            style={{
              gridTemplateColumns: `repeat(${nCols}, 1fr)`,
              '--n-cols': nCols
            }}
          >
            {ids.map((id) => <Obj key={id} score={scores[id]} {...funcs} {...rowById[id]} />)}
          </div>
        )}
      </div>
    </MessageContext.Provider>
  );
}

ReactDOM.createRoot(document.getElementById("main")).render(<App />);
