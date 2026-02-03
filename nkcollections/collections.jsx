/* NK Collections React App
 *
 * TODO likes toggle on page
 * TODO show tags on posts
 * TODO grouping by sim
 * TODO grouping by post
 * TODO quick zoom
 * TODO show dwell times
 * TODO detect broken img
 * TODO dislike btn?
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
const AUTO_LIKES_DELAY_MS = 15000;
const MODES = ['multicol', 'cluster'];

const QUICK_LINKS = {
  'Queued': '{"rels.queue":true}',
  'Q users': '{"otype": "user", "limit": 100, "order": "-lambda o: o.md[\'n_queued_reblogs\']"}',
  'Images': '{"otype":["image", "video"],"limit":500,"embed_ts":">1"}',
  'Posts': '{"otype":"post","limit":500}',
  'Twitter': '{"source":"twitter","limit":500}',
  'Tumblr': '{"source":"tumblr","limit":500}',
  'Users': '{"otype":"user"}',
};

// Detect if we're on a mobile device
const IS_MOBILE = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || (window.innerWidth <= 768);

// Global reference for message handling
let globalSetMessage = null;
let globalSetSourceStr = null;

// Utility function for making API calls
const fetchEndpoint = async (endpoint, data = {}, options = {}) => {
  const t0 = Date.now();
  // if we don't have globalSetMessage yet, set a temporary one
  const setMessage = globalSetMessage || ((msg) => console.log('Message:', msg));
  const {
    method = 'POST',
    headers = { 'Content-Type': 'application/json' },
    onError = (error) => console.error('Fetch error:', error),
    ...fetchOptions
  } = options;

  try {
    setMessage(`Calling API: ${endpoint}...`);
    const response = await fetch(endpoint, {
      method,
      headers,
      body: method !== 'GET' ? JSON.stringify(data) : undefined,
      ...fetchOptions
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const responseData = await response.json();
    // Check for success message in response
    if (responseData.msg) {
      setMessage(`API resp: ${responseData.msg} in ${Date.now() - t0}ms`);
    }
    return responseData;
  } catch (error) {
    setMessage(`API call failed: ${endpoint} - ${error.message}`);
    onError(error);
    throw error;
  }
};

// API helper functions
const api = {
  get: (params) => fetchEndpoint('/get', params),
  classify: (pos) => fetchEndpoint('/classify', { pos }),
  classifyLikes: (options) => fetchEndpoint('/classify', options),
  action: (ids, action) => fetchEndpoint('/action', { ids, action }),
  sourceUrl: (url) => fetchEndpoint('/source', { url }),
  cluster: (clusters, ids) => fetchEndpoint('/cluster', { clusters, ids }),
  filter: (q, cur_ids) => fetchEndpoint('/filter', { q, cur_ids }),
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

.object.positive {
  border-color: #4CAF50;
}

.object.liked {
  border-style: dashed;
  border-width: 3px;
}

.object.queued {
  border-color: orange;
  border-width: 3px;
}

.object.single-col {
  max-width: 400px!important;
}

.user-compact {
  font-size: 0.8em;
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

.infobar {
  display: flex;
  flex-flow: flex-end;
  flex-wrap: wrap;
  position: fixed;
  z-index: 100;
  background: lightgray;
  top: 0;
  padding: 5px;
  right: 0;
  font-size: 0.8em;
}

.infobar .control {
  margin: 0 5px;
}

.labeled, .controls {
  display: flex;
  flex-wrap: wrap;
}

.control {
  margin-right: 10px;
}

.message-display {
  font-size: 0.8em;
}

.text-fields input {
  margin-right: 10px;
}

.search-input {
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

.queue-icon {
  color: #666;
}

.queue-icon.queued {
  color: #007bff;
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

/* Video overlay icon */
.video-overlay {
  position: absolute;
  top: 5px;
  right: 5px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  font-size: 12px;
  padding: 2px 4px;
  border-radius: 3px;
  text-decoration: none;
  pointer-events: auto;
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


.video-link a {
  position: 'absolute';
  top: '5px';
  right: '5px';
  background: 'rgba(0, 0, 0, 0.7)';
  color: 'white';
  fontSize: '12px';
  padding: '2px 4px';
  borderRadius: '3px';
  textDecoration: 'none';
  pointerEvents: 'auto';
}

.src-input {
  font-size: 0.7em;
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

/* Auto likes timer progress styling */
.like-classifier button.timer-active {
  background: linear-gradient(
    to right,
    #007bff 0%,
    #007bff var(--progress, 0%),
    #f8f9fa var(--progress, 0%),
    #f8f9fa 100%
  );
  transition: background 0.1s ease;
  color: white;
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

/* Quick links styles */
.quick-links {
  display: flex;
  overflow-x: auto;
  gap: 8px;
  scrollbar-width: thin;
  -webkit-overflow-scrolling: touch;
  max-height: 20px;
  /*max-width: 350px; doesn't seem to be needed */
}

.quick-links button {
  white-space: nowrap;
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

// section for mobile-specific styles
@media (max-width: 768px) {
  .infobar {

  }
  .infobar .control {
    margin: 5px 0;
  }
  .object {
    max-width: calc(100vw - 20px)!important;
  }
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

const VideoOverlay = ({videoUrl, onClick}) => {
  if (!videoUrl) return null;
  
  return (
    <a 
      href={videoUrl} 
      target="_blank" 
      rel="noreferrer"
      onClick={(e) => {
        e.stopPropagation();
        if (onClick) onClick(e);
      }}
      className="video-overlay"
    >
      ▶
    </a>
  );
};

const MediaCarousel = ({mediaBlocks, currentIndex, setCurrentIndex, setLiked}) => {
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
        const imageUrl = data.local_path ? `/data/${data.local_path}` : data.url;
        const videoUrl = data.md && data.md.video_url;
        return (
          <div style={{position: 'relative'}}>
            <img
              src={imageUrl}
              alt={`Image ${data.id}`}
              onClick={handleImageClick}
              onDoubleClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                const liked = Boolean(data.rels.like);
                setLiked(data.id, !liked);
              }}
              style={{cursor: mediaBlocks.length > 1 ? 'pointer' : 'default'}}
            />
            <VideoOverlay videoUrl={videoUrl} />
          </div>
        );
      case 'video':
        const posterUrl = data.md.poster_url && data.local_path ? `/data/${data.local_path}` : data.md.poster_url;
        return (
          <div className="video-link" style={{position: 'relative'}}>
            <img
              src={posterUrl}
              alt={`Video ${data.id} poster`}
              onClick={handleImageClick}
              style={{cursor: mediaBlocks.length > 1 ? 'pointer' : 'default'}}
            />
            <a 
              href={data.url} 
              target="_blank" 
              rel="noreferrer"
              onClick={(e) => e.stopPropagation()}
            >
              ▶
            </a>
          </div>
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
  const ctx = React.useContext(AppContext);
  let {id, otype, url, md, score, rels, source, media_blocks} = props;
  media_blocks = media_blocks || [];
  //console.log('Obj', id, otype, score, props);
  const liked = Boolean(rels.like);
  const queued = Boolean(rels.queue);
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
      title="Previous media"
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
      title="Next media"
    >
      →
    </div>),
  ];

  let classes = ['object', otype, `source-${source}`, `otype-${otype}`];
  if (score !== undefined && score > 0) {
    classes.push('positive');
  }
  if (liked) {
    classes.push('liked');
  }
  if (queued) {
    classes.push('queued');
  }
  let cClasses = ['icon-button', 'classify-icon', (ctx.data.pos.includes(id) ? 'selected' : '')];
  if (ctx.ui.mode !== 'multicol') {
    classes.push('single-col');
  } else {
    cClasses.push('hidden');
  }
  if (isHovered && ctx.ui.mode === 'cluster') {
    classes.push('keyboard-active');
  }

  // Current cluster for this object
  const currentCluster = (ctx.data.clusters[id]) ? ctx.data.clusters[id].num : null;
  const clusterScore = (ctx.data.clusters[id]) ? ctx.data.clusters[id].score : 0;
  const isManualCluster = (ctx.data.clusters[id]) ? ctx.data.clusters[id].score === 1000 : false;
  const isUnlabeled = (ctx.data.clusters[id]) ? ctx.data.clusters[id].score === 0 : false;
  const normalizedScore = isManualCluster ? 1.0 : clusterScore; // Manual clusters get full opacity
  // Keyboard event handler for cluster assignment
  React.useEffect(() => {
    if (!isHovered || ctx.ui.mode !== 'cluster') return;
    const handleKeyPress = (e) => {
      const num = parseInt(e.key);
      if (num >= 0 && num <= 5) {
        e.preventDefault();
        ctx.actions.setCluster(id, currentCluster === num ? null : num);
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isHovered, ctx.ui.mode, id, currentCluster, ctx.actions.setCluster]);

  // Add cluster assignment visual indicators
  if (ctx.ui.mode === 'cluster') {
    if (ctx.data.clusters[id]) {
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
      style={ctx.ui.mode === 'cluster' && ctx.data.clusters[id] ? {
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
            ctx.actions.setLiked(id, !liked);
          }}
          title={liked ? "Unlike this item" : "Like this item"}
        >
          ♥
        </div>
        <div
          className={cClasses.join(' ')}
          onClick={(e) => {
            e.stopPropagation();
            ctx.actions.togglePos(id);
          }}
          title={ctx.data.pos.includes(id) ? "Remove from positive examples" : "Add to positive examples"}
        >
          🎯
        </div>
        <div
          className="icon-button open-icon"
          onClick={(e) => {
            e.stopPropagation();
            window.open(url, '_blank');
          }}
          title="Open original URL in new tab"
        >
          🔗
        </div>
        {props.parent_url && (
          <div
            className="icon-button parent-icon"
            onClick={(e) => {
              e.stopPropagation();
              window.open(props.parent_url, '_blank');
            }}
            title="Open parent URL"
          >
            ⬆️
          </div>
        )}
        <div
          className={`icon-button queue-icon ${queued ? 'queued' : ''}`}
          onClick={(e) => {
            e.stopPropagation();
            ctx.actions.setQueued(id, !queued);
          }}
          title={queued ? "Remove from queue" : "Add to queue"}
        >
          {queued ? '➖' : '➕'}
        </div>
        {otype === 'user' && (
          <div
            className="icon-button explore-user-icon"
            onClick={(e) => {
              e.stopPropagation();
              ctx.actions.doAction([id], 'explore');
            }}
            title="Explore this user (fetch their posts)"
          >
            🔍
          </div>
        )}
        {otype === 'user' && (
          <div
            className="icon-button show-user-posts-icon"
            onClick={(e) => {
              e.stopPropagation();
              const query = `{"ancestor":${id}}`;
              ctx.actions.doSource(query, true);
            }}
            title="Show all posts from this user"
          >
            📚
          </div>
        )}
        {/* Media navigation controls - only show if multiple media */}
        {hasMultipleMedia && mediaDivs}
        {/* Cluster buttons - only show in cluster mode */}
        {ctx.ui.mode === 'cluster' && (
          <div className="cluster-buttons" title={`Cluster ${currentCluster}: ${clusterScore}`}>
            {[1, 2, 3, 4, 5].map(clusterNum => (
              <div
                key={clusterNum}
                className={`cluster-button ${currentCluster === clusterNum ? 'active' : ''} ${isManualCluster && currentCluster === clusterNum ? 'manual' : 'automatic'}`}
                onClick={(e) => {
                  e.stopPropagation();
                  ctx.actions.setCluster(id, currentCluster === clusterNum ? null : clusterNum);
                }}
                title={currentCluster === clusterNum ? `Remove from cluster ${clusterNum}` : `Assign to cluster ${clusterNum}`}
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
              <div style={{position: 'relative'}}>
                <img
                  src={props.local_path ? `/data/${props.local_path}` : url}
                  alt={`Image ${id}`}
                  onDoubleClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    ctx.actions.setLiked(id, !liked);
                  }}
                />
                <VideoOverlay videoUrl={md && md.video_url} />
              </div>
            </div>
          )}
          {otype === 'video' && (
            <div className="content">
              <div className="video-link" style={{position: 'relative'}}>
                <img src={md.poster_url} alt={`Video ${id} poster`} />
                <a 
                  href={url} 
                  target="_blank" 
                  rel="noreferrer"
                  onClick={(e) => e.stopPropagation()}
                >
                  ▶
                </a>
              </div>
            </div>
          )}
          {otype === 'user' && (
            <div className="content">
              <div className="user-compact" dangerouslySetInnerHTML={{__html: props.compact}}></div>
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
          setLiked={ctx.actions.setLiked}
        />
      )}

    </div>
  );
}

// A floating info/control panel
const InfoBar = () => {
  const ctx = React.useContext(AppContext);
  // Local state for filter string and auto-likes timer
  const [filterStr, setFilterStr] = React.useState('');
  const [autoLikesElapsed, setAutoLikesElapsed] = React.useState(0);
  const autoLikesTimerRef = React.useRef(null);

  // Auto likes mode timer effect
  React.useEffect(() => {
    if (ctx.classification.autoLikesMode) {
      // Reset elapsed time and start tracking
      setAutoLikesElapsed(0);
      const startTime = Date.now();
      // Start the recurring timer for classifier
      autoLikesTimerRef.current = setInterval(() => {
        ctx.actions.doLikeClassifier();
        setAutoLikesElapsed(0); // Reset elapsed time after each run
      }, AUTO_LIKES_DELAY_MS);

      // Start elapsed time tracking (update every 100ms for smooth progress)
      const elapsedTimer = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const cycleElapsed = elapsed % AUTO_LIKES_DELAY_MS;
        setAutoLikesElapsed(cycleElapsed);
      }, 100);

      // Store both timers for cleanup
      autoLikesTimerRef.current = {
        classifierTimer: autoLikesTimerRef.current,
        elapsedTimer: elapsedTimer
      };

      ctx.ui.setMessage(`Auto likes mode enabled - will run classifier every ${AUTO_LIKES_DELAY_MS/1000}s`);
    } else {
      // Clear the timers if they exist
      if (autoLikesTimerRef.current) {
        if (autoLikesTimerRef.current.classifierTimer) {
          clearInterval(autoLikesTimerRef.current.classifierTimer);
        }
        if (autoLikesTimerRef.current.elapsedTimer) {
          clearInterval(autoLikesTimerRef.current.elapsedTimer);
        }
        autoLikesTimerRef.current = null;
        setAutoLikesElapsed(0);
        ctx.ui.setMessage('Auto likes mode disabled');
      }
    }

    // Cleanup function to clear timers on unmount
    return () => {
      if (autoLikesTimerRef.current) {
        if (autoLikesTimerRef.current.classifierTimer) {
          clearInterval(autoLikesTimerRef.current.classifierTimer);
        }
        if (autoLikesTimerRef.current.elapsedTimer) {
          clearInterval(autoLikesTimerRef.current.elapsedTimer);
        }
        autoLikesTimerRef.current = null;
      }
    };
  }, [ctx.classification.autoLikesMode, ctx.actions.doLikeClassifier, ctx.ui.setMessage]);
  const incrCols = (incr) => {
    ctx.ui.setNCols((nCols) => {
      let newCols = nCols + incr;
      if (newCols < 1) newCols = 1;
      if (newCols > 20) newCols = 20;
      return newCols;
    });
  }
  const goTo = (name) => {
    switch (name) {
      case 'top': // top of page
        window.scrollTo({top: 0, behavior: 'smooth'});
        break;
      case 'mid': // middle of page
        window.scrollTo({top: document.body.scrollHeight / 2, behavior: 'smooth'});
        break;
      case 'bot': // bottom of page
        window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
        break;
    }
  }
  // sort scores and compute stats
  const curIds = ctx.data.curIds || [];
  const n = curIds.length;
  const nWithScores = curIds.filter(id => ctx.data.scores[id] !== undefined).length;
  const nWithLikes = curIds.map(id => ctx.data.rowById[id].rels.like).filter(like => like).length;
  let sscores = [];
  if (n > 0) {
    sscores = curIds.map(id => ctx.data.scores[id] || null).filter(s => s !== null).sort((a, b) => a - b);
  }
  //console.log('got sscores', sscores, curIds, scores);
  const nPos = (sscores.length > 0) ? sscores.filter((s) => s > 0).length : 0;
  const pPos = (sscores.length > 0) ? 100.0 * nPos / sscores.length : 0;
  const doFilter = (v) => {
    ctx.actions.doFilter(v);
    setFilterStr('');
    // also unfocus the input
    const inputElem = document.querySelector('.filter-input');
    if (inputElem) {
      inputElem.blur();
    }
  }

  return (
    <div className="infobar">
      <div>{n} items ({nWithScores} scored, {nWithLikes} liked) </div>
      {sscores.length > 0 && (
        <div className="score-stats">
          , {nPos} ({pPos.toFixed(1)}%) pos
        </div>)}
      <span>Cols:</span>
      <div className="control decr-cols"><button onClick={() => incrCols(-1)} title="Decrease number of columns">-</button></div>
      <span>{ctx.ui.nCols}</span>
      <div className="control incr-cols"><button onClick={() => incrCols(1)} title="Increase number of columns">+</button></div>
      <div className="control randomize-btn">
        <button onClick={() => {
          // shuffle curIds
          ctx.filters.setCurIds((curIds) => {
            const shuffled = [...curIds];
            for (let i = shuffled.length - 1; i > 0; i--) {
              const j = Math.floor(Math.random() * (i + 1));
              [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }
            return shuffled;
          });
        }} title="Randomize the order of items">Randomize</button>
      </div>
      <div className="control simple-mode">
        <label title="Hide detailed information like scores and IDs">
          <input
            type="checkbox"
            checked={ctx.ui.simpleMode}
            onChange={(e) => ctx.ui.setSimpleMode(e.target.checked)}
            title="Hide detailed information like scores and IDs"
          />
          Simple
        </label>
      </div>
      <div className="control auto-likes-mode hidden">
        <label title="Automatically run the likes classifier every 15 seconds">
          <input
            type="checkbox"
            checked={ctx.classification.autoLikesMode}
            onChange={(e) => {ctx.classification.setAutoLikesMode(e.target.checked); if (e.target.checked) ctx.actions.doLikeClassifier();}}
            title="Automatically run the likes classifier every 15 seconds"
          />
          Auto Likes
        </label>
      </div>
      <div className="control like-classifier">
        <button
          className={ctx.classification.autoLikesMode ? 'timer-active' : ''}
          style={ctx.classification.autoLikesMode ? {'--progress': `${(autoLikesElapsed / AUTO_LIKES_DELAY_MS) * 100}%`} : {}}
          onClick={ctx.actions.doLikeClassifier}
          title="Run the likes-based classifier to score items"
        >
          Like Classifier
        </button>
      </div>
      <div className="control filter-control">
        <DebouncedInput
          value={filterStr}
          onChange={setFilterStr}
          onDebouncedChange={doFilter}
          placeholder="Filter..."
          className="filter-input"
          title="Filter items by text"
          delay={DEBOUNCE_MS}
        />
      </div>
      <div className="control go-to-top"><button onClick={() => goTo('top')} title="Scroll to top of page">Top</button></div>
      <div className="control go-to-mid"><button onClick={() => goTo('mid')} title="Scroll to middle of page">Mid</button></div>
      <div className="control go-to-bot"><button onClick={() => goTo('bot')} title="Scroll to bottom of page">Bot</button></div>
      <div className="flex-break"></div>
      <div className="control message-display">
        <span>{ctx.ui.message}</span>
      </div>
    </div>
  );
}


const DebouncedInput = ({
  value,
  onChange,
  onDebouncedChange,
  delay = DEBOUNCE_MS,
  placeholder = "",
  className = "",
  title = ""
}) => {
  const [localValue, setLocalValue] = React.useState(value);
  const timeoutRef = React.useRef(null);

  // Update local value when prop changes
  React.useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleChange = (e) => {
    const newValue = e.target.value;
    setLocalValue(newValue);
    onChange(newValue); // Immediate update for UI

    // Clear existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Set new timeout for debounced action
    timeoutRef.current = setTimeout(() => {
      onDebouncedChange(newValue);
    }, delay);
  };

  return (
    <input
      type="text"
      className={className}
      placeholder={placeholder}
      value={localValue}
      onChange={handleChange}
      title={title}
    />
  );
};

const Controls = () => {
  const ctx = React.useContext(AppContext);
  
  // Local state for search string and source string
  const [searchStr, setSearchStr] = React.useState('');
  const [sourceStr, setSourceStr] = React.useState(QUICK_LINKS['Images']);
  //const [sourceStr, setSourceStr] = React.useState('{"source": "twitter", "limit": 500, "embed_ts":">1", "otype": "post"}');
  //const [sourceStr, setSourceStr] = React.useState(`{"added_ts": ">=${Math.floor(Date.now() / 1000) - (24*3600)}", "assemble_posts":true, "limit":500}`);

  // Set up global reference to setSourceStr
  React.useEffect(() => {
    globalSetSourceStr = setSourceStr;
  }, [setSourceStr]);

  // Check for source parameter in URL on page load
  React.useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sourceFromUrl = params.get('source');
    if (sourceFromUrl) {
      const decodedSource = decodeURIComponent(sourceFromUrl);
      setSourceStr(decodedSource);
      setTimeout(() => ctx.actions.doSource(decodedSource), 500);
    } else if (sourceStr) {
      // Use default source if no URL parameter
      setTimeout(() => ctx.actions.doSource(sourceStr), 500);
    }
  }, []); // Only run once on mount
  
  // add a "return" key handler for the source input
  const keyHandler = (e) => {
    if (e.key === 'Enter') {
      ctx.actions.doSource(sourceStr);
    }
  }

  // Function to get clipboard contents and set as source
  const doSourceFromClipboard = React.useCallback(async () => {
    try {
      if (!navigator.clipboard || !navigator.clipboard.readText) {
        console.error('Clipboard API not supported in this browser');
        return;
      }
      const clipboardText = await navigator.clipboard.readText();
      if (!clipboardText.trim()) {
        console.error('Clipboard is empty');
        return;
      }
      const str = clipboardText.trim();
      setSourceStr(str);
      ctx.actions.doSource(str);
    } catch (error) {
      console.error('Failed to read clipboard:', error);
    }
  }, [ctx.actions.doSource]);

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
          size="52"
        />
        <button onClick={() => ctx.actions.doSource(sourceStr)} title="Load data from the source string">Set Source</button>
        {!IS_MOBILE && (
          <button className="source-from-clipboard-btn" onClick={doSourceFromClipboard} title="Paste source from clipboard and load">Source from Clipboard</button>
        )}
      </div>
      <div className="control quick-links">
        {Object.entries(QUICK_LINKS).map(([name, query]) => (
          <button
            key={name}
            onClick={() => {setSourceStr(query); ctx.actions.doSource(query)}}
            title={`Load: ${name}`}
          >
            {name}
          </button>
        ))}
      </div>
      <DebouncedInput
        value={searchStr}
        onChange={setSearchStr}
        onDebouncedChange={ctx.actions.doSearch}
        placeholder="Search..."
        className="search-input"
        title="Search items by text (not yet implemented)"
        delay={DEBOUNCE_MS}
      />
      <div className="control otype-filters">
      {ctx.data.allOtypes.map((otype) => (
        <label key={otype} style={{marginRight: '10px'}}>
          <input
            type="checkbox"
            checked={ctx.filters.curOtypes.includes(otype)}
            onChange={(e) => {
              ctx.filters.setCurOtypes((curOtypes) => {
                if (e.target.checked) {
                  return [...curOtypes, otype];
                } else {
                  return curOtypes.filter((x) => x !== otype);
                }
              });
            }}
            title={`Show/hide ${otype} items`}
          />
          {otype}
        </label>
      ))}
      </div>
      <div className="control mode-select">
        <label>Mode:</label>
        <select
          value={ctx.ui.mode}
          onChange={(e) => ctx.ui.setMode(e.target.value)}
          style={{marginLeft: '5px'}}
          title="Switch between multi-column and cluster view modes"
        >
          {MODES.map((modeOption) => (
            <option key={modeOption} value={modeOption}>
              {modeOption}
            </option>
          ))}
        </select>
      </div>
      <div className="control refresh-masonry">
        <button onClick={ctx.ui.refreshMasonry} title="Refresh the masonry layout">Refresh layout</button>
      </div>
      <div className="control flex-break"></div>
    </div>
  );
}


// Create the App Context
const AppContext = React.createContext();

const AppProvider = ({ children }) => {
  const [rowById, setRowById] = React.useState({});
  const [allOtypes, setAllOtypes] = React.useState([]);
  //const [curOtypes, setCurOtypes] = React.useState(['post', 'image', 'text', 'link']);
  const [curOtypes, setCurOtypes] = React.useState(['image', 'video', 'user']);
  const [curIds, setCurIds] = React.useState([]);
  const [scores, setScores] = React.useState({});
  const [pos, setPos] = React.useState([]);
  const [nCols, setNCols] = React.useState(IS_MOBILE ? 2 : 6);
  const [simpleMode, setSimpleMode] = React.useState(false);
  const [mode, setMode] = React.useState(MODES[0]);
  const [clusters, setClusters] = React.useState({}); // {id: {num: 1, score: 0}}
  const [autoLikesMode, setAutoLikesMode] = React.useState(false);
  const [message, setMessage] = React.useState('Messages show up here');
  const [currentSource, setCurrentSource] = React.useState('');

  // initial init
  React.useEffect(() => {
    document.title = 'NK Collections';
    // insert styles
    const styleEl = document.createElement('style');
    styleEl.innerHTML = STYLES;
    document.head.appendChild(styleEl);
    // add favicon
    const faviconEl = document.createElement('link');
    faviconEl.rel = 'icon';
    faviconEl.href = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🖼️</text></svg>";
    document.head.appendChild(faviconEl);
  }, []);

  // Set up global reference to setMessage
  React.useEffect(() => {
    globalSetMessage = setMessage;
  }, [setMessage]);

  // Handle browser back/forward navigation
  React.useEffect(() => {
    const handlePopState = (event) => {
      const params = new URLSearchParams(window.location.search);
      const sourceFromUrl = params.get('source');
      if (sourceFromUrl) {
        const decodedSource = decodeURIComponent(sourceFromUrl);
        if (decodedSource !== currentSource && globalSetSourceStr) {
          globalSetSourceStr(decodedSource);
          setCurrentSource(decodedSource);
          // Actually execute the source query to load the data
          const isUrl = decodedSource.startsWith('http');
          if (isUrl) {
            api.sourceUrl(decodedSource).then((params) => {
              return api.get(params);
            }).then((data) => {
              updateData(data, true);
            }).catch(() => {
              // Error already handled by fetchEndpoint
            });
          } else {
            try {
              const params = JSON.parse(decodedSource);
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
        }
      }
    };
    
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [currentSource, updateData, setMessage]);


  const doSearch = React.useCallback((value) => {
    console.log('searching for', value);
    //TODO implement
  }, []);

  const doFilter = React.useCallback((value) => {
    if (!value || value.trim() === '') return;
    console.log('filtering for', value);
    api.filter(value, curIds).then((resp) => {
      console.log('got filter resp', resp);
      updateScores(resp.scores, {reset: false});
    });
  }, [curIds, updateScores]);

  // Mode changes
  React.useEffect(() => {
    // if mode changes away from multicol, set nCols to 1
    if (mode !== 'multicol') {
      setNCols(1);
    }
  }, [mode, setNCols]);

  // Setup masonry for tight grid
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

  const updateRowById = React.useCallback((updatedRows) => {
    // use immer to update rowById
    setRowById((rowById) => immer.produce(rowById, (draft) => {
      Object.entries(updatedRows).forEach(([id, row]) => {
        draft[id] = row;
      });
    }));
  }, [setRowById]);

  // called whenever main data updates
  const updateData = React.useCallback((data, resetData=false) => {
    console.log('got data', data, resetData);
    if (resetData) {
      setRowById({});
      setClusters({});
    }
    updateRowById(data.row_by_id);
    if (data.cur_ids){
      setCurIds(data.cur_ids);
    } else {
      setCurIds(Object.keys(data.row_by_id));
    }
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
  }, [updateRowById, setCurIds, setAllOtypes, mode, setClusters]);

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

  // update scores for items on page
  const updateScores = React.useCallback((newScores, options) => {
    if (!newScores) return;
    const {reset=true} = options || {};
    let curScores = {};
    if (reset) {
      curScores = newScores;
    } else {
      Object.entries(newScores).forEach(([id, score]) => {
        const s = scores[id] || 1.0;
        curScores[id] = s * score;
      });
    }
    setScores(curScores);
    //console.log('current ids before', curIds, curScores);
    const nScoredIds = curIds.filter(id => (curScores[id] !== undefined)).length;
    // sort cur ids by score desc, if we have it for that item
    const sortedIds = curIds.sort((a, b) => {
      const scoreA = curScores[a] || -1000;
      const scoreB = curScores[b] || -1000;
      return scoreB - scoreA;
    });
    //console.log('cur ids vs new', curIds, sortedIds);
    setCurIds(sortedIds);
    refreshMasonry()
  }, [scores, setScores, setCurIds, curIds, refreshMasonry]);

  // Call like-based classifier
  const doLikeClassifier = React.useCallback(() => {
    const options = {
      type: 'likes',
      otypes:['image'],
      cur_ids: curIds,
    };
    api.classifyLikes(options).then((resp) => {
      console.log('got like classifier response', resp);
      updateScores(resp.scores);
    });
  }, [curIds]);

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

  // generic action handler
  const doAction = React.useCallback(async (ids, action) => {
    console.log('performing action', action, 'on ids', ids);
    const resp = await api.action(ids, action);
    console.log('action resp', resp);
    if (resp.updated_rows) {
      updateRowById(resp.updated_rows);
    }
    return resp;
  }, [updateRowById]);

  const setLiked = React.useCallback((id, likedState) => {
    const action = likedState ? 'like' : 'unlike';
    doAction([id], action);
  }, [doAction]);

  const setQueued = React.useCallback((id, queueState) => {
    const action = queueState ? 'queue' : 'unqueue';
    doAction([id], action);
  }, [doAction]);

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
  const doSource = React.useCallback((inputStr, updateSourceStr = false) => {
    if (!inputStr) return;
    
    // Update URL parameters
    setCurrentSource(inputStr);
    const params = new URLSearchParams(window.location.search);
    params.set('source', encodeURIComponent(inputStr));
    window.history.pushState({}, '', `?${params.toString()}`);
    
    // Update the source string if requested
    if (updateSourceStr && globalSetSourceStr) {
      globalSetSourceStr(inputStr);
    }
    // Unfocus the source input
    const sourceInput = document.querySelector('.src-input');
    if (sourceInput) {
      sourceInput.blur();
    }
    const isUrl = inputStr.startsWith('http');
    if (isUrl) { // if we got a URL, extract the params and do another fetch to /get
      api.sourceUrl(inputStr).then((params) => {
        return api.get(params);
      }).then((data) => {
        updateData(data, true);
      }).catch(() => {
        // Error already handled by fetchEndpoint
      });
    } else { // Parse as JSON and use as get parameters
      try {
        const params = JSON.parse(inputStr);
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
  }, [updateData, setMessage, setCurrentSource]);


  // Done with all state and effects, now preparing for rendering
  const ids = curIds.filter(id => rowById[id] && curOtypes.includes(rowById[id].otype));

  // Organize all state and functions into nested groups
  const contextValue = {
    data: {
      rowById,
      allOtypes,
      curIds: ids,
      scores,
      clusters,
      pos
    },
    ui: {
      nCols,
      setNCols,
      simpleMode,
      setSimpleMode,
      mode,
      setMode,
      message,
      setMessage,
      refreshMasonry
    },
    filters: {
      curOtypes,
      setCurOtypes,
      setCurIds
    },
    actions: {
      setLiked,
      togglePos,
      doSource,
      doLikeClassifier,
      setQueued,
      setCluster,
      doFilter,
      doSearch,
      doAction
    },
    history: {
      currentSource
    },
    classification: {
      autoLikesMode,
      setAutoLikesMode
    }
  };

  console.log('rowById', rowById, curIds, pos, scores);

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};

const App = () => {
  const ctx = React.useContext(AppContext);

  // Group objects by cluster for cluster mode
  const renderClusterColumns = () => {
    const clusterGroups = {1: [], 2: [], 3: [], 4: [], 5: []};
    ctx.data.curIds.forEach(id => {
      const clusterNum = ctx.data.clusters[id].num || 1;
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
              <Obj key={id} score={ctx.data.scores[id]} {...ctx.data.rowById[id]} />
            ))}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div>
      <h3>Collections</h3>
      <h4>Labeled</h4>
      <div className="labeled">
        {ctx.data.pos.map((id) => <Obj key={id} {...ctx.data.rowById[id]} />)}
      </div>
      <Controls />
      <InfoBar />
      {ctx.ui.mode === 'cluster' ? (
        renderClusterColumns()
      ) : (
        <div
          className="objects"
          style={{
            gridTemplateColumns: `repeat(${ctx.ui.nCols}, 1fr)`,
            '--n-cols': ctx.ui.nCols
          }}
        >
          {ctx.data.curIds.map((id) => <Obj key={id} score={ctx.data.scores[id]} {...ctx.data.rowById[id]} />)}
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("main")).render(
  <AppProvider>
    <App />
  </AppProvider>
);
