/* Twitter components */

/* TwitterContentBlock - Renders individual content blocks within Twitter posts
 *
 * Props:
 * - block: Object with {type, data} representing a content block (text, image, video, etc.)
 */
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

/* TwitterPostContent - Renders the complete content of a Twitter post
 *
 * Props:
 * - id: Post ID
 * - otype: Object type ('post')
 * - url: Original post URL
 * - ts: Timestamp
 * - md: Metadata object containing display_name, handle, replies, reposts, likes, views
 * - score: Classification score (optional)
 * - simpleMode: Boolean to hide detailed info like scores and IDs
 * - content_blocks: Array of content blocks to render (text, media, etc.)
 */
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

