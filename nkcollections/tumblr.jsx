/* Tumblr components */


/* TumblrContentBlock - Renders individual content blocks within Tumblr posts
 *
 * Props:
 * - block: Object with {type, data} representing a content block (text, link, image, video, etc.)
 */
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

/* TumblrPostContent - Renders the complete content of a Tumblr post
 *
 * Props:
 * - id: Post ID
 * - otype: Object type ('post')
 * - url: Original post URL
 * - md: Metadata object containing tags, n_notes, n_likes, n_reblogs
 * - score: Classification score (optional)
 * - simpleMode: Boolean to hide detailed info like scores and IDs
 * - content_blocks: Array of content blocks to render (text, links, media, etc.)
 */
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
