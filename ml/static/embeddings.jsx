// Note that this is imported in the main HTML file directly, so you have to use React.xxx for everything

// Create a context for sharing point data
const PtCtx = React.createContext(null);
const { useState } = React;

const PtItem = ({id, idx}) => {
  return (
    <div className="pt">
      <div>{id}</div>
      <div className="pt-idx">{idx}: pt</div>
    </div>
  );
}

// Displays points in an ordered list with pagination and gap controls
const PtList = () => {
  const ptData = React.useContext(PtCtx);
  const [startIdx, setStartIdx] = useState(0);
  const [gap, setGap] = useState(1);
  const [pageSize, setPageSize] = useState(500);

  const maxStartIdx = Math.max(0, ptData.ids.length - pageSize);
  const visibleIds = ptData.ids
    .slice(startIdx, startIdx + pageSize * gap)
    .filter((_, i) => i % gap === 0);

  return (
    <div>
      <div style={{marginBottom: '10px'}}>
        <label>
          Start: 
          <input 
            type="number" 
            value={startIdx}
            min={0}
            max={maxStartIdx}
            onChange={e => setStartIdx(Math.min(maxStartIdx, Math.max(0, parseInt(e.target.value) || 0)))}
            style={{width: '80px', marginLeft: '5px', marginRight: '15px'}}
          />
        </label>
        <label>
          Gap: 
          <input 
            type="number" 
            value={gap}
            min={1}
            onChange={e => setGap(Math.max(1, parseInt(e.target.value) || 1))}
            style={{width: '60px', marginLeft: '5px', marginRight: '15px'}}
          />
        </label>
        <label>
          Page Size: 
          <input 
            type="number" 
            value={pageSize}
            min={1}
            max={1000}
            onChange={e => setPageSize(Math.max(1, Math.min(1000, parseInt(e.target.value) || 500)))}
            style={{width: '70px', marginLeft: '5px'}}
          />
        </label>
      </div>
      <div>Showing {visibleIds.length} points from {startIdx} to {startIdx + pageSize * gap}</div>
      <div className="ptList">
        {visibleIds.map((id, idx) => (
          <PtItem key={id} id={id} idx={startIdx + idx * gap} />
        ))}
      </div>
    </div>
  );
}

// A view is one way to look at our data - could be a scatter plot, list, etc
const View = ({id, onDelete}) => {
  return (
    <div className="view" style={{border: '1px solid #ccc', margin: '10px', padding: '10px'}}>
      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '10px'}}>
        <h4>View {id}</h4>
        <button onClick={() => onDelete(id)}>×</button>
      </div>
      <PtList />
    </div>
  );
}

const Main = () => {
  const [ids, setIds] = React.useState([]);
  const [views, setViews] = React.useState([0]); // Start with one view
  const [nextViewId, setNextViewId] = React.useState(1);

  const ptData = {ids};

  React.useEffect(() => {
    fetch("/index/")
      .then((res) => res.json())
      .then((data) => {
        console.log('Got index data', data);
        setIds(data.ids);
      });
  }, []);

  const addView = () => {
    setViews([...views, nextViewId]);
    setNextViewId(nextViewId + 1);
  };

  const deleteView = (id) => {
    setViews(views.filter(v => v !== id));
  };

  return (
    <PtCtx.Provider value={ptData}>
      <div>
        <h3>Embeddings Explorer</h3>
        <p>{ids.length} points loaded. Sample IDs: {ids.slice(0, 5).join(', ')}</p>
        <div className="views">
        {views.map(id => (
          <View key={id} id={id} onDelete={deleteView} />
        ))}
        </div>
        <button onClick={addView} className="add-view-button">Add View</button>
      </div>
    </PtCtx.Provider>
  );
};

ReactDOM.createRoot(document.getElementById("main")).render(<Main />);
