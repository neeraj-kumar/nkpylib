// Note that this is imported in the main HTML file directly, so you have to use React.xxx for everything

// Create a context for sharing point data
const PtCtx = React.createContext(null);
const { useState } = React;

const PtItem = ({id, idx, md, value}) => {
  if (!md) {
    md = {};
  }
  let desc = `${idx}: `;
  if (md.name) {
    desc += md.name;
  }
  if (value) {
    desc += ': ' + value.toFixed(3);
  }
  return (
    <div className="pt">
      <div>{id}</div>
      <div className="pt-label">{desc}</div>
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
  const visibleIds = React.useMemo(() => ptData.ids
      .slice(startIdx, startIdx + pageSize * gap)
      .filter((_, i) => i % gap === 0),
    [ptData.ids, startIdx, gap, pageSize]);

  // Fetch metadata for visible points
  React.useEffect(() => {
    ptData.getPtMd(visibleIds)
      .then(changed => {
        if (changed) {
          console.log("Metadata updated for", visibleIds.length, "points");
        }
      });
  }, [visibleIds]);

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
          <PtItem key={id} id={id} idx={startIdx + idx * gap} md={ptData.ptMd[id]}  value={ptData.values[id]}/>
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
  const [values, setValues] = React.useState({});
  const [ptMd, setPtMd] = React.useState({}); // point metadata cache
  const [views, setViews] = React.useState([0]); // Start with one view
  const [nextViewId, setNextViewId] = React.useState(1);

  React.useEffect(() => {
    fetch("/index/")
      .then((res) => res.json())
      .then((data) => {
        console.log('Got index data', data);
        setIds(data.ids);
        setValues(data.values);
      });
  }, []);

  // gets point metadata for given points, returns promise
  const getPtMd = React.useCallback((ids) => {
    // skip those we already have md for
    const toGet = ids.filter(id => !ptMd[id]);
    if (toGet.length === 0) {
      return Promise.resolve(false); // nothing new to get
    }
    return fetch("/pt_md/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ids: toGet }),
      })
      .then((res) => res.json())
      .then((data) => {
        console.log("Got point metadata", data);
        setPtMd(prev => ({...prev, ...data.data}));
        return true; // indicate we got new data
      });
  }, [])

  const ptData = {ids, values, ptMd, getPtMd};

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
        <p>{ids.length} points loaded</p>
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
