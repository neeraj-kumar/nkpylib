// Note that this is imported in the main HTML file directly, so you have to use React.xxx for everything

// Create a context for sharing point data
const PointContext = React.createContext(null);

// A view is one way to look at our data - could be a scatter plot, list, etc
const View = ({id, onDelete}) => {
  const ptData = React.useContext(PointContext);
  return (
    <div className="view" style={{border: '1px solid #ccc', margin: '10px', padding: '10px'}}>
      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '10px'}}>
        <h4>View {id}</h4>
        <button onClick={() => onDelete(id)}>×</button>
      </div>
      <div>View content here (has access to {ptData.ids.length} points)</div>
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
    <PointContext.Provider value={ptData}>
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
    </PointContext.Provider>
  );
};

ReactDOM.createRoot(document.getElementById("main")).render(<Main />);
