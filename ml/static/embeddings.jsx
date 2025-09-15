// Note that this is imported in the main HTML file directly, so you have to use React.xxx for everything

// A view is one view into our data. We can create and delete them as needed.
const View = ({}) => {
  return (<div>View</div>);
}

const Main = () => {
  const [ids, setIds] = React.useState([]);
  const [views, setViews] = React.useState([]);
  React.useEffect(() => {
    fetch("/index/")
      .then((res) => res.json())
      .then((data) => {
        console.log('Got index data', data);
        setIds(data.ids);
      });
  }, []);

  return (<div>
    <h3>Embeddings Page</h3>
    <p>{ids.length} pts: {ids.slice(0, 5).join(', ')}</p>
  </div>);
};

ReactDOM.createRoot(document.getElementById("main")).render(<Main />);
