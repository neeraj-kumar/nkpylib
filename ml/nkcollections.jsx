const STYLES = `

.labeled {
  border: 1px solid #888;
  padding: 5px;
  margin-bottom: 10px;
}

.objects, .labeled {
  display: flex;
  flex-wrap: wrap;
}

.object {
  border: 1px solid #ccc;
  padding: 5px;
  margin: 5px;
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

.image img {
  max-width: 200px;
}

.score {
  font-size: 0.8em;
  color: #555;
}
`;

const Obj = ({id, otype, url, md, togglePos, score, ...props}) => {
  console.log('Obj', id, otype, score, props);
  return (
    <div id={`id-${id}`} className={`object ${otype}`} onClick={() => togglePos(id)}>
      {otype === 'text' && (
        <div className="content">{md.text}</div>
      )}
      {otype === 'image' && (
        <div className="content">
          <img src={url} alt={`Image ${id}`} />
        </div>
      )}
      {score !== undefined && (
        <div className="score">Score: {score.toFixed(3)}</div>
      )}
    </div>
  );
}



const App = () => {
  const [rowById, setRowById] = React.useState({});
  const [curIds, setCurIds] = React.useState([]);
  const [scores, setScores] = React.useState({});
  const [pos, setPos] = React.useState([]);
  React.useEffect(() => {
    document.title = 'NK Collections';
    // insert styles
    const styleEl = document.createElement('style');
    styleEl.innerHTML = STYLES;
    document.head.appendChild(styleEl);
    // fetch events from the server
    fetch('/get/0-100?otypes=text,image')
      .then((response) => response.json())
      .then((data) => {
        console.log('got data', data);
        // use immer to update rowById
        setRowById(immer.produce(rowById, (draft) => {
          Object.entries(data.rows).forEach(([id, row]) => {
            draft[id] = row;
          });
          setCurIds(Object.keys(data.rows));
        }));
      });
  }, []);

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

  // function to call classification, whenever pos changes
  React.useEffect(() => {
    if (pos.length === 0) {
      return;
    }
    console.log('calling classify for pos', pos);
    fetch('/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({pos}),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log('got classify data', data);
        // update curIds and scores
        setCurIds(data.curIds);
        setScores(data.scores);
      });
  }, [pos]);

  const funcs = { togglePos };
  console.log('rowById', rowById, curIds, pos);

  return (
  <div>
    <h3>Collections</h3>
    <h4>Labeled</h4>
    <div className="labeled">
      {pos.map((id) => <Obj key={id} {...funcs} {...rowById[id]} />)}
    </div>
    <div className="objects">
      {curIds.map((id) => <Obj key={id} score={scores[id]} {...funcs} {...rowById[id]} />)}
    </div>
  </div>
  );
}

ReactDOM.createRoot(document.getElementById("main")).render(<App />);
