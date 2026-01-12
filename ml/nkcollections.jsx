const STYLES = `

.labeled {
  border: 1px solid #888;
  padding: 5px;
  margin-bottom: 10px;
}

.objects, .labeled, .controls {
  display: flex;
  flex-wrap: wrap;
}

.control {
  margin-right: 10px;
}

.object {
  border: 1px solid #ccc;
  padding: 5px;
  margin: 5px;
  text-align: center;
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

.object img {
  max-width: 200px;
}

.score {
  font-size: 0.8em;
  color: #555;
}
`;

const Obj = ({id, otype, url, md, togglePos, score, ...props}) => {
  //console.log('Obj', id, otype, score, props);
  return (
    <div id={`id-${id}`} className={`object ${otype}`} onClick={() => togglePos(id)}>
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
      <p className="score">ID: {id}</p>
      {score !== undefined && (
        <div className="score">Score: {score.toFixed(3)}</div>
      )}
    </div>
  );
}

const Controls = ({allOtypes, curOtypes, setCurOtypes, setCurIds, filterStr, setFilterStr, searchStr, setSearchStr, ...props}) => {
  return (
    <div className="controls">
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
      <div className="control text-fields">
        <input
          type="text"
          placeholder="Filter..."
          value={filterStr}
          onChange={(e) => setFilterStr(e.target.value)}
          style={{marginRight: '10px'}}
        />
        <input
          type="text"
          placeholder="Search..."
          value={searchStr}
          onChange={(e) => setSearchStr(e.target.value)}
          style={{marginRight: '10px'}}
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
    </div>
  );
}


const App = () => {
  const [rowById, setRowById] = React.useState({});
  const [allOtypes, setAllOtypes] = React.useState([]);
  const [curOtypes, setCurOtypes] = React.useState(['image', 'text', 'link']);
  const [curIds, setCurIds] = React.useState([]);
  const [scores, setScores] = React.useState({});
  const [pos, setPos] = React.useState([]);
  const [filterStr, setFilterStr] = React.useState('');
  const [searchStr, setSearchStr] = React.useState('');
  React.useEffect(() => {
    document.title = 'NK Collections';
    // insert styles
    const styleEl = document.createElement('style');
    styleEl.innerHTML = STYLES;
    document.head.appendChild(styleEl);
  }, []);

  // fetch data when otypes changes
  React.useEffect(() => {
    // fetch objects from the server
    fetch(`/get/0-1000?otypes=${curOtypes.join(',')}`)
      .then((response) => response.json())
      .then((data) => {
        console.log('got data', data);
        // use immer to update rowById
        setRowById(immer.produce(rowById, (draft) => {
          Object.entries(data.rows).forEach(([id, row]) => {
            draft[id] = row;
          });
        }));
        setCurIds(Object.keys(data.rows));
        setAllOtypes(data.allOtypes);
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
      // reset curIds to all ids and scores to empty
      setCurIds(Object.keys(rowById));
      setScores({});
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
        if (data.curIds && data.scores){
          setCurIds(data.curIds);
          setScores(data.scores);
        }
      });
  }, [pos]);

  const funcs = {allOtypes, curOtypes, togglePos, setCurOtypes, setCurIds, filterStr, setFilterStr, searchStr, setSearchStr};
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
    <div className="objects">
      {ids.map((id) => <Obj key={id} score={scores[id]} {...funcs} {...rowById[id]} />)}
    </div>
  </div>
  );
}

ReactDOM.createRoot(document.getElementById("main")).render(<App />);
