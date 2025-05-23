const MAIN_FIELDS = ['idx', '_name', 'sessionId', '_human_ts', '_ts', '_copy_of',];
const COLORED_FIELDS = ['_name', '_loc', '_ts', '_id', 'sessionId', '_copy_of'];

const RestField = ({data, ...props}) => {
  const style = {
    maxHeight: '100px',
    overflowY: 'auto',
  }
  return <div style={style}>{JSON.stringify(data.rest, null, ' ')}</div>
}

/* A field which includes a small rect colored by the hash of the contents */
const ColoredField = ({data, ...props}) => {
  const field = props.column.colId;
  const content = data[field];
  if (!content) return null;
  const hash = hashString(''+content);
  const color = numberToColor(hash);
  //console.log('hashed', field, content, hash, color);
  // style for the small rect
  const style = {
    backgroundColor: color,
    marginRight: 5,
    borderRadius: 5,
    width: 10,
    height: 10,
    display: 'inline-block',
  };
  return (
    <div>
      <div style={style}></div>
      <span>{content}</span>
    </div>
  );
}

const StateLogger = () => {
  const [gridApi, setGridApi] = React.useState(null);
  const [cols, setCols] = React.useState([]);
  const [length, setLength] = React.useState(0);
  const [log, setLog] = React.useState([]);
  React.useEffect(() => {
    document.title = 'State Logger';
    // fetch events from the server
    fetch('/get/0-100000')
      .then((response) => response.json())
      .then((data) => {
        // data.items is a dict from idx to item -> convert to array, sorted by idx
        const items = Object.keys(data.items).map((key) => {
          const item = {...data.items[key]};
          item.rest = {};
          // move all non-main fields to rest
          Object.keys(item).forEach((k) => {
            if (!MAIN_FIELDS.includes(k) && k !== 'rest') {
              item.rest[k] = item[k];
              delete item[k];
            }
          });
          item.idx = parseInt(key, 10);
          return item;
        });
        items.sort((a, b) => a.idx - b.idx);
        setLog(items);
        setLength(data.length);
      });
  }, []);

  // when data changes, update the columns
  React.useEffect(() => {
    const columns = MAIN_FIELDS.map((field) => {
      const ret = {field};
      if (COLORED_FIELDS.includes(field)) {
        ret.cellRenderer = ColoredField;
      }
      if (field === 'idx') {
        ret.width = 90;
      }
      return ret;
    });
    columns.push({field: 'rest', flex: 1, cellRenderer: RestField});
    setCols(columns);
  }, [log]);

  // AG Grid will use this to set up the grid once rendered
  const onGridReady = (params) => {
    setGridApi(params.api);
  };

  const defaultColDef = {
    sortable: true,
    filter: true,
    autoHeight: true,
    wrapText: true,
    resizable: true,
    suppressMovable: true,
  };

  if (!length) {
    return (<div>
      <h3>Loading...</h3>
    </div>);
  }

  console.log('got cols', cols, log);

  return (
  <div>
    <h3>State Logger</h3>
    <p>Length: {length}</p>
    <p>Log:</p>

    <div className="ag-theme-alpine main-grid" style={{height: '80vh'}}>
      <AgGridReact.AgGridReact
        rowData={log}
        columnDefs={cols}
        defaultColDef={defaultColDef}
        onGridReady={onGridReady}
        getRowId={(params) => params.data._ts}
        pagination={true}
        paginationPageSize={100}
        //isExternalFilterPresent={() => true}
        //doesExternalFilterPass={filterRows}
      />
    </div>
  </div>
  );
}

ReactDOM.createRoot(document.getElementById("main")).render(<StateLogger />);
