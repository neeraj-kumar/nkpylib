const App = () => {
  React.useEffect(() => {
    document.title = 'NK Collections';
    // fetch events from the server
    fetch('/get/0-100000')
      .then((response) => response.json())
      .then((data) => {
        console.log('got data', data);
      });
  }, []);

  return (
  <div>
    <h3>Collections</h3>
  </div>
  );
}

ReactDOM.createRoot(document.getElementById("main")).render(<App />);
