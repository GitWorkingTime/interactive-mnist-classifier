import { useState, useEffect, useRef } from "react";

function App() {
  const [status, setStatus] = useState("connecting");
  const [log, setLog] = useState([]);
  const socketRef = useRef(null);

  // Add a timestamped line to the on-screen log
  const addLog = (msg) =>
    setLog((prev) => [...prev, `${new Date().toLocaleTimeString()}  ${msg}`]);

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8080");
    socketRef.current = socket;

    socket.onopen = () => {
      setStatus("connected");
      addLog("WebSocket connected");
    };
    socket.onmessage = (event) => {
      addLog(`received: ${event.data}`);
    };
    socket.onclose = () => {
      setStatus("disconnected");
      addLog("WebSocket closed");
    };
    socket.onerror = () => {
      setStatus("error");
      addLog("WebSocket error");
    };

    // Clean up on unmount so we don't leak connections during dev hot-reload
    return () => socket.close();
  }, []);

  const statusColor = {
    connecting: "bg-yellow-400",
    connected: "bg-green-500",
    disconnected: "bg-gray-400",
    error: "bg-red-500",
  }[status];

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center justify-center gap-6 p-8">
      <h1 className="text-3xl font-bold">Digit Predictor — Connection Test</h1>

      <div className="flex items-center gap-3">
        <span className={`w-3 h-3 rounded-full ${statusColor}`} />
        <span className="text-lg capitalize">{status}</span>
      </div>

      <div className="w-full max-w-md bg-gray-800 rounded-lg p-4 font-mono text-sm h-64 overflow-y-auto">
        {log.length === 0 ? (
          <p className="text-gray-500">Waiting for events…</p>
        ) : (
          log.map((line, i) => (
            <div key={i} className="text-gray-300">
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default App;