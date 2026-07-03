import { useState, useEffect, useRef } from "react";

function App() {
  const [status, setStatus] = useState("connecting");
  const [log, setLog] = useState([]);
  const socketRef = useRef(null);
  const canvasRef = useRef(null);
  const drawingRef = useRef(false);
  const lastSendRef = useRef(0);

  const addLog = (msg) =>
    setLog((prev) => [...prev, `${new Date().toLocaleTimeString()}  ${msg}`]);

  // ─── WebSocket ─────────────────────────────────────────────
  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8080");
    socketRef.current = socket;
    socket.binaryType = "arraybuffer"; // so the echoed binary frame arrives as ArrayBuffer

    socket.onopen = () => { setStatus("connected"); addLog("WebSocket connected"); };
    socket.onmessage = (event) => {
      // We send a binary frame, so the echo comes back as an ArrayBuffer.
      const bytes = new Uint8Array(event.data);
      addLog(`received: ${bytes.length} bytes`);
    };
    socket.onclose = () => { setStatus("disconnected"); addLog("WebSocket closed"); };
    socket.onerror = () => { setStatus("error"); addLog("WebSocket error"); };

    return () => socket.close();
  }, []);

  // ─── Canvas init: black background, white pen ──────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  // ─── Coordinate mapping (display px → canvas px) ───────────
  const getPos = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const startDraw = (e) => {
    drawingRef.current = true;
    const ctx = canvasRef.current.getContext("2d");
    const { x, y } = getPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e) => {
    if (!drawingRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();

    // Live prediction: send at most once every 100ms while drawing.
    const now = Date.now();
    if (now - lastSendRef.current > 100) {
      lastSendRef.current = now;
      sendCanvas();
    }
  };

  const endDraw = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    sendCanvas(); // final frame when the stroke ends
  };

  // ─── Extract 28×28 grayscale → 784 raw bytes → send ────────
  const sendCanvas = () => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      addLog("cannot send — socket not open");
      return;
    }
    const canvas = canvasRef.current;

    // Downscale the full canvas onto a 28×28 offscreen canvas.
    const small = document.createElement("canvas");
    small.width = 28;
    small.height = 28;
    const sctx = small.getContext("2d");
    sctx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);

    // getImageData returns RGBA: 28*28*4 = 3136 values.
    const rgba = sctx.getImageData(0, 0, 28, 28).data;
    const bytes = new Uint8Array(784);
    for (let i = 0; i < 784; i++) {
      // Grayscale (white-on-black), so R = G = B; take the red channel at index i*4.
      bytes[i] = rgba[i * 4];
    }

    socket.send(bytes); // 784-byte binary payload
    addLog(`sent: ${bytes.length} bytes`);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  const statusColor = {
    connecting: "bg-yellow-400",
    connected: "bg-green-500",
    disconnected: "bg-gray-400",
    error: "bg-red-500",
  }[status];

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center justify-center gap-6 p-8">
      <h1 className="text-3xl font-bold">Digit Predictor</h1>

      <div className="flex items-center gap-3">
        <span className={`w-3 h-3 rounded-full ${statusColor}`} />
        <span className="text-lg capitalize">{status}</span>
      </div>

      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        onPointerDown={startDraw}
        onPointerMove={draw}
        onPointerUp={endDraw}
        onPointerLeave={endDraw}
        className="rounded-lg border border-gray-600 touch-none cursor-crosshair"
        style={{ width: 280, height: 280 }}
      />

      <button
        onClick={clearCanvas}
        className="px-5 py-2 rounded-lg font-semibold bg-blue-600 hover:bg-blue-500 transition-colors"
      >
        Clear
      </button>

      <div className="w-full max-w-md bg-gray-800 rounded-lg p-4 font-mono text-sm h-40 overflow-y-auto">
        {log.length === 0 ? (
          <p className="text-gray-500">Waiting for events…</p>
        ) : (
          log.map((line, i) => (
            <div key={i} className="text-gray-300">{line}</div>
          ))
        )}
      </div>
    </div>
  );
}

export default App;