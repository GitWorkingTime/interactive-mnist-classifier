import { useState, useEffect, useRef } from "react";

function App() {
  const [status, setStatus] = useState("connecting");
  const [log, setLog] = useState([]);
  const [probs, setProbs] = useState(Array(10).fill(0));
  const socketRef = useRef(null);
  const canvasRef = useRef(null);
  const drawingRef = useRef(false);
  const lastSendRef = useRef(0);
  const logRef = useRef(null);

  // Auto-scroll the debug log to the bottom on every new entry.
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [log]);

  const addLog = (msg) =>
    setLog((prev) => [...prev, `${new Date().toLocaleTimeString()}  ${msg}`]);

  // ─── WebSocket ─────────────────────────────────────────────
  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8080");
    socketRef.current = socket;
    socket.binaryType = "arraybuffer";

    socket.onopen = () => { setStatus("connected"); addLog("WebSocket connected"); };
    socket.onmessage = (event) => {
      // Server sends a TEXT frame: "0.01,0.00,0.95,..." (10 values).
      if (typeof event.data === "string") {
        const parsed = event.data.split(",").map(Number);
        if (parsed.length === 10 && parsed.every((v) => !Number.isNaN(v))) {
          setProbs(parsed);
          // Log the predicted probabilities (2 decimals for readability).
          addLog("probs: " + parsed.map((v) => v.toFixed(2)).join(", "));
        } else {
          addLog(`received (unparsed): ${event.data}`);
        }
      }
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
    ctx.lineWidth = 10;
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

    const now = Date.now();
    if (now - lastSendRef.current > 100) {
      lastSendRef.current = now;
      sendCanvas();
    }
  };

  const endDraw = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    sendCanvas();
  };

  // ─── Extract 28×28 grayscale → 784 raw bytes → send ────────
  const sendCanvas = () => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      addLog("cannot send — socket not open");
      return;
    }
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;              // 280 x 280
    const src = ctx.getImageData(0, 0, W, H).data;          // RGBA, row-major

    // 1. Find the bounding box of drawn (non-black) pixels.
    let minX = W, minY = H, maxX = -1, maxY = -1;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = src[(y * W + x) * 4];                     // red channel
        if (v > 20) {                                       // threshold: ignore near-black
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    // Nothing drawn: send all zeros.
    if (maxX < 0) {
      socket.send(new Uint8Array(784));
      addLog("sent: 784 bytes (empty)");
      return;
    }

    const boxW = maxX - minX + 1;
    const boxH = maxY - minY + 1;

    // 2. Scale the digit to fit a 20x20 box, preserving aspect ratio.
    const scale = 20 / Math.max(boxW, boxH);
    const drawW = Math.round(boxW * scale);
    const drawH = Math.round(boxH * scale);

    // 3. Draw the cropped+scaled digit centered into a 28x28 field.
    const small = document.createElement("canvas");
    small.width = 28; small.height = 28;
    const sctx = small.getContext("2d");
    sctx.fillStyle = "black";
    sctx.fillRect(0, 0, 28, 28);

    const offsetX = Math.round((28 - drawW) / 2);
    const offsetY = Math.round((28 - drawH) / 2);
    sctx.drawImage(
      canvas,
      minX, minY, boxW, boxH,       // source crop (bounding box)
      offsetX, offsetY, drawW, drawH // destination, centered
    );

    // 4. Extract 784 grayscale bytes.
    const rgba = sctx.getImageData(0, 0, 28, 28).data;
    const bytes = new Uint8Array(784);
    for (let i = 0; i < 784; i++) bytes[i] = rgba[i * 4];

    socket.send(bytes);
    addLog(`sent: 784 bytes (bbox ${boxW}x${boxH})`);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setProbs(Array(10).fill(0));
    addLog("canvas cleared");
  };

  const statusColor = {
    connecting: "bg-yellow-400",
    connected: "bg-green-500",
    disconnected: "bg-gray-400",
    error: "bg-red-500",
  }[status];

  const maxVal = Math.max(...probs);
  const maxIdx = maxVal > 0 ? probs.indexOf(maxVal) : -1;

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center gap-6 p-8">
      <h1 className="text-3xl font-bold">Interactive MNIST Classifier</h1>

      <div className="flex items-center gap-3">
        <span className={`w-3 h-3 rounded-full ${statusColor}`} />
        <span className="text-lg capitalize">{status}</span>
      </div>

      {/* Three columns: log · canvas · bar graph */}
      <div className="flex flex-col lg:flex-row items-start justify-center gap-8 w-full max-w-5xl">

        {/* ─── Left: debug log ─── */}
        <div ref={logRef} className="w-full lg:w-72 bg-gray-800 rounded-lg p-4 font-mono text-xs h-72 overflow-y-auto shrink-0">
          {log.length === 0 ? (
            <p className="text-gray-500">Waiting for events…</p>
          ) : (
            log.map((line, i) => (
              <div key={i} className="text-gray-300 whitespace-pre-wrap break-all">{line}</div>
            ))
          )}
        </div>

        {/* ─── Middle: canvas ─── */}
        <div className="flex flex-col items-center gap-4 shrink-0">
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
        </div>

        {/* ─── Right: horizontal bars, digit column at left, bar grows right ─── */}
        <div className="w-full lg:w-72 bg-gray-800 rounded-lg p-4 shrink-0">
          <div className="flex flex-col gap-2">
            {probs.map((p, digit) => {
              const pct = Math.max(0, Math.min(1, p)) * 100; // clamp to [0,100]
              const isMax = digit === maxIdx;
              return (
                <div key={digit} className="flex items-center gap-2">
                  {/* digit label (the column of numbers) */}
                  <span
                    className={`w-4 text-sm font-mono text-right ${isMax ? "text-green-400 font-bold" : "text-gray-400"
                      }`}
                  >
                    {digit}
                  </span>
                  {/* fixed-width track; fill grows rightward, 0–100% axis */}
                  <div className="relative flex-1 h-5 bg-gray-700 rounded-sm">
                    <div
                      className={`absolute left-0 top-0 h-full rounded-sm transition-all duration-100 ${isMax ? "bg-green-500" : "bg-blue-500"
                        }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  {/* percentage label */}
                  <span className="w-10 text-xs text-gray-300 text-right">
                    {pct.toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;