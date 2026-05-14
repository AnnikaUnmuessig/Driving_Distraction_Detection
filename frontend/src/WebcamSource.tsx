import React, { useState } from "react";
import { VideoPanel } from "./VideoPanel";
import { useVideoStream } from "./useVideoStream";

const API_WS = "ws://localhost:8000";

const CAMERAS = [
  { label: "Default (index 0)", index: 0 },
  { label: "External USB (index 1)", index: 1 },
  { label: "Virtual cam (index 2)", index: 2 },
];

export function WebcamSource() {
  const [camIndex, setCamIndex] = useState(0);
  const [wsUrl, setWsUrl] = useState<string | null>(null);

  const { status, frameUrl, alerts, handState, connect, disconnect } = useVideoStream(wsUrl);

  const isStreaming = status === "streaming" || status === "connecting";

  const handleStart = () => {
    const url = `${API_WS}/stream/webcam?cam_index=${camIndex}`;
    setWsUrl(url); // WebSocket will connect automatically
  };

  const handleStop = () => {
    disconnect();
    setWsUrl(null);
  };

  return (
    <VideoPanel label="live webcam" frameUrl={frameUrl} alerts={alerts} status={status}
      confirmedHandState={handState.confirmed}
      pendingCount={handState.pendingCount}
      debounceThreshold={3}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {!isStreaming && (
          <select
            value={camIndex}
            onChange={(e) => setCamIndex(Number(e.target.value))}
            disabled={isStreaming}
            style={{
              fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
              fontSize: 9, padding: "4px 8px", borderRadius: 4,
              border: "1px solid rgba(255,255,255,0.15)",
              background: "rgba(255,255,255,0.04)", color: "rgba(255,255,255,0.5)",
              cursor: "pointer", letterSpacing: "0.04em",
            }}
          >
            {CAMERAS.map((c) => (
              <option key={c.index} value={c.index} style={{ background: "#1a1a18" }}>
                {c.label}
              </option>
            ))}
          </select>
        )}

        {!isStreaming ? (
          <button onClick={handleStart} style={primaryBtnStyle}>start</button>
        ) : (
          <button onClick={handleStop} style={stopBtnStyle}>stop</button>
        )}
      </div>
    </VideoPanel>
  );
}

const primaryBtnStyle: React.CSSProperties = {
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 10, fontWeight: 600, padding: "5px 12px",
  borderRadius: 4, border: "none", letterSpacing: "0.06em",
  background: "#185FA5", color: "#fff", cursor: "pointer",
};

const stopBtnStyle: React.CSSProperties = {
  ...primaryBtnStyle,
  background: "transparent",
  border: "1px solid rgba(163,45,45,0.6)",
  color: "#f87171",
};
