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
  const [mediapipeOn, setMediapipeOn] = useState(true);
  const [videomaeOn, setVideomaeOn] = useState(true);
  const [actionInterval, setActionInterval] = useState(1.0);

  const { status, frameUrl, alerts, handState, connect, disconnect, sendMessage } = useVideoStream(wsUrl);

  const isStreaming = status === "streaming" || status === "connecting";

  const handleStart = () => {
    setMediapipeOn(true);
    setVideomaeOn(true);
    setActionInterval(1.0);
    const url = `${API_WS}/stream/webcam?cam_index=${camIndex}`;
    setWsUrl(url);
  };

  const handleStop = () => {
    disconnect();
    setWsUrl(null);
  };

  const toggleMediapipe = () => {
    const next = !mediapipeOn;
    setMediapipeOn(next);
    sendMessage({ type: "toggle", target: "mediapipe", enabled: next });
  };

  const toggleVideomae = () => {
    const next = !videomaeOn;
    setVideomaeOn(next);
    sendMessage({ type: "toggle", target: "videomae", enabled: next });
  };

  return (
    <VideoPanel label="live webcam" frameUrl={frameUrl} alerts={alerts} status={status}
      confirmedHandState={handState.confirmed}
      pendingCount={handState.pendingCount}
      debounceThreshold={3}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {isStreaming && (
          <>
            <ToggleButton label="MediaPipe" on={mediapipeOn} disabled={status !== "streaming"} onClick={toggleMediapipe} />
            <ToggleButton label="VideoMAE" on={videomaeOn} disabled={status !== "streaming"} onClick={toggleVideomae} />
            <select
              value={actionInterval}
              onChange={(e) => {
                const val = Number(e.target.value);
                setActionInterval(val);
                sendMessage({ type: "config", key: "action_interval", value: val });
              }}
              style={selectStyle}
              disabled={status !== "streaming"}
            >
              <option value={0.5}>Interval: 0.5s</option>
              <option value={1.0}>Interval: 1.0s</option>
              <option value={1.5}>Interval: 1.5s</option>
              <option value={2.0}>Interval: 2.0s</option>
            </select>
          </>
        )}
        {!isStreaming && (
          <select
            value={camIndex}
            onChange={(e) => setCamIndex(Number(e.target.value))}
            disabled={isStreaming}
            style={selectStyle}
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

function ToggleButton({ label, on, disabled, onClick }: { label: string; on: boolean; disabled?: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 9, fontWeight: 600, padding: "4px 10px",
        borderRadius: 12, letterSpacing: "0.04em",
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "all 0.2s ease",
        border: on 
          ? (disabled ? "1px solid rgba(0,255,100,0.15)" : "1px solid rgba(0,255,100,0.4)") 
          : "1px solid rgba(255,255,255,0.15)",
        background: on 
          ? (disabled ? "rgba(0,255,100,0.04)" : "rgba(0,255,100,0.12)") 
          : "rgba(255,255,255,0.04)",
        color: on 
          ? (disabled ? "rgba(0,255,100,0.3)" : "#4ade80") 
          : "rgba(255,255,255,0.35)",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <span style={{
        display: "inline-block", width: 6, height: 6, borderRadius: "50%",
        background: on ? "#4ade80" : "rgba(255,255,255,0.2)",
        marginRight: 5, verticalAlign: "middle",
        boxShadow: on && !disabled ? "0 0 6px #4ade80" : "none",
        transition: "all 0.2s ease",
      }} />
      {label}
    </button>
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

const selectStyle: React.CSSProperties = {
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 9, padding: "4px 8px", borderRadius: 4,
  border: "1px solid rgba(255,255,255,0.15)",
  background: "rgba(255,255,255,0.04)", color: "rgba(255,255,255,0.5)",
  cursor: "pointer", letterSpacing: "0.04em",
};

