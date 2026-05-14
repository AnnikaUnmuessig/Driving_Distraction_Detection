import React, { useRef, useState } from "react";
import { VideoPanel } from "./VideoPanel";
import { useUploadJob } from "./useUploadJob";
import { useVideoStream } from "./useVideoStream";

const API_WS = "ws://localhost:8000";
const API = "http://localhost:8000";

export function UploadSource() {
  const { job, uploadProgress, upload, reset } = useUploadJob();
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const wsUrl = job ? `${API_WS}/stream/upload/${job.job_id}` : null;
  const { status: streamStatus, frameUrl, alerts, handState, connect, disconnect } = useVideoStream(wsUrl);

  const handleFile = (f: File) => {
    setFile(f);
    reset();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const handleStart = async () => {
    if (!file) return;
    await upload(file); // Upload first, get job_id - WebSocket will connect automatically
  };

  const handleReset = () => {
    disconnect();
    reset();
    setFile(null);
  };

  const isRunning = job && (job.status === "queued" || job.status === "processing");
  const isDone = job?.status === "done";
  const displayStatus = job?.status ?? streamStatus;

  return (
    <div data-upload-container>
      <VideoPanel
      label="video file"
      frameUrl={frameUrl}
      alerts={alerts}
      status={displayStatus}
      confirmedHandState={handState.confirmed}
      pendingCount={handState.pendingCount}
      debounceThreshold={3}
      progress={job?.progress}
    >
      {/* ── controls slot ── */}
      <div style={{ display: "flex", gap: 6 }}>
        {isDone && job?.job_id && (
          <>
            <button
              onClick={() => {
                const video = document.createElement('video');
                video.src = `${API}/output/${job.job_id}`;
                video.controls = true;
                video.style.width = '100%';
                video.style.maxHeight = '400px';
                video.style.borderRadius = '6px';
                video.style.marginTop = '16px';

                // Replace the current content with the video
                const container = document.querySelector('[data-upload-container]');
                if (container) {
                  // Clear existing video if any
                  const existingVideo = container.querySelector('video');
                  if (existingVideo) {
                    existingVideo.remove();
                  }
                  container.appendChild(video);
                  video.play();
                }
              }}
              style={secondaryBtnStyle}
            >
              ▶ replay
            </button>
            <a
              href={`${API}/output/${job.job_id}`}
              download
              style={primaryBtnStyle}
            >
              ↓ download
            </a>
          </>
        )}
        {(file || job) && (
          <button onClick={handleReset} style={ghostBtnStyle}>clear</button>
        )}
      </div>

      {/* ── drop zone (shown below VideoPanel via children ordering trick — rendered here as a sibling) ── */}
      {/* We render extra UI after the VideoPanel by using a wrapper */}
      <ExtraUI
        file={file} job={job} isRunning={!!isRunning} isDone={!!isDone}
        dragging={dragging} uploadProgress={uploadProgress}
        inputRef={inputRef}
        onDragOver={(e: React.DragEvent) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onFileChange={(e: React.ChangeEvent<HTMLInputElement>) => e.target.files?.[0] && handleFile(e.target.files[0])}
        onBrowseClick={() => inputRef.current?.click()}
        onStart={handleStart}
      />
    </VideoPanel>
    </div>
  );
}

// Extra UI rendered as children of VideoPanel's "children" slot wraps the controls row,
// but we also need the drop zone below the video. We pass it all as a fragment.
function ExtraUI({
  file, job, isRunning, isDone, dragging, uploadProgress,
  inputRef, onDragOver, onDragLeave, onDrop, onFileChange, onBrowseClick, onStart,
}: any) {
  return (
    <>
      {/* drop zone */}
      {!file && (
        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={onBrowseClick}
          style={{
            marginTop: 10,
            border: `1px dashed ${dragging ? "#4ade80" : "rgba(255,255,255,0.15)"}`,
            borderRadius: 6,
            padding: "22px 16px",
            display: "flex", flexDirection: "column", alignItems: "center", gap: 5,
            cursor: "pointer",
            background: dragging ? "rgba(0,255,100,0.04)" : "rgba(255,255,255,0.02)",
            transition: "all 0.15s",
          }}
        >
          <span style={{ fontSize: 18, color: "rgba(255,255,255,0.2)" }}>↑</span>
          <span style={{ fontSize: 10, color: "rgba(255,255,255,0.35)", letterSpacing: "0.08em" }}>
            DROP VIDEO OR CLICK TO BROWSE
          </span>
          <span style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: "0.06em" }}>
            MP4 · AVI · MOV
          </span>
          <input ref={inputRef} type="file" accept="video/*" style={{ display: "none" }}
            onChange={onFileChange} />
        </div>
      )}

      {/* file loaded row */}
      {file && !isRunning && !isDone && (
        <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 8,
          background: "rgba(255,255,255,0.04)", borderRadius: 5,
          padding: "8px 10px", border: "1px solid rgba(255,255,255,0.08)" }}>
          <span style={{ fontSize: 14, color: "#4ade80" }}>▶</span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 10, color: "rgba(255,255,255,0.7)", letterSpacing: "0.04em",
              whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{file.name}</div>
            <div style={{ fontSize: 9, color: "rgba(255,255,255,0.3)", marginTop: 2 }}>
              {(file.size / 1_000_000).toFixed(1)} MB
            </div>
          </div>
          <button onClick={onStart} style={primaryBtnStyle}>run pipeline</button>
        </div>
      )}

      {/* upload progress */}
      {job?.status === "queued" && uploadProgress < 100 && (
        <MiniProgress label="UPLOADING" value={uploadProgress} color="#3b82f6" />
      )}

      {/* done stats */}
      {isDone && (
        <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
          <StatChip label="ALERTS" value={String(job.alert_count)} />
          <StatChip label="STATUS" value="COMPLETE" color="#4ade80" />
        </div>
      )}
    </>
  );
}

function MiniProgress({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={monoSmall}>{label}</span>
        <span style={monoSmall}>{value}%</span>
      </div>
      <div style={{ height: 2, borderRadius: 2, background: "rgba(255,255,255,0.08)" }}>
        <div style={{ height: "100%", width: `${value}%`, background: color,
          borderRadius: 2, transition: "width 0.3s", boxShadow: `0 0 6px ${color}` }} />
      </div>
    </div>
  );
}

function StatChip({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 4,
      padding: "5px 10px", border: "1px solid rgba(255,255,255,0.08)" }}>
      <div style={{ fontSize: 8, color: "rgba(255,255,255,0.3)", marginBottom: 2, letterSpacing: "0.1em" }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 600, color: color ?? "rgba(255,255,255,0.7)" }}>{value}</div>
    </div>
  );
}

const monoSmall: React.CSSProperties = {
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 9, letterSpacing: "0.06em", color: "rgba(255,255,255,0.35)",
};

const primaryBtnStyle: React.CSSProperties = {
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 10, fontWeight: 600, padding: "5px 12px",
  borderRadius: 4, border: "none", letterSpacing: "0.06em",
  background: "#185FA5", color: "#fff", cursor: "pointer",
  textDecoration: "none", display: "inline-block",
};

const secondaryBtnStyle: React.CSSProperties = {
  ...primaryBtnStyle,
  background: "#2d5a3d",
  border: "1px solid #4ade80",
};

const ghostBtnStyle: React.CSSProperties = {
  ...primaryBtnStyle,
  background: "transparent",
  border: "1px solid rgba(255,255,255,0.15)",
  color: "rgba(255,255,255,0.5)",
};
