import React from "react";
import type { AlertEvent } from "./useVideoStream";

type Props = {
  label: string;
  frameUrl: string | null;
  alerts: AlertEvent[];
  status: string;
  progress?: number;
  confirmedHandState?: { left: boolean; right: boolean };
  pendingCount?: number;
  debounceThreshold?: number;
  children?: React.ReactNode;
};

const SEV_COLORS: Record<string, { bg: string; fg: string; dot: string }> = {
  "mid-heavy": { bg: "rgba(163,45,45,0.18)", fg: "#f87171", dot: "#ef4444" },
  mid:         { bg: "rgba(180,120,0,0.18)",  fg: "#fbbf24", dot: "#f59e0b" },
  "light-mid": { bg: "rgba(30,160,80,0.18)",  fg: "#4ade80", dot: "#22c55e" },
};

export function VideoPanel({
  label, frameUrl, alerts, status, progress,
  confirmedHandState, pendingCount = 0, debounceThreshold = 3,
  children,
}: Props) {
  const left  = confirmedHandState?.left  ?? true;
  const right = confirmedHandState?.right ?? true;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, height: "100%" }}>
      {/* ── header row ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <StatusDot status={status} />
        <span style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace", fontSize: 11,
          fontWeight: 600, letterSpacing: "0.08em", color: "#a0e4b0", textTransform: "uppercase" }}>
          {label}
        </span>
        <StatusPill status={status} />
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {children}
        </div>
      </div>

      {/* ── video frame ── */}
      <div style={{
        background: "#050508",
        borderRadius: 6,
        overflow: "hidden",
        aspectRatio: "16/9",
        display: "flex", alignItems: "center", justifyContent: "center",
        border: "1px solid rgba(0,255,100,0.12)",
        position: "relative",
        boxShadow: "0 0 0 1px rgba(0,0,0,0.6), inset 0 0 40px rgba(0,0,0,0.5)",
      }}>
        {frameUrl ? (
          <img src={frameUrl} alt="Annotated frame"
            style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        ) : (
          <EmptyState status={status} />
        )}

        {/* scanline overlay */}
        <div style={{
          position: "absolute", inset: 0, pointerEvents: "none",
          backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px)",
        }} />

        {/* corner brackets */}
        {["tl","tr","bl","br"].map(c => <CornerBracket key={c} corner={c as any} />)}
      </div>

      {/* ── debounce state bars ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        <DebounceBar label="LEFT" on={left} pending={pendingCount} threshold={debounceThreshold} />
        <DebounceBar label="RIGHT" on={right} pending={pendingCount} threshold={debounceThreshold} />
      </div>

      {/* ── processing progress ── */}
      {status === "processing" && progress !== undefined && (
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
            <span style={monoStyle}>PROCESSING</span>
            <span style={monoStyle}>{progress}%</span>
          </div>
          <div style={{ height: 2, background: "rgba(255,255,255,0.08)", borderRadius: 2 }}>
            <div style={{ height: "100%", width: `${progress}%`, borderRadius: 2,
              background: "linear-gradient(90deg, #1a6b3a, #4ade80)",
              transition: "width 0.4s ease", boxShadow: "0 0 6px #4ade80" }} />
          </div>
        </div>
      )}

      {/* ── alert log ── */}
      <div style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 10, display: "flex", flexDirection: "column", gap: 3,
        maxHeight: 140, overflowY: "auto", flex: 1 }}>
        {alerts.length === 0 ? (
          <div style={{ color: "rgba(255,255,255,0.2)", padding: "4px 0" }}>— no alerts —</div>
        ) : alerts.map((a, i) => {
          const sev = SEV_COLORS[a.severity ?? ""] ?? { bg: "rgba(255,255,255,0.05)", fg: "#aaa", dot: "#888" };
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 6,
              padding: "4px 8px", borderRadius: 4, background: sev.bg,
              border: `1px solid ${sev.dot}33` }}>
              <span style={{ width: 5, height: 5, borderRadius: "50%",
                background: sev.dot, flexShrink: 0, boxShadow: `0 0 4px ${sev.dot}` }} />
              <span style={{ color: sev.fg, flex: 1, letterSpacing: "0.04em" }}>
                {a.distraction_type ?? a.message ?? "event"}
              </span>
              <span style={{ color: "rgba(255,255,255,0.25)", fontSize: 9 }}>
                {a.severity ?? "alert"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DebounceBar({ label, on, pending, threshold }:
  { label: string; on: boolean; pending: number; threshold: number }) {
  const fill = on ? threshold : pending;
  return (
    <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 4, padding: "5px 8px",
      border: "1px solid rgba(255,255,255,0.07)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ ...monoStyle, color: on ? "#4ade80" : "#f87171" }}>{label}</span>
        <span style={monoStyle}>{fill}/{threshold}</span>
      </div>
      <div style={{ display: "flex", gap: 3 }}>
        {Array.from({ length: threshold }).map((_, i) => (
          <div key={i} style={{
            flex: 1, height: 3, borderRadius: 2,
            background: i < fill
              ? (on ? "#4ade80" : "#f87171")
              : "rgba(255,255,255,0.1)",
            boxShadow: i < fill ? `0 0 4px ${on ? "#4ade80" : "#f87171"}` : "none",
            transition: "background 0.2s",
          }} />
        ))}
      </div>
    </div>
  );
}

function StatusDot({ status }: { status: string }) {
  const colors: Record<string, string> = {
    streaming: "#22c55e", live: "#22c55e", processing: "#3b82f6",
    connecting: "#f59e0b", queued: "#3b82f6", done: "#22c55e", error: "#ef4444",
  };
  const c = colors[status] ?? "#555";
  return (
    <span style={{ width: 6, height: 6, borderRadius: "50%", background: c, flexShrink: 0,
      boxShadow: `0 0 6px ${c}`, animation: ["streaming","processing","connecting"].includes(status)
        ? "pulse 1.5s ease-in-out infinite" : "none" }} />
  );
}

function StatusPill({ status }: { status: string }) {
  const map: Record<string, { label: string; color: string }> = {
    idle:       { label: "IDLE",       color: "#555" },
    connecting: { label: "CONNECTING", color: "#f59e0b" },
    streaming:  { label: "LIVE",       color: "#22c55e" },
    processing: { label: "PROCESSING", color: "#3b82f6" },
    queued:     { label: "QUEUED",     color: "#3b82f6" },
    done:       { label: "DONE",       color: "#22c55e" },
    error:      { label: "ERROR",      color: "#ef4444" },
  };
  const s = map[status] ?? map.idle;
  return (
    <span style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      fontSize: 9, letterSpacing: "0.1em", color: s.color,
      padding: "2px 6px", border: `1px solid ${s.color}55`, borderRadius: 3 }}>
      {s.label}
    </span>
  );
}

function EmptyState({ status }: { status: string }) {
  const msgs: Record<string, string> = {
    idle: "waiting for source", connecting: "connecting…",
    queued: "pipeline queued", processing: "processing…",
    done: "complete", error: "stream error",
  };
  return (
    <div style={{ textAlign: "center", color: "rgba(255,255,255,0.15)",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace" }}>
      <div style={{ fontSize: 28, marginBottom: 8, opacity: 0.4 }}>⬡</div>
      <div style={{ fontSize: 11, letterSpacing: "0.1em" }}>{msgs[status] ?? "idle"}</div>
    </div>
  );
}

function CornerBracket({ corner }: { corner: "tl"|"tr"|"bl"|"br" }) {
  const size = 12;
  const t = corner.startsWith("t") ? 0 : undefined;
  const b = corner.startsWith("b") ? 0 : undefined;
  const l = corner.endsWith("l") ? 0 : undefined;
  const r = corner.endsWith("r") ? 0 : undefined;
  const bTop    = t !== undefined ? "2px solid rgba(0,255,100,0.4)" : "none";
  const bBottom = b !== undefined ? "2px solid rgba(0,255,100,0.4)" : "none";
  const bLeft   = l !== undefined ? "2px solid rgba(0,255,100,0.4)" : "none";
  const bRight  = r !== undefined ? "2px solid rgba(0,255,100,0.4)" : "none";
  return (
    <div style={{ position: "absolute", top: t, bottom: b, left: l, right: r,
      width: size, height: size, pointerEvents: "none",
      borderTop: bTop, borderBottom: bBottom, borderLeft: bLeft, borderRight: bRight }} />
  );
}

const monoStyle: React.CSSProperties = {
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 9, letterSpacing: "0.06em", color: "rgba(255,255,255,0.35)",
};
