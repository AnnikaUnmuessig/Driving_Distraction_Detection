import { UploadSource } from "./UploadSource";
import { WebcamSource } from "./WebcamSource";
import { useState } from "react";

export default function App() {
  const [mode, setMode] = useState<"upload" | "webcam" | null>(null);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#09090d",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      padding: "20px",
      backgroundImage: "radial-gradient(ellipse at 20% 0%, rgba(0,60,30,0.15) 0%, transparent 60%)",
    }}>
      {/* Google Font import */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.04); }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 2px; }
      `}</style>

      {/* top bar */}
      <div style={{
        display: "flex", alignItems: "center", gap: 12, marginBottom: 20,
        paddingBottom: 14, borderBottom: "1px solid rgba(0,255,100,0.1)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#22c55e",
            boxShadow: "0 0 8px #22c55e", animation: "pulse 2s ease-in-out infinite",
            display: "inline-block" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#22c55e", letterSpacing: "0.15em" }}>
            DRIVER MONITOR
          </span>
          <span style={{ fontSize: 9, color: "rgba(255,255,255,0.25)", letterSpacing: "0.08em",
            borderLeft: "1px solid rgba(255,255,255,0.1)", paddingLeft: 8 }}>
            debug monitor
          </span>
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          <ConfigChip label="DEBOUNCE 3/15" />
          <ConfigChip label="HANDS 5s/2s" />
          <ConfigChip label="16F WINDOW" />
        </div>
      </div>

      {/* mode selection */}
      {!mode && (
        <div style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 20,
          maxWidth: 600,
          margin: "0 auto",
        }}>
          <h2 style={{ color: "#a0e4b0", fontSize: 18, margin: 0, textAlign: "center" }}>
            Choose Monitoring Mode
          </h2>
          <div style={{ display: "flex", gap: 16 }}>
            <ModeButton
              icon="📹"
              title="Live Webcam"
              description="Monitor live camera feed"
              onClick={() => setMode("webcam")}
            />
            <ModeButton
              icon="📁"
              title="Upload Video"
              description="Process recorded video file"
              onClick={() => setMode("upload")}
            />
          </div>
        </div>
      )}

      {/* selected mode */}
      {mode === "webcam" && (
        <div style={{ maxWidth: 1280, margin: "0 auto" }}>
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
            <button
              onClick={() => setMode(null)}
              style={{
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.1)",
                color: "#a0e4b0",
                padding: "8px 16px",
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 12,
                fontFamily: "inherit",
              }}
            >
              ← Back to Selection
            </button>
          </div>
          <PanelCard>
            <WebcamSource />
          </PanelCard>
        </div>
      )}

      {mode === "upload" && (
        <div style={{ maxWidth: 1280, margin: "0 auto" }}>
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
            <button
              onClick={() => setMode(null)}
              style={{
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.1)",
                color: "#a0e4b0",
                padding: "8px 16px",
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 12,
                fontFamily: "inherit",
              }}
            >
              ← Back to Selection
            </button>
          </div>
          <PanelCard>
            <UploadSource />
          </PanelCard>
        </div>
      )}

      {/* footer */}
      <div style={{ marginTop: 16, textAlign: "center", fontSize: 9,
        color: "rgba(255,255,255,0.15)", letterSpacing: "0.1em" }}>
        HANDS-OFF THRESHOLD 1.0s · DEBOUNCE 3 FRAMES · ACTION WINDOW 16F
      </div>
    </div>
  );
}

function ModeButton({ icon, title, description, onClick }: {
  icon: string;
  title: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(0,255,100,0.2)",
        borderRadius: 12,
        padding: "24px",
        cursor: "pointer",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 12,
        minWidth: 200,
        transition: "all 0.2s ease",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = "rgba(0,255,100,0.4)";
        e.currentTarget.style.background = "rgba(255,255,255,0.06)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "rgba(0,255,100,0.2)";
        e.currentTarget.style.background = "rgba(255,255,255,0.03)";
      }}
    >
      <div style={{ fontSize: 32 }}>{icon}</div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 16, fontWeight: 600, color: "#a0e4b0", marginBottom: 4 }}>
          {title}
        </div>
        <div style={{ fontSize: 12, color: "rgba(255,255,255,0.6)" }}>
          {description}
        </div>
      </div>
    </button>
  );
}

function ConfigChip({ label }: { label: string }) {
  return (
    <span style={{
      fontSize: 9, fontWeight: 500, padding: "4px 8px",
      borderRadius: 12, background: "rgba(0,255,100,0.1)",
      color: "#4ade80", border: "1px solid rgba(0,255,100,0.2)",
      letterSpacing: "0.02em",
    }}>
      {label}
    </span>
  );
}

function PanelCard({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      border: "1px solid rgba(0,255,100,0.1)",
      borderRadius: 8,
      padding: "14px 16px",
      backdropFilter: "blur(4px)",
    }}>
      {children}
    </div>
  );
}
