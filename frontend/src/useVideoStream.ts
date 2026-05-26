import { useCallback, useEffect, useRef, useState } from "react";

export type AlertEvent = {
  type: "alert" | "done" | "error" | "hand_state";
  distraction_type?: string;
  severity?: string;
  message?: string;
  confirmed?: { left: boolean; right: boolean };
  pending_count?: number;
};

type StreamStatus = "idle" | "connecting" | "streaming" | "done" | "error";

export function useVideoStream(url: string | null) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  const [handState, setHandState] = useState<{ confirmed: { left: boolean; right: boolean }; pendingCount: number }>({
    confirmed: { left: true, right: true },
    pendingCount: 0
  });
  const prevBlobUrl = useRef<string | null>(null);

  const connect = useCallback(() => {
    if (!url || wsRef.current) return;
    setStatus("connecting");
    setAlerts([]);

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setStatus("streaming");

    ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        const event: AlertEvent = JSON.parse(e.data);
        if (event.type === "done") {
          setStatus("done");
        } else if (event.type === "error") {
          setStatus("error");
        } else if (event.type === "hand_state") {
          setHandState({
            confirmed: event.confirmed || { left: true, right: true },
            pendingCount: event.pending_count || 0
          });
        } else {
          setAlerts((prev) => [event, ...prev].slice(0, 50));
        }
        return;
      }

      // binary = JPEG frame
      if (prevBlobUrl.current) URL.revokeObjectURL(prevBlobUrl.current);
      const blob = new Blob([e.data], { type: "image/jpeg" });
      const blobUrl = URL.createObjectURL(blob);
      prevBlobUrl.current = blobUrl;
      setFrameUrl(blobUrl);
    };

    ws.onerror = () => setStatus("error");
    ws.onclose = () => {
      wsRef.current = null;
      if (status !== "done") setStatus("idle");
    };
  }, [url]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setStatus("idle");
    setFrameUrl(null);
  }, []);

  const sendMessage = useCallback((msg: object) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  useEffect(() => {
    if (url && !wsRef.current) {
      connect();
    } else if (!url && wsRef.current) {
      disconnect();
    }
  }, [url, connect, disconnect]);

  useEffect(() => () => { wsRef.current?.close(); }, []);

  return { status, frameUrl, alerts, handState, connect, disconnect, sendMessage };
}
