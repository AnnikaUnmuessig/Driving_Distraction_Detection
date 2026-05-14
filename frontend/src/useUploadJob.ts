import { useCallback, useEffect, useRef, useState } from "react";

const API = "http://localhost:8000";

export type JobStatus = "idle" | "uploading" | "queued" | "processing" | "done" | "error";

export type Job = {
  job_id: string;
  status: JobStatus;
  progress: number;
  alert_count: number;
  output_path: string | null;
  error: string | null;
};

export function useUploadJob() {
  const [job, setJob] = useState<Job | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const upload = useCallback(async (file: File) => {
    setJob(null);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append("file", file);

    // XHR for upload progress tracking
    const jobId = await new Promise<string>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API}/upload`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) setUploadProgress(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        if (xhr.status === 200) resolve(JSON.parse(xhr.responseText).job_id);
        else reject(new Error(xhr.responseText));
      };
      xhr.onerror = () => reject(new Error("Upload failed"));
      xhr.send(formData);
    });

    setJob({ job_id: jobId, status: "queued", progress: 0, alert_count: 0, output_path: null, error: null });

    // start polling
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API}/jobs/${jobId}`);
        const data: Job = await res.json();
        setJob(data);
        if (data.status === "done" || data.status === "error") {
          clearInterval(pollRef.current!);
        }
      } catch {
        clearInterval(pollRef.current!);
      }
    }, 1000);
  }, []);

  const reset = useCallback(() => {
    clearInterval(pollRef.current!);
    setJob(null);
    setUploadProgress(0);
  }, []);

  useEffect(() => () => { clearInterval(pollRef.current!); }, []);

  return { job, uploadProgress, upload, reset };
}
