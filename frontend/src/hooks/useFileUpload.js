import { useState, useCallback } from "react";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

export function useFileUpload() {
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [analysis, setAnalysis] = useState(null);

  const handleUpload = useCallback(async (files) => {
    if (files.length === 0) {
      setUploadStatus({ type: "error", message: "No files selected" });
      return false;
    }

    setUploading(true);
    setUploadStatus(null);

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f.file));

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload/files`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUploadStatus({
          type: "success",
          message: data.message,
          files: data.files,
        });

        // Analyze the first uploaded file
        if (data.files && data.files.length > 0) {
          await analyzeFile(data.files[0].filename);
        }

        return true;
      } else {
        setUploadStatus({
          type: "error",
          message: data.error || "Upload failed",
        });
        return false;
      }
    } catch (error) {
      setUploadStatus({
        type: "error",
        message: `Network error: ${error.message}`,
      });
      return false;
    } finally {
      setUploading(false);
    }
  }, []);

  const analyzeFile = useCallback(
    async (filename) => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/api/upload/analyze/${filename}`,
          {
            method: "POST",
          }
        );

        const data = await response.json();

        if (response.ok) {
          setAnalysis(data);
        }
      } catch (error) {
        console.error("Analysis failed:", error);
      }
    },
    [API_BASE_URL]
  );

  return { handleUpload, uploading, uploadStatus, analysis };
}