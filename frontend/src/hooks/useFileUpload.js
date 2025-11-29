import { useState } from "react";

const API_URL = "http://localhost:5000";

export function useFileUpload() {
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);

  const handleUpload = async (files = [], options = {}) => {
    try {
      if (!files.length) return false;

      setUploading(true);
      setUploadStatus("Uploading");

      const formData = new FormData();
      files.forEach((file) => {
        const filePayload = file.file || file;
        formData.append("files", filePayload, filePayload.name);
      });

      // If a modelType was selected, include it (nullable)
      if (options && options.modelType) {
        formData.append("model_type", options.modelType);
      }

      const resp = await fetch(`${API_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      const json = await resp.json();
      setUploadStatus(json?.message || "Upload complete");

      setUploading(false);
      return json?.success ?? resp.ok;
    } catch (err) {
      setUploadStatus(err.message || "Upload failed");
      setUploading(false);
      return false;
    }
  };

  return {
    handleUpload,
    uploading,
    uploadStatus,
  };
}

export default useFileUpload;