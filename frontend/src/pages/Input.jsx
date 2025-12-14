import React, { useState, useRef, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import FileInputField from "../components/InputField.jsx";
import { useDragAndDrop } from "../hooks/useDragAndDrop.js";
import { useFileSelection } from "../hooks/useFileSelection.js";
import useFileUpload from "../hooks/useFileUpload.js";

export default function DecisivAIUploadPage() {
  const [files, setFiles] = useState([]);
  const inputRef = useRef(null);
  const dropRef = useRef(null);

  const { onFilesSelected: addFiles } = useFileSelection();
  const { handleUpload, uploading, uploadStatus } = useFileUpload();

  const handleFilesSelected = useCallback(
    (selectedFiles) => {
      const newFiles = addFiles(selectedFiles);
      setFiles((prev) => {
        const existingIds = new Set(prev.map((p) => p.id));
        const filtered = newFiles.filter((a) => !existingIds.has(a.id));
        return [...prev, ...filtered];
      });
    },
    [addFiles]
  );

  const handleUploadClick = useCallback(async () => {
    const success = await handleUpload(files);
    if (success) {
      setFiles([]);
    }
  }, [files, handleUpload]);

  const handleRemove = (id) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const handleClearAll = () => {
    setFiles([]);
  };

  useDragAndDrop(dropRef, handleFilesSelected);

  useEffect(() => {
    return () => {
      files.forEach((f) => f.preview && URL.revokeObjectURL(f.preview));
    };
  }, [files]);

  return (
    <div className="min-h-screen bg-linear-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] p-8">
      <div className="w-full max-w-7xl mx-auto rounded-3xl shadow-[0_0_60px_-20px_rgba(0,0,0,0.6)] overflow-hidden relative border border-white/10 bg-linear-to-br from-[#0d111a]/60 to-[#0e1320]/40 backdrop-blur-xl p-8">
        <div className="flex flex-col lg:flex-row gap-8">
              <motion.h1
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-3xl lg:text-4xl font-semibold text-slate-100 tracking-tight mb-4"
              >
                Upload your data to DecisivAI
              </motion.h1>

              <motion.p
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                className="max-w-3xl text-lg lg:text-xl text-slate-200/85 leading-relaxed mb-6"
              >
                Provide your data below — we accept multiple files. Drag & drop,
                or click the button to select files from your device.
              </motion.p>

              <div className="w-full px-4">
                <FileInputField
                  files={files}
                  onFilesSelected={handleFilesSelected}
                  inputRef={inputRef}
                  dropRef={dropRef}
                  handleClearAll={handleClearAll}
                  onFileRemove={handleRemove}
                  onOpenPicker={() => inputRef.current && inputRef.current.click()}
                  onUpload={handleUploadClick}
                  uploading={uploading}
                  uploadStatus={uploadStatus}
                />
              </div>
        </div>

        <div className="absolute bottom-0 left-0 w-full h-px bg-linear-to-r from-transparent via-white/20 to-transparent" />
      </div>
    </div>
  );
}
