import React, { useState, useRef, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import FileInputField from "../components/InputField.jsx";
import { useDragAndDrop } from "../hooks/useDragAndDrop.js";
import { useFileSelection } from "../hooks/useFileSelection.js";
import useFileUpload from "../hooks/useFileUpload.js";

const MODEL_CATEGORIES = [
  "Auto-detect (Best-fit)",
  "Sales, Demand & Financial Forecasting Model",
  "Pricing & Revenue Optimization Model",
  "Marketing ROI & Attribution Model",
  "Customer Segmentation & Modeling",
  "Customer Value & Retention Model",
  "Sentiment & Intent NLP Model",
  "Inventory & Replenishment Optimization Model",
  "Logistics & Supplier Risk Model",
];

export default function DecisivAIUploadPage() {
  const [files, setFiles] = useState([]);
  const inputRef = useRef(null);
  const dropRef = useRef(null);

  const { onFilesSelected: addFiles } = useFileSelection();
  const { handleUpload, uploading, uploadStatus } = useFileUpload();

  const [modelType, setModelType] = useState(MODEL_CATEGORIES[0]);

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
    const selectedModel = modelType === MODEL_CATEGORIES[0] ? null : modelType;
    const success = await handleUpload(files, { modelType: selectedModel });
    if (success) {
      setFiles([]);
    }
  }, [files, handleUpload, modelType]);

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
    <div className="min-h-screen flex items-center justify-center bg-linear-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] p-8">
      <div className="w-full max-w-5xl rounded-3xl shadow-[0_0_60px_-20px_rgba(0,0,0,0.6)] overflow-hidden relative border border-white/10 bg-linear-to-br from-[#0d111a]/60 to-[#0e1320]/40 backdrop-blur-xl p-12">
        <main className="flex flex-col items-center text-center relative z-10">
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
            className="max-w-3xl text-lg lg:text-xl text-slate-200/85 leading-relaxed mb-8"
          >
            Provide your data below — we accept multiple files. Drag & drop, or
            click the button to select files from your device.
          </motion.p>

          {/* Model selection UI */}
          <div className="mb-6 w-full max-w-3xl">
            <label className="block text-slate-200 mb-2 text-left">
              Model Category
            </label>
            <select
              className="w-full rounded-md p-2 bg-slate-800 text-slate-100 border border-white/10"
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
            >
              {MODEL_CATEGORIES.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

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
        </main>

        <div className="absolute bottom-0 left-0 w-full h-px bg-linear-to-r from-transparent via-white/20 to-transparent" />
      </div>
    </div>
  );
}
