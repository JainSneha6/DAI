import React, { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, CheckCircle2, Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import FileInputField from "../components/InputField.jsx";
import { useDragAndDrop } from "../hooks/useDragAndDrop.js";
import { useFileSelection } from "../hooks/useFileSelection.js";
import useFileUpload from "../hooks/useFileUpload.js";

export default function DecisivAIUploadPage() {
  const [files, setFiles] = useState([]);
  const [showSuccess, setShowSuccess] = useState(false);
  const inputRef = useRef(null);
  const dropRef = useRef(null);
  const navigate = useNavigate();

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
      setShowSuccess(true);
      setTimeout(() => {
        navigate("/dashboard");
      }, 2000);
    }
  }, [files, handleUpload, navigate]);

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
    <div className="min-h-screen bg-gradient-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] relative overflow-hidden">

      {/* Ambient glow effects */}
      <div className="absolute top-20 left-32 w-96 h-96 bg-blue-600/15 rounded-full blur-[120px]" />
      <div className="absolute bottom-10 right-40 w-[28rem] h-[28rem] bg-violet-600/15 rounded-full blur-[120px]" />

      {/* Grid overlay */}
      <div className="absolute inset-0 opacity-[0.02]">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="upload-grid" width="60" height="60" patternUnits="userSpaceOnUse">
              <path d="M 60 0 L 0 0 0 60" fill="none" stroke="white" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#upload-grid)" />
        </svg>
      </div>

      <div className="relative z-10 p-8">

        {/* Progress indicator */}
        <div className="max-w-7xl mx-auto mb-8">
          <div className="flex items-center justify-center gap-2 text-sm">
            <div className="flex items-center gap-2 text-slate-400">
              <div className="w-8 h-8 rounded-full bg-blue-500/20 border border-blue-400/50 flex items-center justify-center text-blue-400 font-medium">
                1
              </div>
              <span>Upload Data</span>
            </div>
            <div className="w-16 h-px bg-slate-700" />
            <div className="flex items-center gap-2 text-slate-600">
              <div className="w-8 h-8 rounded-full bg-slate-800/50 border border-slate-700 flex items-center justify-center text-slate-600 font-medium">
                2
              </div>
              <span>Dashboard</span>
            </div>
            <div className="w-16 h-px bg-slate-700" />
            <div className="flex items-center gap-2 text-slate-600">
              <div className="w-8 h-8 rounded-full bg-slate-800/50 border border-slate-700 flex items-center justify-center text-slate-600 font-medium">
                3
              </div>
              <span>Analyze</span>
            </div>
          </div>
        </div>

        <div className="w-full max-w-7xl mx-auto">
          <div className="rounded-[2rem] shadow-[0_20px_80px_-20px_rgba(0,0,0,0.7)] overflow-hidden relative border border-white/10 bg-gradient-to-br from-[#0d111a]/80 to-[#0e1320]/60 backdrop-blur-2xl">

            {/* Top accent */}
            <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-blue-400/40 to-transparent" />

            <div className="p-12 lg:p-16">

              <div className="mb-12">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-400/20 text-blue-400 text-sm font-medium mb-6"
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                  Step 1 of 3
                </motion.div>

                <motion.h1
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                  className="text-4xl lg:text-5xl font-bold text-slate-50 tracking-tight mb-4 leading-tight"
                >
                  Upload Your Data
                </motion.h1>

                <motion.p
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.2 }}
                  className="max-w-3xl text-lg text-slate-300 leading-relaxed font-light"
                >
                  Provide your data files to begin your analysis journey. We support multiple formats
                  including CSV, XLSX, PDF, images, JSON, and more.
                </motion.p>
              </div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
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
              </motion.div>

            </div>

            {/* Bottom accent */}
            <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-violet-400/40 to-transparent" />
          </div>
        </div>
      </div>

      {/* Success Modal */}
      <AnimatePresence>
        {showSuccess && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gradient-to-br from-[#0d111a]/95 to-[#0e1320]/95 backdrop-blur-xl rounded-2xl p-8 border border-white/10 max-w-md w-full text-center shadow-[0_20px_80px_-20px_rgba(0,0,0,0.9)]"
            >
              <div className="w-20 h-20 rounded-full bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border border-emerald-400/30 flex items-center justify-center mx-auto mb-6">
                <CheckCircle2 className="w-10 h-10 text-emerald-400" />
              </div>
              <h3 className="text-2xl font-bold text-slate-50 mb-3">Upload Successful!</h3>
              <p className="text-slate-400 mb-6">Redirecting to dashboard...</p>
              <Loader2 className="w-6 h-6 text-blue-400 animate-spin mx-auto" />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}