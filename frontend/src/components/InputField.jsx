import React from "react";
import { motion } from "framer-motion";
import { ArrowRight, UploadCloud, X, FileText, Image, FileSpreadsheet, File } from "lucide-react";

function getFileIcon(fileType) {
  if (fileType.startsWith('image/')) return Image;
  if (fileType.includes('spreadsheet') || fileType.includes('csv') || fileType.includes('excel')) return FileSpreadsheet;
  return File;
}

function formatFileSize(bytes) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function FileInputField({
  files,
  onFilesSelected,
  inputRef,
  dropRef,
  handleClearAll,
  onFileRemove,
  onOpenPicker,
  onUpload,
  uploading,
  uploadStatus,
}) {
  return (
    <div ref={dropRef} className="w-full">

      {/* Drop Zone */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="rounded-2xl border-2 border-dashed border-white/10 bg-gradient-to-br from-white/5 to-white/[0.02] backdrop-blur-xl p-8 hover:border-blue-400/30 transition-colors duration-300"
      >
        <div className="flex flex-col lg:flex-row items-center justify-between gap-6">
          <div className="flex-1 text-center lg:text-left">
            <label htmlFor="file-input" className="block cursor-pointer">
              <div className="flex flex-col lg:flex-row items-center gap-4 lg:gap-6">
                <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-[#4c6ef5]/20 to-[#7c3aed]/20 border border-white/10 shadow-lg">
                  <UploadCloud size={32} className="text-blue-400" strokeWidth={1.5} />
                </div>

                <div>
                  <div className="text-slate-300 text-sm mb-1">Provide your data</div>
                  <div className="text-xl lg:text-2xl font-semibold text-slate-50 mb-2">
                    Multiple files supported
                  </div>
                  <div className="text-sm text-slate-400">
                    CSV, XLSX, PDF, images, JSON, and more
                  </div>
                </div>
              </div>
            </label>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onOpenPicker}
              className="inline-flex items-center gap-3 rounded-full px-6 py-3 bg-gradient-to-r from-[#4c6ef5] to-[#7c3aed] text-white font-medium shadow-[0_0_30px_-10px_rgba(124,58,237,0.5)] hover:shadow-[0_0_40px_-10px_rgba(124,58,237,0.7)] transition-all duration-300 hover:scale-105 focus:outline-none focus:ring-4 focus:ring-violet-700/40"
            >
              Select Files
            </button>

            {files.length > 0 && (
              <button
                onClick={handleClearAll}
                className="inline-flex items-center gap-2 rounded-full px-5 py-3 bg-white/5 border border-white/10 text-slate-300 hover:bg-white/10 transition-all duration-300 focus:outline-none"
                aria-label="Clear all files"
              >
                Clear All
              </button>
            )}
          </div>
        </div>

        <input
          id="file-input"
          ref={inputRef}
          type="file"
          multiple
          onChange={(e) => {
            onFilesSelected(e.target.files);
            e.target.value = null;
          }}
          className="hidden"
        />
      </motion.div>

      {/* File Preview List */}
      {files.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mt-6"
        >
          <div className="rounded-2xl border border-white/10 bg-gradient-to-br from-white/5 to-white/[0.02] backdrop-blur-xl p-6">

            {/* Header */}
            <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/10">
              <div>
                <h3 className="text-lg font-semibold text-slate-100">
                  Selected Files
                </h3>
                <p className="text-sm text-slate-400 mt-1">
                  {files.length} file{files.length > 1 ? "s" : ""} • Total: {formatFileSize(
                    files.reduce((s, f) => s + f.file.size, 0)
                  )}
                </p>
              </div>
            </div>

            {/* Files Grid */}
            <div className="space-y-3 max-h-[400px] overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10 pr-2">
              {files.map((f, index) => {
                const FileIcon = getFileIcon(f.file.type);

                return (
                  <motion.div
                    key={f.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className="group flex items-center gap-4 p-4 rounded-xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 hover:border-blue-400/30 transition-all duration-300"
                  >

                    {/* File Preview/Icon */}
                    <div className="w-14 h-14 rounded-lg flex items-center justify-center bg-gradient-to-br from-blue-500/10 to-violet-500/10 border border-white/10 overflow-hidden flex-shrink-0">
                      {f.preview ? (
                        <img
                          src={f.preview}
                          alt={f.file.name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <FileIcon size={24} className="text-blue-400" strokeWidth={1.5} />
                      )}
                    </div>

                    {/* File Info */}
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-slate-100 text-sm truncate mb-1">
                        {f.file.name}
                      </div>
                      <div className="flex items-center gap-3 text-xs text-slate-400">
                        <span>{formatFileSize(f.file.size)}</span>
                        {f.file.type && (
                          <>
                            <span>•</span>
                            <span className="truncate">{f.file.type}</span>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Remove Button */}
                    <button
                      onClick={() => onFileRemove(f.id)}
                      className="p-2 rounded-lg bg-white/5 hover:bg-red-500/20 border border-white/10 hover:border-red-400/30 transition-all duration-300 opacity-0 group-hover:opacity-100 focus:opacity-100 focus:outline-none"
                      aria-label={`Remove ${f.file.name}`}
                    >
                      <X size={16} className="text-slate-400 group-hover:text-red-400" />
                    </button>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </motion.div>
      )}

      {/* Action Button */}
      {files.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mt-6 flex items-center justify-end"
        >
          <button
            disabled={uploading}
            className="group inline-flex items-center gap-3 rounded-full px-8 py-4 bg-gradient-to-r from-[#10b981] to-[#06b6d4] text-white font-medium shadow-[0_0_30px_-10px_rgba(16,185,129,0.5)] hover:shadow-[0_0_40px_-10px_rgba(16,185,129,0.7)] disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 hover:scale-105 focus:outline-none focus:ring-4 focus:ring-emerald-700/40"
            onClick={onUpload}
          >
            {uploading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Uploading...
              </>
            ) : (
              <>
                Send to DecisivAI
                <ArrowRight
                  size={18}
                  className="transition-transform duration-300 group-hover:translate-x-1"
                />
              </>
            )}
          </button>
        </motion.div>
      )}

      {/* Upload Status */}
      {uploadStatus && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 rounded-xl text-sm bg-gradient-to-br from-blue-500/10 to-violet-500/10 text-slate-200 border border-blue-400/30 backdrop-blur-xl"
        >
          {typeof uploadStatus === "string"
            ? uploadStatus
            : uploadStatus?.message ?? JSON.stringify(uploadStatus)}
        </motion.div>
      )}
    </div>
  );
}

export default FileInputField;