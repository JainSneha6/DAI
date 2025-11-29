import React from "react";
import { motion } from "framer-motion";
import { ArrowRight, UploadCloud, X, FileText } from "lucide-react";

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
    <div ref={dropRef} className="w-full max-w-3xl">
      <div className="rounded-2xl border-2 border-dashed border-white/10 bg-white/3 p-6">
        <div className="flex items-center justify-between gap-6">
          <div className="flex-1">
            <label htmlFor="file-input" className="block text-left">
              <div className="flex items-center gap-4">
                <div className="flex items-center justify-center w-14 h-14 rounded-full bg-linear-to-br from-[#4c6ef5]/30 to-[#7c3aed]/30 shadow-md">
                  <UploadCloud size={28} className="text-white/90" />
                </div>

                <div>
                  <div className="text-sm text-slate-200">Provide your data</div>
                  <div className="text-base lg:text-lg font-medium text-slate-100">Multiple files supported</div>
                </div>
              </div>

              <div className="mt-4 text-sm text-slate-300">
                Accepts any file type — CSV, XLSX, PDF, images, JSON, and more.
              </div>
            </label>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onOpenPicker}
              className="inline-flex items-center gap-3 rounded-full px-6 py-2 bg-linear-to-r from-[#4c6ef5] to-[#7c3aed] text-white font-medium shadow-lg hover:opacity-95 focus:outline-none focus:ring-4 focus:ring-violet-700/40"
            >
              Select files
            </button>

            <button
              onClick={handleClearAll}
              className="inline-flex items-center gap-2 rounded-full px-4 py-2 bg-white/5 text-slate-200 hover:bg-white/6 focus:outline-none"
              aria-label="Clear all files"
            >
              Clear
            </button>
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
      </div>

      {/* File preview list */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, delay: 0.08 }}
        className="mt-6 bg-white/3 rounded-xl p-4 border border-white/6"
      >
        {files.length === 0 ? (
          <div className="text-slate-300 text-sm">No files selected yet.</div>
        ) : (
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm text-slate-200">{files.length} file{files.length > 1 ? "s" : ""} selected</div>
              <div className="text-sm text-slate-300">Total: {(() => {
                const total = files.reduce((s, f) => s + f.file.size, 0);
                if (total === 0) return "0 B";
                const k = 1024;
                const sizes = ["B", "KB", "MB", "GB", "TB"];
                const i = Math.floor(Math.log(total) / Math.log(k));
                return parseFloat((total / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
              })()}</div>
            </div>

            <ul className="space-y-3">
              {files.map((f) => (
                <li key={f.id} className="flex items-center gap-4 bg-white/4 p-3 rounded-md">
                  <div className="w-12 h-12 rounded-md flex items-center justify-center bg-white/6 overflow-hidden">
                    {f.preview ? (
                      <img src={f.preview} alt={f.file.name} className="w-full h-full object-cover" />
                    ) : (
                      <div className="flex items-center gap-2">
                        <FileText size={20} />
                      </div>
                    )}
                  </div>

                  <div className="flex-1 text-left">
                    <div className="font-medium text-slate-100 text-sm truncate">{f.file.name}</div>
                    <div className="text-xs text-slate-300">{(() => {
                      const bytes = f.file.size;
                      if (bytes === 0) return "0 B";
                      const k = 1024;
                      const sizes = ["B", "KB", "MB", "GB", "TB"];
                      const i = Math.floor(Math.log(bytes) / Math.log(k));
                      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
                    })()} • {f.file.type || "—"}</div>
                  </div>

                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => onFileRemove(f.id)}
                      className="p-2 rounded-md bg-white/5 hover:bg-white/6 focus:outline-none"
                      aria-label={`Remove ${f.file.name}`}
                    >
                      <X size={16} />
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </motion.div>

      {/* Action row */}
      <div className="mt-6 flex items-center justify-end gap-3">
        <button
          disabled={files.length === 0 || uploading}
          className="inline-flex items-center gap-3 rounded-full px-6 py-2 bg-linear-to-r from-[#10b981] to-[#06b6d4] text-white font-medium shadow-lg hover:opacity-95 disabled:opacity-50 focus:outline-none"
          onClick={onUpload}
        >
          {uploading ? "Uploading..." : "Send to DecisivAI"}
          <ArrowRight size={16} />
        </button>
      </div>

      {uploadStatus && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 rounded-lg text-sm bg-white/4 text-slate-200 border border-white/6"
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