import React, { useEffect, useState } from "react";

export default function FilesList() {
    const [files, setFiles] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setLoading(true);
        fetch("http://localhost:5000/api/files")
            .then((r) => r.json())
            .then((j) => {
                if (j && j.success && Array.isArray(j.files)) {
                    setFiles(j.files);
                } else {
                    setFiles([]);
                }
            })
            .catch((e) => {
                console.error("Failed to fetch files", e);
                setFiles([]);
            })
            .finally(() => setLoading(false));
    }, []);

    return (
        <div className="p-4 bg-white/3 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">Uploaded Files</h3>
            {loading && <div>Loading…</div>}

            <ul className="space-y-3">
                {files.map((f) => {
                    const filename = f.filename || "unknown-file";

                    // CATEGORY — ALWAYS SAFE
                    let category = "Unknown";
                    if (typeof f.category === "string") {
                        category = f.category;
                    } else if (f.category && typeof f.category === "object") {
                        category =
                            f.category.data_domain ||
                            f.category.category ||
                            "Unknown";
                    }

                    const rowCount = f.row_count || 0;

                    const columns = Array.isArray(f.columns) ? f.columns : [];

                    const classification =
                        f.classification && typeof f.classification === "object"
                            ? f.classification
                            : null;

                    return (
                        <li
                            key={filename}
                            className="p-3 rounded-md bg-white/4"
                        >
                            <div className="flex justify-between items-start gap-3">
                                <div>
                                    <div className="font-medium">{filename}</div>

                                    <div className="text-xs text-slate-300">
                                        Category:{" "}
                                        <span className="font-semibold">
                                            {category}
                                        </span>
                                        {classification?.confidence != null && (
                                            <span className="ml-2 text-slate-400">
                                                (
                                                {Math.round(
                                                    classification.confidence *
                                                        100
                                                )}
                                                %)
                                            </span>
                                        )}
                                    </div>

                                    <div className="text-xs text-slate-300">
                                        Rows: {rowCount}
                                    </div>

                                    <div className="text-xs text-slate-300 mt-2">
                                        Columns:{" "}
                                        {columns.slice(0, 6).join(", ") || "—"}
                                    </div>
                                </div>

                                <div>
                                    <a
                                        className="text-sm underline"
                                        href={`/models/${encodeURIComponent(
                                            filename
                                        )}`}
                                        target="_blank"
                                        rel="noreferrer"
                                    >
                                        Download
                                    </a>
                                </div>
                            </div>
                        </li>
                    );
                })}
            </ul>
        </div>
    );
}
