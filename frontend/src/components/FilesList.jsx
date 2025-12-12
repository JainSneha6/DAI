// src/components/FilesList.jsx
import React, { useEffect, useState } from "react";

export default function FilesList() {
    const [files, setFiles] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        setLoading(true);
        fetch("/api/files")
            .then((r) => r.json())
            .then((j) => {
                if (j.success) setFiles(j.files || []);
            })
            .catch((e) => console.error(e))
            .finally(() => setLoading(false));
    }, []);

    return (
        <div className="p-4 bg-white/3 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">Uploaded Files</h3>
            {loading ? <div>Loading…</div> : null}
            <ul className="space-y-3">
                {files.map((f) => (
                    <li key={f.filename} className="p-3 rounded-md bg-white/4">
                        <div className="flex justify-between items-start gap-3">
                            <div>
                                <div className="font-medium">{f.filename}</div>
                                <div className="text-xs text-slate-300">Category: <span className="font-semibold">{f.category}</span></div>
                                <div className="text-xs text-slate-300">Rows: {f.row_count ?? "?"}</div>
                                <div className="text-xs text-slate-300 mt-2">Columns: {(f.columns || []).slice(0, 6).join(", ")}</div>
                            </div>
                            <div>
                                <a className="text-sm underline" href={`/models/${f.filename}`} target="_blank" rel="noreferrer">Download</a>
                            </div>
                        </div>
                    </li>
                ))}
            </ul>
        </div>
    );
}
