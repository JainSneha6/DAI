import { useCallback } from "react";

export function useFileSelection() {
  const onFilesSelected = useCallback((selectedFiles) => {
    const arr = Array.from(selectedFiles).map((file) => ({
      id: `${file.name}-${file.size}-${file.lastModified}`,
      file,
      preview: file.type.startsWith("image/") ? URL.createObjectURL(file) : null,
    }));
    return arr;
  }, []);

  return { onFilesSelected };
}