import { useEffect } from "react";

export function useDragAndDrop(dropRef, onFilesSelected) {
  useEffect(() => {
    const dropArea = dropRef.current;
    if (!dropArea) return;

    const onDragOver = (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
      dropArea.classList.add("ring-2", "ring-offset-2", "ring-violet-500/60");
    };

    const onDragLeave = () => {
      dropArea.classList.remove("ring-2", "ring-offset-2", "ring-violet-500/60");
    };

    const onDrop = (e) => {
      e.preventDefault();
      dropArea.classList.remove("ring-2", "ring-offset-2", "ring-violet-500/60");
      if (e.dataTransfer.files && e.dataTransfer.files.length) {
        onFilesSelected(e.dataTransfer.files);
      }
    };

    dropArea.addEventListener("dragover", onDragOver);
    dropArea.addEventListener("dragleave", onDragLeave);
    dropArea.addEventListener("drop", onDrop);

    return () => {
      dropArea.removeEventListener("dragover", onDragOver);
      dropArea.removeEventListener("dragleave", onDragLeave);
      dropArea.removeEventListener("drop", onDrop);
    };
  }, [onFilesSelected, dropRef]);
}