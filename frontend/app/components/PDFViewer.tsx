"use client";

import { Worker, Viewer } from "@react-pdf-viewer/core";
import "@react-pdf-viewer/core/lib/styles/index.css";

interface PDFViewerProps {
  url: string;
}

export default function PDFViewer({ url }: PDFViewerProps) {
  if (!url) {
    return (
      <div className="flex-1 p-4">
        <h2 className="text-xl font-bold mb-4">PDF Viewer</h2>
        <div className="h-[750px] flex items-center justify-center text-gray-500">Enter a PDF URL to view the document</div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-4">
      <h2 className="text-xl font-bold mb-4">PDF Viewer</h2>
      <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.4.120/build/pdf.worker.min.js">
        <div style={{ height: "750px" }}>
          <Viewer fileUrl={url} />
        </div>
      </Worker>
    </div>
  );
}
