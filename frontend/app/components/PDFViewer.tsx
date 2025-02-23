"use client";

import { Worker, Viewer } from "@react-pdf-viewer/core";
import { defaultLayoutPlugin } from "@react-pdf-viewer/default-layout";
import { HighlightArea, highlightPlugin, RenderHighlightsProps, Trigger } from "@react-pdf-viewer/highlight";
import "@react-pdf-viewer/core/lib/styles/index.css";
import "@react-pdf-viewer/default-layout/lib/styles/index.css";
import { useEffect } from "react";

interface ExtractedItem {
  text: string;
  bbox: [number, number, number, number];
  pageNumber?: number;
}

interface PDFViewerProps {
  url: string;
  selectedText?: ExtractedItem | null;
}

export default function PDFViewer({ url, selectedText }: PDFViewerProps) {
  const defaultLayoutPluginInstance = defaultLayoutPlugin();

  const convertToHighlightArea = (item: ExtractedItem): HighlightArea => {
    const [left, top, right, bottom] = item.bbox;
    return {
      pageIndex: item.pageNumber || 0,
      left: left,
      top: top,
      width: right - left,
      height: bottom - top,
    };
  };

  const areas: HighlightArea[] = selectedText ? [convertToHighlightArea(selectedText)] : [];

  const renderHighlights = (props: RenderHighlightsProps) => (
    <div>
      {areas
        .filter((area) => area.pageIndex === props.pageIndex)
        .map((area, idx) => (
          <div
            key={idx}
            className="highlight-area"
            style={Object.assign(
              {},
              {
                background: "yellow",
                opacity: 0.4,
              },
              props.getCssProperties(area, props.rotation)
            )}
          />
        ))}
    </div>
  );

  const highlightPluginInstance = highlightPlugin({
    renderHighlights,
    trigger: Trigger.None,
  });

  const { jumpToHighlightArea } = highlightPluginInstance;

  useEffect(() => {
    if (selectedText && jumpToHighlightArea) {
      const highlightArea = convertToHighlightArea(selectedText);
      jumpToHighlightArea(highlightArea);
    }
  }, [selectedText, jumpToHighlightArea]);

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
      <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
        <div style={{ height: "750px" }}>
          <Viewer fileUrl={url} plugins={[defaultLayoutPluginInstance, highlightPluginInstance]} />
        </div>
      </Worker>
    </div>
  );
}
