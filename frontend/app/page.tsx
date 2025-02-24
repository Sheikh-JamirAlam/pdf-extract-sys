"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import dynamic from "next/dynamic";

const PDFViewer = dynamic(() => import("./components/PDFViewer"), {
  ssr: false,
});

interface ExtractedItem {
  text: string;
  bbox: [number, number, number, number];
  pageNumber?: number;
}

export default function Home() {
  const [pdfUrl, setPdfUrl] = useState("");
  const [transcript, setTranscript] = useState<ExtractedItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const [selectedText, setSelectedText] = useState<ExtractedItem | null>(null);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const handleExtract = async () => {
    try {
      setIsLoading(true);
      const response = await axios.post("http://localhost:8000/extract", { pdf_url: pdfUrl });
      setTranscript(response.data.results);
    } catch (error) {
      console.error("Extraction error:", error);
      alert("Failed to extract PDF");
    } finally {
      setIsLoading(false);
    }
  };

  const handleTranscriptClick = (item: ExtractedItem) => {
    setSelectedText(item);
  };

  if (!isMounted) {
    return null;
  }

  return (
    <div className="flex">
      <div className="w-1/2 p-4">
        <h2 className="text-xl font-bold mb-4">PDF URL Input</h2>
        <input type="text" className="border p-2 w-full" value={pdfUrl} onChange={(e) => setPdfUrl(e.target.value)} placeholder="Enter PDF URL" />
        <button onClick={handleExtract} className="mt-2 bg-blue-500 text-white py-2 px-4 rounded">
          {isLoading ? "Extracting..." : "Extract PDF"}
        </button>
        <h2 className="text-xl font-bold mt-6">Transcript</h2>
        <div className="mt-2 p-4 border rounded max-h-[600px] overflow-y-auto">
          {transcript.map((item, index) => (
            <div key={index} className="whitespace-pre-line" onClick={() => handleTranscriptClick(item)}>
              {item.text}
            </div>
          ))}
        </div>
      </div>
      <PDFViewer url={pdfUrl} selectedText={selectedText} />
    </div>
  );
}
