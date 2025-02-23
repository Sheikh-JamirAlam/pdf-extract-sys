"use client";

import { useState } from "react";
import axios from "axios";
import dynamic from "next/dynamic";

const PDFViewer = dynamic(() => import("./components/PDFViewer"), {
  ssr: false,
});

interface ExtractedItem {
  text: string;
  bbox: [number, number, number, number];
}

export default function Home() {
  const [pdfUrl, setPdfUrl] = useState("");
  const [transcript, setTranscript] = useState<ExtractedItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Function to call the backend extraction API
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

  return (
    <div className="flex">
      {/* Sidebar for URL input and transcript */}
      <div className="w-1/2 p-4">
        <h2 className="text-xl font-bold mb-4">PDF URL Input</h2>
        <input type="text" className="border p-2 w-full" value={pdfUrl} onChange={(e) => setPdfUrl(e.target.value)} placeholder="Enter PDF URL" />
        <button onClick={handleExtract} className="mt-2 bg-blue-500 text-white py-2 px-4 rounded">
          {isLoading ? "Extracting..." : "Extract PDF"}
        </button>
        <h2 className="text-xl font-bold mt-6">Transcript</h2>
        <div className="mt-2 p-4 border rounded max-h-[600px] overflow-y-auto">
          {transcript.map((item, index) => (
            <div key={index} className="whitespace-pre-line">
              {item.text}
            </div>
          ))}
        </div>
      </div>
      {/* Main area for PDF viewing */}
      <PDFViewer url={pdfUrl} />
    </div>
  );
}
