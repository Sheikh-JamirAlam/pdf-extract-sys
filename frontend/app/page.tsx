"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import dynamic from "next/dynamic";
import { FixedSizeList as List } from "react-window";

const PDFViewer = dynamic(() => import("./components/PDFViewer"), { ssr: false });

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
        <button onClick={handleExtract} className={`mt-2 text-white py-2 px-4 rounded ${isLoading ? "bg-blue-900" : "bg-blue-500"}`} disabled={isLoading}>
          {isLoading ? "Extracting..." : "Extract PDF"}
        </button>
        <h2 className="text-xl font-bold mt-6">Transcript</h2>
        <div className="mt-2 p-4 border rounded overflow-y-auto whitespace-pre-line">
          <List height={600} itemCount={transcript.length} itemSize={30} width={"100%"} itemData={transcript}>
            {({ index, style, data }) => {
              const item = data[index];
              return (
                <div key={index} style={style} className="flex items-center cursor-pointer p-2 hover:bg-gray-200" onClick={() => handleTranscriptClick(item)}>
                  {item.text}
                </div>
              );
            }}
          </List>
        </div>
      </div>
      <PDFViewer url={pdfUrl} selectedText={selectedText} />
    </div>
  );
}
