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

  // const [currentPage, setCurrentPage] = useState(0);
  // const itemsPerLoad = 200; // Load 200 items at a time

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const handleExtract = async () => {
    try {
      setIsLoading(true);
      const response = await axios.post("http://localhost:8000/extract", { pdf_url: pdfUrl });
      setTranscript(response.data.results); // Load first batch
    } catch (error) {
      console.error("Extraction error:", error);
      alert("Failed to extract PDF");
    } finally {
      setIsLoading(false);
    }
  };

  // const loadMorePages = () => {
  //   const nextPage = currentPage + 1;
  //   const newItems = transcript.slice(nextPage * itemsPerLoad, (nextPage + 1) * itemsPerLoad);
  //   setTranscript((prev) => [...prev, ...newItems]); // Append more data
  //   setCurrentPage(nextPage);
  // };

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
        {/* <div
          className="mt-2 p-4 border rounded max-h-[600px] overflow-y-auto whitespace-pre-line"
          onScroll={(e) => {
            const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
            if (scrollTop + clientHeight >= scrollHeight - 20) loadMorePages(); // Auto-load on scroll
          }}
        >
          {visiblePages.map((item, index) => (
            <div key={index} className="cursor-pointer p-2 hover:bg-gray-200" onClick={() => handleTranscriptClick(item)}>
              {item.text}
            </div>
          ))}
          {transcript.map((item, index) => (
            <div key={index} className="cursor-pointer p-2 hover:bg-gray-200" onClick={() => handleTranscriptClick(item)}>
              {item.text}
            </div>
          ))}
        </div> */}
        <div className="mt-2 p-4 border rounded overflow-y-auto whitespace-pre-line">
          <List
            height={600} // Visible height
            itemCount={transcript.length}
            itemSize={50} // Each row height
            width={"100%"}
            itemData={transcript}
          >
            {({ index, style, data }) => {
              const item = data[index];
              return (
                <div key={index} style={style} className="cursor-pointer p-2 hover:bg-gray-200" onClick={() => handleTranscriptClick(item)}>
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
