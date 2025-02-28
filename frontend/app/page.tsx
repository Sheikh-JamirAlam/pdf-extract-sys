"use client";

import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import dynamic from "next/dynamic";
import { VariableSizeList as List } from "react-window";

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

  const handlePollingTranscript = async (jobId: string) => {
    const pollingInterval = setInterval(async () => {
      try {
        const response = await axios.get(`http://localhost:8000/status/${jobId}`);

        // Update progress indicator
        //updateProgress(jobStatus.pages_processed, jobStatus.total_pages);

        // Display new results
        setTranscript(response.data.results);

        // If job is complete, stop polling
        if (response.data.status === "completed" || response.data.status === "failed") {
          clearInterval(pollingInterval);

          if (response.data.status === "failed") {
            alert(response.data.error);
          } else {
            alert("Processing complete!");
          }
        }
      } catch (error) {
        clearInterval(pollingInterval);
        console.error("Polling error:", error);
      }
    }, 2000);
  };

  const handleExtract = async () => {
    try {
      setIsLoading(true);
      const response = await axios.post("http://localhost:8000/extract", { pdf_url: pdfUrl });
      setTranscript(response.data.results);
      if (response.data.job_id && response.data.status === "processing") {
        handlePollingTranscript(response.data.job_id);
      }
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

  // Simple heuristic to compute the height of a row based on its text length.
  const getItemSize = useCallback(
    (index: number) => {
      const text = transcript[index]?.text || "";
      // Assume roughly 50 characters per line and 20px per line.
      const lines = Math.ceil(text.length / 50);
      // Minimum height of 30px, add 20px per additional line.
      return Math.max(30, lines * 20);
    },
    [transcript]
  );

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
          <List height={600} itemCount={transcript.length} itemSize={getItemSize} width={"100%"} itemData={transcript}>
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
