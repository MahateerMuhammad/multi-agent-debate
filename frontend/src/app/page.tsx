"use client";

import React, { useState } from "react";
import DebateForm from "@/components/DebateForm";
import LiveStatus from "@/components/LiveStatus";
import AgentCard from "@/components/AgentCard";
import JudgeVerdict from "@/components/JudgeVerdict";

export default function Home() {
  const [topic, setTopic] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("");
  
  const [proponentData, setProponentData] = useState<any>(null);
  const [opponentData, setOpponentData] = useState<any>(null);
  const [criticData, setCriticData] = useState<any>(null);
  const [judgeData, setJudgeData] = useState<any>(null);

  const startDebate = async (debateTopic: string) => {
    setTopic(debateTopic);
    setIsRunning(true);
    setStatus("Connecting to debate server...");
    
    // Reset state
    setProponentData(null);
    setOpponentData(null);
    setCriticData(null);
    setJudgeData(null);

    try {
      const response = await fetch("http://localhost:8000/api/v1/debate/run-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: debateTopic, rounds: 1 }),
      });

      if (!response.body) {
        throw new Error("No response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let done = false;
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n").filter((line) => line.trim() !== "");
          
          for (const line of lines) {
            try {
              const event = JSON.parse(line);
              if (event.event === "status") {
                setStatus(event.data);
              } else if (event.event === "proponent") {
                setProponentData(event.data);
              } else if (event.event === "opponent") {
                setOpponentData(event.data);
              } else if (event.event === "critic") {
                setCriticData(event.data);
              } else if (event.event === "judge") {
                setJudgeData(event.data);
                setStatus("Debate concluded.");
                setIsRunning(false);
              }
            } catch (e) {
              console.error("Failed to parse chunk", line);
            }
          }
        }
      }
    } catch (err: any) {
      console.error(err);
      setStatus("Error: " + err.message);
      setIsRunning(false);
    }
  };

  return (
    <main style={{ maxWidth: "1200px", margin: "0 auto", padding: "64px 24px" }}>
      <header style={{ marginBottom: "64px", textAlign: "center" }}>
        <h1 className="font-serif" style={{ fontSize: "3rem", fontWeight: "400", color: "var(--teal-deep)" }}>
          Multi-Agent Debate System
        </h1>
        <p style={{ fontSize: "1.1rem", color: "var(--teal-muted)", marginTop: "12px", maxWidth: "600px", margin: "12px auto 0" }}>
          Observe autonomous AI agents construct arguments, rebuttals, and evaluations in real-time.
        </p>
      </header>

      {!isRunning && !judgeData && (
        <div className="premium-card animate-in" style={{ maxWidth: "600px", margin: "0 auto" }}>
          <DebateForm onSubmit={startDebate} />
        </div>
      )}

      {topic && (
        <div className="animate-in" style={{ marginBottom: "48px", textAlign: "center" }}>
          <div style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-muted)", marginBottom: "8px" }}>
            Proposition
          </div>
          <div className="font-serif" style={{ fontSize: "1.75rem", color: "var(--teal-deep)" }}>
            {topic}
          </div>
        </div>
      )}

      {isRunning && (
        <div style={{ marginBottom: "48px", display: "flex", justifyContent: "center" }}>
          <LiveStatus status={status} />
        </div>
      )}

      {/* Grid Layout for Proponent and Opponent */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: "32px", marginBottom: "32px" }}>
        {(isRunning || proponentData) && (
          <div className="animate-in">
            <AgentCard role="Proponent" data={proponentData} isLoading={isRunning && !proponentData} />
          </div>
        )}
        
        {(proponentData || opponentData) && (
          <div className="animate-in" style={{ animationDelay: "0.1s" }}>
            <AgentCard role="Opponent" data={opponentData} isLoading={!opponentData && !!proponentData} />
          </div>
        )}
      </div>

      {/* Centered layout for Critic */}
      {(opponentData || criticData) && (
        <div className="animate-in" style={{ maxWidth: "800px", margin: "0 auto 48px", animationDelay: "0.2s" }}>
          <AgentCard role="Critic" data={criticData} isLoading={!criticData && !!opponentData} />
        </div>
      )}
      
      {/* Full width verdict for Judge */}
      {judgeData && (
        <div className="animate-in" style={{ animationDelay: "0.3s" }}>
          <JudgeVerdict data={judgeData} />
        </div>
      )}
    </main>
  );
}
