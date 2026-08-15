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
    setStatus("Initializing debate...");
    
    // Reset state
    setProponentData(null);
    setOpponentData(null);
    setCriticData(null);
    setJudgeData(null);

    try {
      const response = await fetch("http://localhost:8000/api/v1/debate/run-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: debateTopic,
          max_rounds: 3,
        }),
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
              } else if (event.event === "error") {
                console.error("Debate error:", event.data);
                setStatus("An error occurred during the debate.");
              }
            } catch (e) {
              console.error("Failed to parse event line", line, e);
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setStatus("Failed to connect to debate stream.");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <main className="max-w-[1200px] mx-auto px-6 py-16">
      <header className="mb-12">
        <h1 className="font-serif text-3xl text-primary mb-1">
          Execution Environment
        </h1>
        <p className="font-sans text-sm text-muted-foreground">
          Multi-Agent Evaluation Pipeline
        </p>
      </header>

      {!isRunning && (
        <div className="animate-in-subtle max-w-[600px] mx-auto bg-card border border-border/20 rounded-xl p-8 shadow-sm">
          <DebateForm onSubmit={startDebate} />
        </div>
      )}

      {topic && (
        <div className="animate-in-subtle text-center mb-12">
          <div className="text-xs uppercase tracking-widest text-muted-foreground mb-2">
            Proposition
          </div>
          <div className="font-serif text-2xl text-primary">
            {topic}
          </div>
        </div>
      )}

      {isRunning && (
        <div className="mb-12 flex justify-center">
          <LiveStatus status={status} />
        </div>
      )}

      {/* Grid Layout for Proponent and Opponent */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {(isRunning || proponentData) && (
          <div className="animate-in-subtle">
            <AgentCard role="Proponent" data={proponentData} isLoading={isRunning && !proponentData} />
          </div>
        )}
        
        {(proponentData || opponentData) && (
          <div className="animate-in-subtle [animation-delay:100ms]">
            <AgentCard role="Opponent" data={opponentData} isLoading={!opponentData && !!proponentData} />
          </div>
        )}
      </div>

      {/* Centered layout for Critic */}
      {(opponentData || criticData) && (
        <div className="animate-in-subtle [animation-delay:200ms] max-w-[800px] mx-auto mb-12">
          <AgentCard role="Critic" data={criticData} isLoading={!criticData && !!opponentData} />
        </div>
      )}
      
      {/* Full width verdict for Judge */}
      {judgeData && (
        <div className="animate-in-subtle [animation-delay:300ms]">
          <JudgeVerdict data={judgeData} />
        </div>
      )}
    </main>
  );
}
