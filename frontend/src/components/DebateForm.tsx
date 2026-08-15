"use client";

import React, { useState } from "react";

export default function DebateForm({ onSubmit }: { onSubmit: (topic: string) => void }) {
  const [topic, setTopic] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (topic.trim()) {
      onSubmit(topic);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div>
        <label htmlFor="topic" className="font-sans" style={{ display: "block", marginBottom: "8px", fontWeight: "500", color: "var(--ink)" }}>
          Proposition
        </label>
        <textarea
          id="topic"
          className="premium-input font-sans"
          placeholder="Enter the claim to be debated..."
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          required
          rows={4}
          style={{ resize: "vertical", minHeight: "100px", lineHeight: "1.5" }}
        />
      </div>
      <button type="submit" className="premium-button" style={{ alignSelf: "flex-start" }}>
        Execute Run
      </button>
    </form>
  );
}
