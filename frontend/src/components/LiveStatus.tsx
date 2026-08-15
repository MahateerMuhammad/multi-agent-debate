"use client";

import React from "react";

export default function LiveStatus({ status }: { status: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "12px", color: "var(--teal-muted)", fontStyle: "italic", fontSize: "1.1rem" }}>
      <div style={{
        width: "6px",
        height: "6px",
        backgroundColor: "var(--teal-muted)",
        borderRadius: "50%",
        animation: "pulse-opacity 1.5s infinite"
      }} />
      <div>{status || "Agent is thinking..."}</div>
      
      <style>{`
        @keyframes pulse-opacity {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
