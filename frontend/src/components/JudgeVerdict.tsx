"use client";

import React from "react";

export default function JudgeVerdict({ data }: { data: any }) {
  const winner = data.winner;
  const scores = data.scores || {};
  
  // Calculate confidence percentage safely
  const confidence = scores.confidence_score ? Math.min(100, Math.max(0, scores.confidence_score * 100)) : 0;
  
  return (
    <div className="premium-card" style={{ 
      borderTop: `6px solid var(--teal-primary)`,
      backgroundColor: "var(--bg-base)",
      color: "var(--ink)",
      marginTop: "16px"
    }}>
      <div style={{ textAlign: "center", marginBottom: "32px" }}>
        <h2 className="font-serif" style={{ fontSize: "2.5rem", color: "var(--teal-deep)", marginBottom: "8px" }}>
          Final Verdict
        </h2>
        <p style={{ fontSize: "1.1rem", color: "var(--teal-muted)" }}>
          Evaluated by the Blind Judge based on logical rigor and factual accuracy.
        </p>
      </div>

      <div style={{ 
        display: "flex", 
        flexDirection: "row",
        flexWrap: "wrap",
        gap: "32px",
        padding: "32px", 
        backgroundColor: "var(--bg-surface)", 
        borderRadius: "8px",
        marginBottom: "32px",
        border: "1px solid rgba(124, 140, 138, 0.2)"
      }}>
        <div style={{ flex: "1 1 200px" }}>
          <h3 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-muted)", marginBottom: "12px" }}>
            Declared Winner
          </h3>
          <div className="font-serif" style={{ fontSize: "2.5rem", color: "var(--teal-primary)" }}>
            {winner}
          </div>
        </div>
        
        {scores.confidence_score !== undefined && (
          <div style={{ flex: "2 1 300px" }}>
            <h3 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-muted)", marginBottom: "12px" }}>
              Confidence Score
            </h3>
            <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
              <div className="font-mono" style={{ fontSize: "2rem", color: "var(--teal-deep)" }}>
                {confidence.toFixed(0)}%
              </div>
              <div style={{ flex: 1, height: "8px", backgroundColor: "var(--bg-base)", borderRadius: "4px", overflow: "hidden" }}>
                <div style={{ 
                  height: "100%", 
                  width: `${confidence}%`, 
                  backgroundColor: "var(--teal-primary)",
                  transition: "width 1s cubic-bezier(0.16, 1, 0.3, 1)"
                }} />
              </div>
            </div>
          </div>
        )}
      </div>
      
      {scores.verdict_summary && (
        <div style={{ padding: "0 16px", marginBottom: "32px" }}>
          <h4 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-deep)", marginBottom: "12px" }}>
            Judgement Rationale
          </h4>
          <p style={{ fontSize: "1.1rem", lineHeight: "1.7" }}>{scores.verdict_summary}</p>
        </div>
      )}
      
      {(data.total_latency || data.total_tokens) && (
        <div style={{ 
          marginTop: "32px", 
          paddingTop: "24px", 
          borderTop: "1px solid rgba(124, 140, 138, 0.2)", 
          display: "flex", 
          gap: "32px",
          color: "var(--teal-muted)",
          fontSize: "0.9rem",
          textTransform: "uppercase",
          letterSpacing: "0.05em"
        }}>
          {data.total_latency && (
            <div>Execution Time: <span className="font-mono" style={{ color: "var(--ink)", marginLeft: "8px", fontSize: "1rem" }}>{data.total_latency.toFixed(2)}s</span></div>
          )}
          {data.total_tokens && (
            <div>Tokens Processed: <span className="font-mono" style={{ color: "var(--ink)", marginLeft: "8px", fontSize: "1rem" }}>{data.total_tokens.toLocaleString()}</span></div>
          )}
        </div>
      )}
    </div>
  );
}
