"use client";

import React from "react";

interface AgentCardProps {
  role: string;
  data: any;
  isLoading?: boolean;
}

export default function AgentCard({ role, data, isLoading }: AgentCardProps) {
  if (isLoading || !data) {
    return (
      <div className="premium-card" style={{ 
        borderTop: "4px solid var(--teal-muted)", 
        opacity: 0.7,
        height: "100%",
        display: "flex",
        flexDirection: "column"
      }}>
        <h2 className="font-serif" style={{ fontSize: "1.75rem", color: "var(--teal-muted)", marginBottom: "16px" }}>
          {role}
        </h2>
        <div style={{ display: "flex", alignItems: "center", gap: "12px", color: "var(--teal-muted)", fontStyle: "italic" }}>
          <div style={{
            width: "8px", height: "8px", backgroundColor: "var(--teal-muted)", borderRadius: "50%",
            animation: "pulse-opacity 1.5s infinite"
          }} />
          Awaiting formulation...
        </div>
        <style>{`
          @keyframes pulse-opacity {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
          }
        `}</style>
      </div>
    );
  }

  // Determine core claim field dynamically
  const claim = data.claim || data.target_claim || "Critical Analysis";
  
  return (
    <div className="premium-card" style={{ borderTop: `4px solid var(--teal-primary)`, height: "100%" }}>
      <h2 className="font-serif" style={{ fontSize: "1.75rem", color: "var(--teal-deep)", marginBottom: "24px" }}>
        {role}
      </h2>
      
      {claim !== "Critical Analysis" && (
        <div style={{ marginBottom: "24px", paddingBottom: "24px", borderBottom: "1px solid rgba(124, 140, 138, 0.2)" }}>
          <div style={{ fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-muted)", marginBottom: "8px" }}>
            {data.target_claim ? "Target Claim" : "Core Argument"}
          </div>
          <p style={{ fontSize: "1.1rem", fontWeight: "500", color: "var(--ink)" }}>{claim}</p>
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
        {/* Proponent Fields */}
        {data.reasoning && (
          <div>
            <h4 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-deep)" }}>Reasoning</h4>
            <ul className="data-list">
              {data.reasoning.map((r: string, i: number) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}
        
        {data.supporting_evidence && (
          <div>
            <h4 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-deep)" }}>Evidence</h4>
            <ul className="data-list">
              {data.supporting_evidence.map((r: string, i: number) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}

        {/* Opponent Fields */}
        {data.counter_arguments && (
          <div>
            <h4 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-deep)" }}>Counter Arguments</h4>
            <ul className="data-list">
              {data.counter_arguments.map((r: string, i: number) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}

        {data.flaws_identified && (
          <div>
            <h4 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-deep)" }}>Flaws Identified</h4>
            <ul className="data-list">
              {data.flaws_identified.map((r: string, i: number) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}

        {/* Critic Fields */}
        {data.logical_fallacies && (
          <div>
            <h4 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-deep)" }}>Logical Fallacies</h4>
            {data.logical_fallacies.length > 0 ? (
              <ul className="data-list">
                {data.logical_fallacies.map((r: string, i: number) => <li key={i}>{r}</li>)}
              </ul>
            ) : (
              <p style={{ color: "var(--teal-muted)", marginTop: "8px", fontStyle: "italic" }}>None identified.</p>
            )}
          </div>
        )}

        {data.argument_a_analysis && (
          <div>
            <h4 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-deep)" }}>Analysis: Proponent</h4>
            <p style={{ color: "var(--ink)", marginTop: "8px" }}>{data.argument_a_analysis}</p>
          </div>
        )}

        {data.argument_b_analysis && (
          <div>
            <h4 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--teal-deep)" }}>Analysis: Opponent</h4>
            <p style={{ color: "var(--ink)", marginTop: "8px" }}>{data.argument_b_analysis}</p>
          </div>
        )}
      </div>
    </div>
  );
}
