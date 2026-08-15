import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface AgentCardProps {
  role: string;
  data: any;
  isLoading?: boolean;
}

export default function AgentCard({ role, data, isLoading }: AgentCardProps) {
  if (isLoading || !data) {
    return (
      <Card className="h-full border-t-4 border-t-muted opacity-70 bg-card shadow-sm">
        <CardHeader>
          <CardTitle className="font-serif text-2xl text-muted-foreground">{role}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-3 text-muted-foreground italic">
            <div className="w-2 h-2 bg-muted-foreground rounded-full animate-pulse" />
            Awaiting formulation...
          </div>
        </CardContent>
      </Card>
    );
  }

  const claim = data.claim || data.target_claim || "Critical Analysis";
  
  return (
    <Card className="h-full border-t-4 border-t-primary shadow-sm bg-card">
      <CardHeader>
        <CardTitle className="font-serif text-2xl text-primary">{role}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        {claim !== "Critical Analysis" && (
          <div className="pb-6 border-b border-border/50">
            <div className="text-xs uppercase tracking-widest text-muted-foreground mb-2">
              {data.target_claim ? "Target Claim" : "Core Argument"}
            </div>
            <p className="text-lg font-medium text-foreground leading-relaxed">{claim}</p>
          </div>
        )}

        {/* Proponent Fields */}
        {data.reasoning && (
          <div>
            <h4 className="text-xs uppercase tracking-widest text-primary mb-3">Reasoning</h4>
            <ul className="list-disc pl-5 space-y-2 text-foreground/90">
              {data.reasoning.map((r: string, i: number) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}
        
        {data.supporting_evidence && (
          <div>
            <h4 className="text-xs uppercase tracking-widest text-primary mb-3">Evidence</h4>
            <ul className="list-disc pl-5 space-y-2 text-foreground/90">
              {data.supporting_evidence.map((r: string, i: number) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}

        {/* Opponent Fields */}
        {data.counter_arguments && (
          <div>
            <h4 className="text-xs uppercase tracking-widest text-primary mb-3">Counter Arguments</h4>
            <ul className="list-disc pl-5 space-y-2 text-foreground/90">
              {data.counter_arguments.map((r: string, i: number) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}

        {data.flaws_identified && (
          <div>
            <h4 className="text-xs uppercase tracking-widest text-primary mb-3">Flaws Identified</h4>
            <ul className="list-disc pl-5 space-y-2 text-foreground/90">
              {data.flaws_identified.map((r: string, i: number) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}

        {data.sources && data.sources.length > 0 && (
          <div>
            <h4 className="text-xs uppercase tracking-widest text-primary mb-3">Sources & Citations</h4>
            <ul className="list-disc pl-5 space-y-2 text-muted-foreground italic text-sm">
              {data.sources.map((r: string, i: number) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        )}

        {/* Critic Fields */}
        {data.logical_fallacies && (
          <div>
            <h4 className="text-xs uppercase tracking-widest text-primary mb-3">Logical Fallacies</h4>
            {data.logical_fallacies.length > 0 ? (
              <ul className="list-disc pl-5 space-y-2 text-foreground/90">
                {data.logical_fallacies.map((r: string, i: number) => <li key={i}>{r}</li>)}
              </ul>
            ) : (
              <p className="text-muted-foreground italic">None identified.</p>
            )}
          </div>
        )}

        {data.argument_a_analysis && (
          <div>
            <h4 className="text-xs uppercase tracking-widest text-primary mb-2">Analysis: Proponent</h4>
            <p className="text-foreground/90 leading-relaxed">{data.argument_a_analysis}</p>
          </div>
        )}

        {data.argument_b_analysis && (
          <div>
            <h4 className="text-xs uppercase tracking-widest text-primary mb-2">Analysis: Opponent</h4>
            <p className="text-foreground/90 leading-relaxed">{data.argument_b_analysis}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
