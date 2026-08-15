import React from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Progress } from "./ui/progress";

export default function JudgeVerdict({ data }: { data: any }) {
  const winner = data.winner;
  const scores = data.scores || {};
  
  // Calculate confidence percentage safely
  const confidence = scores.confidence_score ? Math.min(100, Math.max(0, scores.confidence_score * 100)) : 0;
  
  return (
    <Card className="mt-4 border-t-8 border-t-primary bg-card/50 shadow-md">
      <CardHeader className="text-center pb-8 pt-8">
        <CardTitle className="font-serif text-4xl text-primary mb-2">Final Verdict</CardTitle>
        <CardDescription className="text-base">
          Evaluated by the Blind Judge based on logical rigor and factual accuracy.
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-8">
        <div className="flex flex-row flex-wrap gap-8 p-8 bg-card rounded-xl border border-border/50">
          <div className="flex-1 min-w-[200px]">
            <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-3">
              Declared Winner
            </h3>
            <div className="font-serif text-4xl text-primary">
              {winner}
            </div>
          </div>
          
          {scores.confidence_score !== undefined && (
            <div className="flex-[2] min-w-[300px]">
              <h3 className="text-xs uppercase tracking-widest text-muted-foreground mb-3">
                Confidence Score
              </h3>
              <div className="flex items-center gap-4">
                <div className="font-mono text-3xl text-primary">
                  {confidence.toFixed(0)}%
                </div>
                <Progress value={confidence} className="h-3 flex-1" />
              </div>
            </div>
          )}
        </div>
        
        {scores.verdict_summary && (
          <div className="px-4">
            <h4 className="text-xs uppercase tracking-widest text-primary mb-3">
              Judgement Rationale
            </h4>
            <p className="text-lg leading-relaxed text-foreground/90">{scores.verdict_summary}</p>
          </div>
        )}
        
        {(data.total_latency || data.total_tokens) && (
          <div className="mt-8 pt-6 border-t border-border/50 flex flex-wrap gap-8 text-xs uppercase tracking-widest text-muted-foreground px-4">
            {data.total_latency && (
              <div>Execution Time: <span className="font-mono text-sm text-foreground ml-2">{data.total_latency.toFixed(2)}s</span></div>
            )}
            {data.total_tokens && (
              <div>Tokens Processed: <span className="font-mono text-sm text-foreground ml-2">{data.total_tokens.toLocaleString()}</span></div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
