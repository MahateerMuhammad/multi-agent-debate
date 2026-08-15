"use client";

import React, { useState } from "react";

import { Textarea } from "./ui/textarea";
import { Button } from "./ui/button";

export default function DebateForm({ onSubmit }: { onSubmit: (topic: string) => void }) {
  const [topic, setTopic] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (topic.trim()) {
      onSubmit(topic);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-6">
      <div className="space-y-2">
        <label htmlFor="topic" className="block text-sm font-medium text-foreground">
          Proposition
        </label>
        <Textarea
          id="topic"
          placeholder="Enter the claim to be debated..."
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          required
          rows={4}
          className="resize-y min-h-[120px] text-base"
        />
      </div>
      <Button type="submit" size="lg" className="w-fit">
        Execute Run
      </Button>
    </form>
  );
}
