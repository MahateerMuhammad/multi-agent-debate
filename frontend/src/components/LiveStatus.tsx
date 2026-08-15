"use client";

import React from "react";

export default function LiveStatus({ status }: { status: string }) {
  return (
    <div className="flex items-center gap-3 text-muted-foreground italic text-lg">
      <div className="w-1.5 h-1.5 bg-muted-foreground rounded-full animate-pulse" />
      <div>{status || "Awaiting initialization..."}</div>
    </div>
  );
}
