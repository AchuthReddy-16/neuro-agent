import { Suspense } from "react";
import { WorkspaceLayout } from "@/components/workspace/WorkspaceLayout";

export default function WorkspacePage() {
  return (
    <Suspense
      fallback={
        <div className="h-screen flex items-center justify-center bg-surface text-muted text-sm">
          Loading workspace…
        </div>
      }
    >
      <WorkspaceLayout />
    </Suspense>
  );
}
