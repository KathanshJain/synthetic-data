import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import type { Job } from "../api/types";
export const terminal = (job?: Job) =>
  Boolean(
    job &&
    (["completed", "failed", "succeeded", "error", "cancelled", "refused"].includes(
      job.status.toLowerCase(),
    ) ||
      job.error ||
      job.result != null),
  );
export function useLabJob(id: string) {
  const client = useQueryClient();
  const [streamed, setStreamed] = useState("");
  const query = useQuery({
    queryKey: ["job", id],
    queryFn: () => api.job(id),
    enabled: !!id && streamed !== id,
    refetchInterval: (q) => (terminal(q.state.data) ? false : 1800),
  });
  useEffect(() => {
    if (!id || !("EventSource" in window)) return;
    const source = api.jobEvents(id);
    source.onopen = () => {
      setStreamed(id);
      void client.cancelQueries({ queryKey: ["job", id] });
    };
    source.onmessage = (e) => {
      try {
        const job = JSON.parse(e.data) as Job;
        if (typeof job.status !== "string" || !Array.isArray(job.log)) return;
        client.setQueryData(["job", id], job);
        if (terminal(job)) source.close();
      } catch {
        /* Ignore non-JSON keep-alives. */
      }
    };
    source.onerror = () => {
      source.close();
      setStreamed("");
    };
    return () => source.close();
  }, [id, client]);
  return query;
}
