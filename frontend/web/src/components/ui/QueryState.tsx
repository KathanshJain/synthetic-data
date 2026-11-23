import { Info } from "lucide-react";
import { DataUnavailable } from "../../api/client";
import { Empty } from "./Primitives";
export function QueryState({
  query,
  message,
}: {
  query: { isLoading: boolean; error: Error | null; refetch: () => unknown };
  message: string;
}) {
  if (query.isLoading)
    return <div className="skeleton chart-skeleton" role="status" aria-label="Loading chart" />;
  if (query.error && !(query.error instanceof DataUnavailable))
    return (
      <Empty
        title="We couldn’t load this view"
        message="Check the connection and try again. The rest of the page is still available."
        retry={() => query.refetch()}
      />
    );
  return <Empty message={message} />;
}
export function Control({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string | number;
  options: (string | number | { value: string | number; label: string })[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="control">
      <span>{label}</span>
      <select
        aria-label={label}
        value={value}
        disabled={!options.length}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.length ? (
          options.map((o) => {
            const { value, label } = typeof o === "object" ? o : { value: o, label: String(o) };
            return (
              <option key={value} value={value}>
                {label}
              </option>
            );
          })
        ) : (
          <option value="">Not available</option>
        )}
      </select>
    </label>
  );
}
export function ContextNote({ children }: { children: React.ReactNode }) {
  return (
    <p className="context-note">
      <Info size={15} />
      {children}
    </p>
  );
}
