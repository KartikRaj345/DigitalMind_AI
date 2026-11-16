import { useCallback, useState } from "react";

import { AppShell } from "./components/AppShell";
import { EmptyState, PageHeader } from "./components/ui";
import { useRecordId, useRoute, type Route } from "./lib/router";
import { DashboardView } from "./pages/DashboardView";
import { RecordsView } from "./pages/RecordsView";
import { RecordDetailView } from "./pages/RecordDetailView";
import { GraphView } from "./pages/GraphView";
import { TimelineView } from "./pages/TimelineView";
import { RiskView } from "./pages/RiskView";
import { AskView } from "./pages/AskView";
import { InsightsView } from "./pages/InsightsView";
import { ImportView } from "./pages/ImportView";
import { SettingsView } from "./pages/SettingsView";

export default function App() {
  const route = useRoute();
  // Bumped whenever data changes, so views that read the API on mount refetch.
  const [revision, setRevision] = useState(0);
  const onImported = useCallback(() => setRevision((value) => value + 1), []);
  const recordId = useRecordId(route);

  return (
    <AppShell>
      <div key={`${route.name}-${route.hash}-${revision}`}>
        <View route={route} recordId={recordId} onImported={onImported} />
      </div>
    </AppShell>
  );
}

function View({
  route,
  recordId,
  onImported,
}: {
  route: Route;
  recordId: number | null;
  onImported: () => void;
}) {
  switch (route.name) {
    case "records":
      return <RecordsView />;
    case "record":
      return recordId === null ? (
        <>
          <PageHeader title="Record not found" />
          <EmptyState title="No record id in the address">
            <a href="#/records" className="font-medium text-teal-800 underline">
              Go back to the records inbox
            </a>
          </EmptyState>
        </>
      ) : (
        <RecordDetailView recordId={recordId} spanParam={route.query.get("span")} />
      );
    case "graph":
      return <GraphView focusParam={route.query.get("focus")} />;
    case "timeline":
      return <TimelineView />;
    case "risk":
      return <RiskView />;
    case "ask":
      return <AskView />;
    case "insights":
      return <InsightsView />;
    case "import":
      return <ImportView onImported={onImported} />;
    case "settings":
      return <SettingsView />;
    case "dashboard":
    default:
      return <DashboardView />;
  }
}
