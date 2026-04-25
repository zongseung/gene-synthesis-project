import { CdssConsole } from "@/components/platform/console";

export const metadata = {
  title: "CDSS Console — AKMPP",
};

export default function PlatformPage() {
  return (
    <div className="flex flex-1 flex-col">
      <CdssConsole />
    </div>
  );
}
