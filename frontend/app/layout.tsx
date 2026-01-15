import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Aion Analytics",
  description: "Stock analysis and prediction platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
