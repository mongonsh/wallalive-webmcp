import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://wallalive-webmcp.mungunshagai-tb.chatgpt.site"),
  title: "WallAlive — Draw it. Wake it. Play.",
  description:
    "Turn a child's wall drawing into an animated character that lives in the room, directed by a browser agent through WebMCP.",
  applicationName: "WallAlive",
  keywords: ["WebMCP", "WebXR", "augmented reality", "children's drawings", "browser agent"],
  authors: [{ name: "WallAlive" }],
  openGraph: {
    title: "WallAlive — Draw it. Wake it. Play.",
    description: "A child-safe AR playground where approved drawings become 3D characters directed through WebMCP.",
    type: "website",
    url: "/",
    siteName: "WallAlive",
    images: [{ url: "/wallalive-og-v27.png", width: 1536, height: 1024, alt: "A coral crayon creature lifting off graph paper as a tactile 3D WallAlive character" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "WallAlive — Draw it. Wake it. Play.",
    description: "Turn an approved wall drawing into a 3D character a browser agent can animate through WebMCP.",
    images: ["/wallalive-og-v27.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
