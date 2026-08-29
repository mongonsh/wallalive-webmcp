import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CutRoom — Direct the intention",
  description:
    "A human-agent storyboard studio powered by WebMCP. Lock what matters, then let your agent find the coverage.",
  applicationName: "CutRoom",
  keywords: ["WebMCP", "storyboard", "filmmaking", "browser agent", "creative tools"],
  authors: [{ name: "CutRoom" }],
  openGraph: {
    type: "website",
    title: "CutRoom — Direct the intention",
    description: "Lock what matters. Let your browser agent find the coverage.",
    siteName: "CutRoom",
    images: [{ url: "/cutroom-og.png", width: 1536, height: 1024, alt: "CutRoom paper-cut laundromat storyboard" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "CutRoom — A WebMCP storyboard studio",
    description: "Direct the intention. Let the agent find the coverage.",
    images: ["/cutroom-og.png"],
  },
  icons: {
    icon: "/cutroom-og.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
