import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hands Free PDF Reader",
  description: "입모양과 버튼으로 페이지를 넘길 수 있는 핸즈프리 PDF 리더",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
