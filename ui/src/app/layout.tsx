import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'GenAI Reliability Framework',
  description: 'Domain-grounded LLM evaluation harness for regulated industries. OSFI E-23 aligned.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
