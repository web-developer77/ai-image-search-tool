import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AI Image Search Tool',
  description: 'This is AI Image Search Tool',
  generator: '',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
