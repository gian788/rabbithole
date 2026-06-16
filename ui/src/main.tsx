import React from 'react'
import ReactDOM from 'react-dom/client'
import { RagChat } from './components/RagChat'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
      <RagChat
        apiUrl={import.meta.env.VITE_API_URL ?? 'http://localhost:8000'}
        placeholder="Ask a question..."
        className="w-full max-w-2xl h-[600px]"
      />
    </div>
  </React.StrictMode>
)
