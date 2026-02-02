import { useState } from 'react'
import AnalyzeTab from './components/AnalyzeTab'
import QuestionsTab from './components/QuestionsTab'

type TabType = 'analyze' | 'about'

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('analyze')

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-2xl font-bold text-gray-900">
            GRADING-LLM
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Measure LLM consistency when grading statements across binary, ternary, quaternary, and continuous scales
          </p>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="max-w-7xl mx-auto px-4 mt-6">
        <div className="flex space-x-1 border-b border-gray-200">
          <button
            className={`tab-button ${activeTab === 'analyze' ? 'active' : ''}`}
            onClick={() => setActiveTab('analyze')}
          >
            Analyze Statement
          </button>
          <button
            className={`tab-button ${activeTab === 'about' ? 'active' : ''}`}
            onClick={() => setActiveTab('about')}
          >
            Documentation
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'analyze' ? <AnalyzeTab /> : <QuestionsTab />}
      </main>
    </div>
  )
}

export default App
