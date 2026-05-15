import { useState } from 'react'
import AnalyzeTab from './components/AnalyzeTab'
import QuestionsTab from './components/QuestionsTab'
import BatchAnalysisTab from './components/BatchAnalysisTab'
import ResearchTab from './components/ResearchTab'
import LogprobsTab from './components/LogprobsTab'
import ComparisonTab from './components/ComparisonTab'
import NotebookTab from './components/NotebookTab'
import PresentationTab from './components/PresentationTab'

type TabType = 'presentation' | 'analyze' | 'about' | 'batch' | 'research' | 'logprobs' | 'comparison' | 'notebook'

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('presentation')

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
            className={`tab-button ${activeTab === 'presentation' ? 'active' : ''}`}
            onClick={() => setActiveTab('presentation')}
          >
            Presentation
          </button>
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
          <button
            className={`tab-button ${activeTab === 'batch' ? 'active' : ''}`}
            onClick={() => setActiveTab('batch')}
          >
            Batch Analysis
          </button>
          <button
            className={`tab-button ${activeTab === 'research' ? 'active' : ''}`}
            onClick={() => setActiveTab('research')}
          >
            Research
          </button>
          <button
            className={`tab-button ${activeTab === 'logprobs' ? 'active' : ''}`}
            onClick={() => setActiveTab('logprobs')}
          >
            Logprobs
          </button>
          <button
            className={`tab-button ${activeTab === 'comparison' ? 'active' : ''}`}
            onClick={() => setActiveTab('comparison')}
          >
            Comparison
          </button>
          <button
            className={`tab-button ${activeTab === 'notebook' ? 'active' : ''}`}
            onClick={() => setActiveTab('notebook')}
          >
            Notebook
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'presentation' && <PresentationTab />}
        {activeTab === 'analyze' && <AnalyzeTab />}
        {activeTab === 'about' && <QuestionsTab />}
        {activeTab === 'batch' && <BatchAnalysisTab />}
        {activeTab === 'research' && <ResearchTab />}
        {activeTab === 'logprobs' && <LogprobsTab />}
        {activeTab === 'comparison' && <ComparisonTab />}
        {activeTab === 'notebook' && <NotebookTab />}
      </main>
    </div>
  )
}

export default App
