import { useState, useEffect } from 'react'
import { getQuestions, Question, QuestionMode } from '../api/client'
import { ChevronDown, ChevronRight, HelpCircle, AlertTriangle, Rocket, Calculator } from 'lucide-react'

export default function QuestionsTab() {
  const [questions, setQuestions] = useState<Question[]>([])
  const [expandedFamilies, setExpandedFamilies] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(true)
  const [questionMode, setQuestionMode] = useState<QuestionMode>('mech')

  useEffect(() => {
    setLoading(true)
    getQuestions(questionMode)
      .then(setQuestions)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [questionMode])

  const families = [...new Set(questions.map((q) => q.family))].sort()

  const toggleFamily = (family: string) => {
    const newExpanded = new Set(expandedFamilies)
    if (newExpanded.has(family)) {
      newExpanded.delete(family)
    } else {
      newExpanded.add(family)
    }
    setExpandedFamilies(newExpanded)
  }

  const formatFamilyName = (family: string) =>
    family.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

  return (
    <div className="space-y-8">
      {/* Methodology Section */}
      <div className="card bg-gradient-to-br from-blue-50 to-indigo-50">
        <div className="flex items-center space-x-2 mb-4">
          <Calculator className="w-6 h-6 text-blue-600" />
          <h2 className="text-xl font-bold text-gray-900">Methodology: 1600 Evaluations</h2>
        </div>

        <div className="prose prose-sm max-w-none text-gray-700 space-y-4">
          <p>
            This tool measures how consistently an LLM grades statements when using different
            grading scales. Each analysis performs <strong>1600 total evaluations</strong>:
          </p>

          <div className="bg-white rounded-lg p-4 border border-blue-200">
            <div className="text-center text-lg font-mono text-gray-800 mb-2">
              20 questions × 20 samples × 4 scales = <span className="text-blue-600 font-bold">1600 evaluations</span>
            </div>
            <div className="text-center text-sm text-gray-500">
              (~80 API calls, batching 20 questions per call)
            </div>
          </div>

          <h3 className="text-lg font-semibold text-gray-800 mt-6">How It Works</h3>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <h4 className="font-semibold text-gray-800 mb-2">1. Questions (20)</h4>
              <p className="text-sm">
                A fixed set of 20 semantic questions that probe different aspects of the statement
                (factuality, clarity, objectivity, etc.). These form a 20-dimensional embedding.
              </p>
            </div>

            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <h4 className="font-semibold text-gray-800 mb-2">2. Samples (20)</h4>
              <p className="text-sm">
                Each set of 20 questions is asked <strong>20 times with identical prompts</strong>.
                Same statement, same questions, same instructions. This measures consistency:
                does the model give the same answer when asked the same thing?
              </p>
            </div>

            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <h4 className="font-semibold text-gray-800 mb-2">3. Grading Scales (4)</h4>
              <p className="text-sm">
                The same sampling is repeated across 4 different grading granularities:
              </p>
              <ul className="text-sm mt-2 space-y-1">
                <li><span className="font-mono text-blue-600">Binary</span>: 0 or 1</li>
                <li><span className="font-mono text-green-600">Ternary</span>: 0, 0.5, or 1</li>
                <li><span className="font-mono text-orange-600">Quaternary</span>: 0, 0.33, 0.66, or 1</li>
                <li><span className="font-mono text-pink-600">Continuous</span>: any value from 0 to 1</li>
              </ul>
            </div>

            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <h4 className="font-semibold text-gray-800 mb-2">4. Analysis</h4>
              <p className="text-sm">
                For each scale, we compute variance and consistency across the 20 samples.
                PCA reduces the 20-dimensional embeddings to 3D for visualization.
                Tight clusters = consistent, spread = uncertain.
              </p>
            </div>
          </div>

          <h3 className="text-lg font-semibold text-gray-800 mt-6">Why This Matters</h3>
          <p>
            When using LLMs as evaluators or embedders, understanding their consistency is crucial.
            A model that gives different answers to the same question each time it's asked is
            unreliable for grading or ranking tasks. This tool helps quantify that reliability
            across different levels of granularity.
          </p>
        </div>
      </div>

      {/* Questions Section */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">
            20 Semantic Questions
          </h2>

          {/* Mode Toggle */}
          <div className="flex rounded-lg border border-gray-300 overflow-hidden">
            <button
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                questionMode === 'mech'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
              onClick={() => setQuestionMode('mech')}
            >
              Mechanistic
            </button>
            <button
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                questionMode === 'interp'
                  ? 'bg-purple-500 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
              onClick={() => setQuestionMode('interp')}
            >
              Interpretability
            </button>
          </div>
        </div>

        <p className="text-gray-600 mb-6">
          {questionMode === 'mech' ? (
            <>
              <strong className="text-blue-600">Mechanistic questions</strong> probe explicit
              linguistic and semantic features (named entities, causality, temporal references, etc.).
            </>
          ) : (
            <>
              <strong className="text-purple-600">Interpretability questions</strong> probe implicit
              meaning, inference, and social understanding (unstated judgments, implied tension, etc.).
            </>
          )}
        </p>

        {loading ? (
          <div className="text-center py-8 text-gray-500">Loading questions...</div>
        ) : (
          <div className="space-y-2">
            {families.map((family) => {
              const familyQuestions = questions.filter((q) => q.family === family)
              const isExpanded = expandedFamilies.has(family)

              return (
                <div key={family} className="border border-gray-200 rounded-lg overflow-hidden">
                  <button
                    className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors"
                    onClick={() => toggleFamily(family)}
                  >
                    <span className="font-medium text-gray-800">
                      {formatFamilyName(family)}
                    </span>
                    {isExpanded ? (
                      <ChevronDown className="w-5 h-5 text-gray-500" />
                    ) : (
                      <ChevronRight className="w-5 h-5 text-gray-500" />
                    )}
                  </button>

                  {isExpanded && (
                    <div className="p-4 space-y-3 bg-white">
                      {familyQuestions.map((q) => (
                        <div key={q.id} className="text-sm">
                          <p className="text-gray-800 mb-1">
                            <span className="font-mono text-xs text-gray-400 mr-2">
                              {q.id}
                            </span>
                            {q.question}
                          </p>
                          {q.minimal_pairs && (
                            <div className="ml-4 text-xs text-gray-500 space-y-1">
                              <p>
                                <span className="text-green-600">+</span> {q.minimal_pairs.positive}
                              </p>
                              <p>
                                <span className="text-red-600">−</span> {q.minimal_pairs.negative}
                              </p>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* How PCA Works */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <HelpCircle className="w-6 h-6 text-blue-600" />
          <h2 className="text-xl font-bold text-gray-900">How PCA Works</h2>
        </div>

        <div className="prose prose-sm max-w-none text-gray-700 space-y-4">
          <p>
            <strong>Principal Component Analysis (PCA)</strong> reduces the 20-dimensional embedding
            (one dimension per question) to 3 dimensions for visualization.
          </p>

          <h3 className="text-lg font-semibold text-gray-800 mt-6">What the Plot Shows</h3>
          <ul className="list-disc list-inside space-y-1">
            <li><strong>80 points</strong>: 20 samples × 4 scales</li>
            <li><strong>Colors</strong>: Binary (blue), Ternary (green), Quaternary (orange), Continuous (pink)</li>
            <li><strong>Tight clusters</strong>: Model is consistent on that scale</li>
            <li><strong>Spread points</strong>: Model is uncertain or inconsistent</li>
          </ul>

          <h3 className="text-lg font-semibold text-gray-800 mt-6">Interpreting Loadings</h3>
          <p>
            Each principal component is a linear combination of questions. High-loading questions
            "drive" variance in that dimension. If a question has a high absolute loading on PC1,
            it means responses to that question vary the most across samples.
          </p>

          <h3 className="text-lg font-semibold text-gray-800 mt-6">Explained Variance</h3>
          <p>
            The percentage shown for each PC indicates how much of the total variance it captures.
            PC1 captures the most, PC2 the second most, etc. Together, PC1-3 typically capture
            60-80% of total variance.
          </p>
        </div>
      </div>

      {/* Limitations */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <AlertTriangle className="w-6 h-6 text-amber-600" />
          <h2 className="text-xl font-bold text-gray-900">Limitations</h2>
        </div>

        <div className="space-y-3">
          {[
            {
              title: 'Temperature=0 doesn\'t guarantee determinism',
              desc: 'OpenAI models can still produce varying outputs even at temperature 0, especially for nuanced questions.',
            },
            {
              title: '20 questions may miss nuances',
              desc: 'Trade-off between coverage and efficiency. Some semantic dimensions may not be captured.',
            },
            {
              title: 'Binary grounding',
              desc: 'Questions are designed for binary answers and may feel forced on continuous scales.',
            },
            {
              title: 'Single model',
              desc: 'Results are specific to gpt-4o-mini. Other models may show different patterns.',
            },
            {
              title: 'Sample size',
              desc: '20 samples per scale may not capture the full distribution of model responses.',
            },
          ].map((item, i) => (
            <div key={i} className="p-3 bg-amber-50 rounded-lg">
              <p className="font-medium text-gray-800">{i + 1}. {item.title}</p>
              <p className="text-sm text-gray-600 mt-1">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* What's Next */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <Rocket className="w-6 h-6 text-purple-600" />
          <h2 className="text-xl font-bold text-gray-900">What's Next</h2>
        </div>

        <div className="space-y-2">
          {/* Key research question - highlighted */}
          <div className="flex items-start space-x-3 p-3 bg-purple-50 rounded-lg border border-purple-200">
            <div className="w-6 h-6 rounded-full bg-purple-500 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
              1
            </div>
            <span className="text-gray-800">
              <strong>Once we figure out that boundary, how do we push LLMs to be consistent?
              Do they ever surpass calibration and consistency compared to human inter-annotators?</strong>
            </span>
          </div>

          {[
            'Compare across different models (GPT-4, Claude, Llama, etc.)',
            'Increase sample size for better statistical power',
            'Add confidence intervals on metrics',
            'Test on domain-specific statements (legal, medical, technical)',
            'Bring your own question set',
            'Make this a simple library to use',
            'Longitudinal analysis: track consistency over model versions',
            'Correlation with downstream task performance',
          ].map((item, i) => (
            <div key={i} className="flex items-center space-x-3 p-2 hover:bg-gray-50 rounded">
              <div className="w-6 h-6 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center text-xs font-medium">
                {i + 2}
              </div>
              <span className="text-gray-700">{item}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
