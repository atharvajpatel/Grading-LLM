import { useState, useEffect } from 'react'
import { getQuestions, Question, QuestionMode } from '../api/client'

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
      <div className="card">
        <p className="panel-title mb-1">Methodology</p>
        <h2 className="section-title mb-4">1600 Evaluations</h2>

        <div className="prose prose-sm max-w-none text-mute space-y-4">
          <p>
            This tool measures how consistently an LLM grades statements when using different
            grading scales. Each analysis performs <strong className="text-ink">1600 total evaluations</strong>:
          </p>

          <div className="bg-white rounded-ctl p-4 border border-hair">
            <div className="text-center text-lg font-mono text-ink mb-2">
              20 questions × 20 samples × 4 scales = <span className="text-gold font-bold">1600 evaluations</span>
            </div>
            <div className="text-center text-sm text-mute">
              (~80 API calls, batching 20 questions per call)
            </div>
          </div>

          <h3 className="section-title mt-6">How It Works</h3>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white rounded-ctl p-4 border border-hair">
              <h4 className="font-semibold text-ink mb-2">1. Questions (20)</h4>
              <p className="text-sm">
                A fixed set of 20 semantic questions that probe different aspects of the statement
                (factuality, clarity, objectivity, etc.). These form a 20-dimensional embedding.
              </p>
            </div>

            <div className="bg-white rounded-ctl p-4 border border-hair">
              <h4 className="font-semibold text-ink mb-2">2. Samples (20)</h4>
              <p className="text-sm">
                Each set of 20 questions is asked <strong className="text-ink">20 times with identical prompts</strong>.
                Same statement, same questions, same instructions. This measures consistency:
                does the model give the same answer when asked the same thing?
              </p>
            </div>

            <div className="bg-white rounded-ctl p-4 border border-hair">
              <h4 className="font-semibold text-ink mb-2">3. Grading Scales (4)</h4>
              <p className="text-sm">
                The same sampling is repeated across 4 different grading granularities:
              </p>
              <ul className="text-sm mt-2 space-y-1">
                <li><span className="font-mono text-scale-binary">Binary</span>: 0 or 1</li>
                <li><span className="font-mono text-scale-ternary">Ternary</span>: 0, 0.5, or 1</li>
                <li><span className="font-mono text-scale-quaternary">Quaternary</span>: 0, 0.33, 0.66, or 1</li>
                <li><span className="font-mono text-scale-continuous">Continuous</span>: any value from 0 to 1</li>
              </ul>
            </div>

            <div className="bg-white rounded-ctl p-4 border border-hair">
              <h4 className="font-semibold text-ink mb-2">4. Analysis</h4>
              <p className="text-sm">
                For each scale, we compute variance and consistency across the 20 samples.
                PCA reduces the 20-dimensional embeddings to 3D for visualization.
                Tight clusters = consistent, spread = uncertain.
              </p>
            </div>
          </div>

          <h3 className="section-title mt-6">Why This Matters</h3>
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
          <div>
            <p className="panel-title mb-1">Questions</p>
            <h2 className="section-title">20 Semantic Questions</h2>
          </div>

          {/* Mode Toggle */}
          <div className="flex rounded-ctl border border-hair overflow-hidden">
            <button
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                questionMode === 'mech'
                  ? 'bg-gold text-white'
                  : 'bg-transparent text-mute border-hair hover:bg-cream'
              }`}
              onClick={() => setQuestionMode('mech')}
            >
              Mechanistic
            </button>
            <button
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                questionMode === 'interp'
                  ? 'bg-gold text-white'
                  : 'bg-transparent text-mute border-hair hover:bg-cream'
              }`}
              onClick={() => setQuestionMode('interp')}
            >
              Interpretability
            </button>
          </div>
        </div>

        <p className="text-mute mb-6">
          {questionMode === 'mech' ? (
            <>
              <strong className="text-gold font-semibold">Mechanistic questions</strong> probe explicit
              linguistic and semantic features (named entities, causality, temporal references, etc.).
            </>
          ) : (
            <>
              <strong className="text-gold font-semibold">Interpretability questions</strong> probe implicit
              meaning, inference, and social understanding (unstated judgments, implied tension, etc.).
            </>
          )}
        </p>

        {loading ? (
          <div className="text-center py-8 text-mute">Loading questions...</div>
        ) : (
          <div className="space-y-2">
            {families.map((family) => {
              const familyQuestions = questions.filter((q) => q.family === family)
              const isExpanded = expandedFamilies.has(family)

              return (
                <div key={family} className="border border-hair rounded-ctl overflow-hidden">
                  <button
                    className="w-full flex items-center justify-between px-4 py-3 bg-cream hover:bg-hair transition-colors"
                    onClick={() => toggleFamily(family)}
                  >
                    <span className="font-medium text-ink">
                      {formatFamilyName(family)}
                    </span>
                    <span className="text-xs font-semibold uppercase tracking-wider text-gold">
                      {isExpanded ? 'Hide' : 'Show'}
                    </span>
                  </button>

                  {isExpanded && (
                    <div className="p-4 space-y-3 bg-white">
                      {familyQuestions.map((q) => (
                        <div key={q.id} className="text-sm">
                          <p className="text-ink mb-1">
                            <span className="font-mono text-xs text-mute mr-2">
                              {q.id}
                            </span>
                            {q.question}
                          </p>
                          {q.minimal_pairs && (
                            <div className="ml-4 text-xs text-mute space-y-1">
                              <p>
                                <span className="text-gold font-semibold">+</span> {q.minimal_pairs.positive}
                              </p>
                              <p>
                                <span className="text-mute font-semibold">−</span> {q.minimal_pairs.negative}
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

      {/* 10 Statement Archetypes */}
      <div className="card">
        <p className="panel-title mb-1">Corpus</p>
        <h2 className="section-title mb-4">10 Statement Archetypes</h2>

        <p className="text-mute mb-4">
          The batch experiments on the Research, Logprobs, and Presentation tabs grade these same
          10 texts. They&apos;re hand-picked archetypes designed to span the difficulty surface
          (factual, evaluative, ambiguous, clinical, abstract, narrative, technical), not a
          representative corpus. Source: <code className="text-xs bg-cream border border-hair px-1 py-0.5 rounded">data/batch_input_10.jsonl</code>.
        </p>

        <div className="space-y-2">
          {[
            { id: 'factual_simple',         tag: 'Factual',     color: 'blue',    text: 'The Eiffel Tower was completed in 1889 and stands 330 meters tall in Paris, France.' },
            { id: 'sentiment_positive',     tag: 'Evaluative+', color: 'green',   text: 'The community garden transformed our neighborhood — strangers became friends, children learned patience, and even the most skeptical residents admitted the fresh tomatoes were worth the effort.' },
            { id: 'sentiment_negative',     tag: 'Evaluative−', color: 'red',     text: 'The project was a catastrophic failure that wasted millions in taxpayer money, displaced vulnerable families, and left behind nothing but empty promises and crumbling infrastructure.' },
            { id: 'ambiguous',              tag: 'Ambiguous',   color: 'amber',   text: 'She said she was fine with the decision, but everyone in the room knew that wasn’t the whole story.' },
            { id: 'medical_clinical',       tag: 'Clinical',    color: 'pink',    text: 'Patient presents with acute onset chest pain radiating to the left arm, diaphoresis, and shortness of breath. ECG shows ST-elevation in leads II, III, and aVF. Troponin levels pending.' },
            { id: 'negation_heavy',         tag: 'Negation',    color: 'purple',  text: 'The study found no significant evidence that the treatment was neither ineffective nor harmful, leaving researchers unable to draw any definitive conclusions.' },
            { id: 'imperative_action',      tag: 'Imperative',  color: 'orange',  text: 'Immediately evacuate the building through the nearest emergency exit. Do not use the elevators. Proceed to the designated assembly point and wait for further instructions from emergency personnel.' },
            { id: 'abstract_philosophical', tag: 'Abstract',    color: 'indigo',  text: 'Consciousness may be less like a light switch and more like a dimmer — not something that is simply present or absent, but something that exists in degrees across a spectrum we barely understand.' },
            { id: 'narrative_paragraph',    tag: 'Narrative',   color: 'rose',    text: 'Maria had been working at the clinic for twelve years when the new director arrived. Within weeks, he had restructured the scheduling system, fired two senior nurses, and implemented a policy requiring all staff to log their breaks. Morale plummeted. Patients noticed the tension. By March, three more staff members had quietly submitted their resignations.' },
            { id: 'technical_ml',           tag: 'Technical',   color: 'cyan',    text: 'The transformer architecture uses multi-head self-attention to compute weighted representations of input tokens, where attention weights are derived from scaled dot-product similarity between query and key projections, enabling the model to capture long-range dependencies without recurrence.' },
          ].map((t, i) => (
            <div key={t.id} className="border border-hair rounded-ctl p-3 hover:border-gold transition-colors">
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 w-7 h-7 rounded-full bg-cream text-ink flex items-center justify-center text-xs font-bold font-mono">
                  {i + 1}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <code className="text-xs font-mono text-ink bg-cream border border-hair px-1.5 py-0.5 rounded">{t.id}</code>
                    <span className="chip-accent text-[10px] font-semibold uppercase tracking-wider">
                      {t.tag}
                    </span>
                  </div>
                  <p className="text-sm text-ink leading-relaxed">{t.text}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="callout mt-5 text-sm">
          <p className="font-semibold mb-1">N = 10 is a probe, not a benchmark.</p>
          <p>
            With 10 texts × 20 questions = 200 paired data points per scale, the Wilcoxon
            signed-rank tests on scale-degradation are well-powered (p &lt; 0.001 binary→quaternary,
            replicated across two independent 16,000-grade runs at Pearson r ≥ 0.995). What 10
            texts can&apos;t support is per-text difficulty generalization (e.g.,{' '}
            <em>&ldquo;ambiguous text is the hardest type in general&rdquo;</em>) or precise per-family
            percentages. Scaling to a held-out per-domain set is the natural next experiment; see
            the Limitations card below.
          </p>
        </div>
      </div>

      {/* How PCA Works */}
      <div className="card">
        <p className="panel-title mb-1">Method</p>
        <h2 className="section-title mb-4">How PCA Works</h2>

        <div className="prose prose-sm max-w-none text-mute space-y-4">
          <p>
            <strong className="text-ink">Principal Component Analysis (PCA)</strong> reduces the 20-dimensional embedding
            (one dimension per question) to 3 dimensions for visualization.
          </p>

          <h3 className="section-title mt-6">What the Plot Shows</h3>
          <ul className="list-disc list-inside space-y-1">
            <li><strong className="text-ink">80 points</strong>: 20 samples × 4 scales</li>
            <li><strong className="text-ink">Colors</strong>: <span className="text-scale-binary">Binary</span>, <span className="text-scale-ternary">Ternary</span>, <span className="text-scale-quaternary">Quaternary</span>, <span className="text-scale-continuous">Continuous</span></li>
            <li><strong className="text-ink">Tight clusters</strong>: Model is consistent on that scale</li>
            <li><strong className="text-ink">Spread points</strong>: Model is uncertain or inconsistent</li>
          </ul>

          <h3 className="section-title mt-6">Interpreting Loadings</h3>
          <p>
            Each principal component is a linear combination of questions. High-loading questions
            "drive" variance in that dimension. If a question has a high absolute loading on PC1,
            it means responses to that question vary the most across samples.
          </p>

          <h3 className="section-title mt-6">Explained Variance</h3>
          <p>
            The percentage shown for each PC indicates how much of the total variance it captures.
            PC1 captures the most, PC2 the second most, etc. Together, PC1-3 typically capture
            60-80% of total variance.
          </p>
        </div>
      </div>

      {/* Limitations */}
      <div className="card">
        <p className="panel-title mb-1">Caveats</p>
        <h2 className="section-title mb-4">Limitations</h2>

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
            <div key={i} className="callout p-3">
              <p className="font-medium text-ink">{i + 1}. {item.title}</p>
              <p className="text-sm text-mute mt-1">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* What's Next */}
      <div className="card">
        <p className="panel-title mb-1">Roadmap</p>
        <h2 className="section-title mb-4">What's Next</h2>

        <div className="space-y-2">
          {/* Key research question - highlighted */}
          <div className="callout flex items-start space-x-3 p-3">
            <div className="w-6 h-6 rounded-full bg-gold text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
              1
            </div>
            <span className="text-ink">
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
            <div key={i} className="flex items-center space-x-3 p-2 hover:bg-cream rounded">
              <div className="chip-accent w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium">
                {i + 2}
              </div>
              <span className="text-ink">{item}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
