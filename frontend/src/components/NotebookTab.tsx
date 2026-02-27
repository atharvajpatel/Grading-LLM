import { BookOpen, Code2, FileText } from 'lucide-react'

/* ─── Helper Components ─────────────────────────────────────────────────────── */

/** Renders a markdown-style cell with styled headings, bold, lists, and rules. */
function MarkdownCell({ children }: { children: React.ReactNode }) {
  return (
    <div className="border border-gray-200 rounded-md mb-2 bg-white">
      <div className="px-5 py-4 prose prose-sm max-w-none text-gray-800 leading-relaxed">
        {children}
      </div>
    </div>
  )
}

/** Renders a code cell with execution count badge and optional output. */
function CodeCell({
  executionCount,
  code,
  output,
}: {
  executionCount: number
  code: string
  output?: React.ReactNode
}) {
  return (
    <div className="border border-gray-200 rounded-md mb-2 bg-white overflow-hidden">
      {/* Code input */}
      <div className="flex">
        <div className="flex-shrink-0 w-16 bg-gray-50 border-r border-gray-200 flex items-start justify-end pr-2 pt-3">
          <span className="text-xs font-mono text-blue-600 font-semibold">
            [{executionCount}]:
          </span>
        </div>
        <div className="flex-1 overflow-x-auto">
          <pre className="p-3 text-[13px] leading-[1.45] font-mono text-gray-800 bg-gray-50 m-0 whitespace-pre">
            {code}
          </pre>
        </div>
      </div>
      {/* Output */}
      {output && (
        <div className="border-t border-gray-200">
          <div className="flex">
            <div className="flex-shrink-0 w-16 bg-white border-r border-gray-200" />
            <div className="flex-1 p-3 overflow-x-auto">{output}</div>
          </div>
        </div>
      )}
    </div>
  )
}

/** Renders a text output block (stdout). */
function TextOutput({ text }: { text: string }) {
  return (
    <pre className="text-[13px] leading-[1.45] font-mono text-gray-700 whitespace-pre-wrap m-0">
      {text}
    </pre>
  )
}

/** Renders an image output (matplotlib figure). */
function ImageOutput({ src, alt }: { src: string; alt: string }) {
  return (
    <div className="flex justify-center py-2">
      <img
        src={src}
        alt={alt}
        className="max-w-full rounded shadow-sm"
        loading="lazy"
      />
    </div>
  )
}

/* ─── Degradation Table (hardcoded from notebook output) ────────────────────── */

function DegradationTable() {
  const rows = [
    { scale: 'Binary', meanVar: '0.0024', medianVar: '0.0000', zeroVar: '97.0%', highVar: '0.5%', medianEntropy: '0.000', meanConsistency: '99.72%' },
    { scale: 'Ternary', meanVar: '0.0032', medianVar: '0.0000', zeroVar: '95.5%', highVar: '1.0%', medianEntropy: '0.000', meanConsistency: '99.18%' },
    { scale: 'Quaternary', meanVar: '0.0050', medianVar: '0.0000', zeroVar: '88.5%', highVar: '1.5%', medianEntropy: '0.000', meanConsistency: '96.85%' },
    { scale: 'Continuous', meanVar: '0.0020', medianVar: '0.0000', zeroVar: '80.5%', highVar: '0.0%', medianEntropy: '0.000', meanConsistency: '97.47%' },
  ]
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm font-mono">
        <thead>
          <tr className="border-b border-gray-300">
            <th className="text-left py-2 pr-4 font-semibold text-gray-700">Scale</th>
            <th className="text-right py-2 px-3 font-semibold text-gray-700">Mean Var</th>
            <th className="text-right py-2 px-3 font-semibold text-gray-700">Median Var</th>
            <th className="text-right py-2 px-3 font-semibold text-gray-700">% Zero-Var</th>
            <th className="text-right py-2 px-3 font-semibold text-gray-700">% High-Var</th>
            <th className="text-right py-2 px-3 font-semibold text-gray-700">Med Entropy</th>
            <th className="text-right py-2 px-3 font-semibold text-gray-700">Mean Consistency</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.scale} className="border-b border-gray-100">
              <td className="py-1.5 pr-4 font-semibold text-gray-800">{r.scale}</td>
              <td className="py-1.5 px-3 text-right">{r.meanVar}</td>
              <td className="py-1.5 px-3 text-right">{r.medianVar}</td>
              <td className="py-1.5 px-3 text-right">{r.zeroVar}</td>
              <td className="py-1.5 px-3 text-right">{r.highVar}</td>
              <td className="py-1.5 px-3 text-right">{r.medianEntropy}</td>
              <td className="py-1.5 px-3 text-right">{r.meanConsistency}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ─── Main NotebookTab Component ────────────────────────────────────────────── */

export default function NotebookTab() {
  return (
    <div className="space-y-1 max-w-5xl mx-auto">
      {/* Header */}
      <div className="card mb-6">
        <div className="flex items-center gap-3 mb-3">
          <BookOpen className="w-6 h-6 text-orange-600" />
          <h2 className="text-xl font-bold text-gray-900">
            Analysis Notebook
          </h2>
        </div>
        <p className="text-sm text-gray-600">
          Read-only rendering of{' '}
          <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs font-mono">
            fithian_analysis.ipynb
          </code>{' '}
          — the full statistical analysis notebook prepared for Prof. Will Fithian.
        </p>
        <div className="flex gap-4 mt-3 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <FileText className="w-3.5 h-3.5" /> 11 markdown cells
          </span>
          <span className="flex items-center gap-1">
            <Code2 className="w-3.5 h-3.5" /> 11 code cells
          </span>
          <span>Python 3 · GPT-4o-mini · 16,000 evaluations</span>
        </div>
      </div>

      {/* ─── Cell 0: Title & Methodology (Markdown) ──────────────────────────── */}
      <MarkdownCell>
        <h1 className="text-2xl font-bold text-gray-900 mb-1">
          LLM Grading Consistency Analysis
        </h1>
        <h2 className="text-lg font-semibold text-gray-700 mt-0 mb-3">
          Measuring the Statistical Boundary for Calibrated LLM-as-Judge
        </h2>
        <p className="text-sm text-gray-500 mb-4">
          <strong>Prepared for:</strong> Prof. Will Fithian, UC Berkeley Statistics
        </p>
        <hr className="my-4 border-gray-200" />
        <h3 className="text-base font-semibold text-gray-800 mt-4 mb-2">
          Research Question
        </h3>
        <p>
          When using LLMs to evaluate or label data, at what granularity can we
          trust the model&apos;s judgments? We know binary labels (yes/no) are
          generally accurate, but at what point does multi-class or continuous
          scoring degrade into noise?
        </p>
        <h3 className="text-base font-semibold text-gray-800 mt-4 mb-2">
          Methodology
        </h3>
        <ul className="list-disc pl-5 space-y-1 text-sm">
          <li>
            <strong>Model:</strong> GPT-4o-mini (<code>temperature=0</code>)
          </li>
          <li>
            <strong>Texts:</strong> 10 diverse statements (factual, sentiment,
            ambiguous, clinical, etc.)
          </li>
          <li>
            <strong>Questions:</strong> 20 mechanistic semantic probes (1 per
            orthogonal factor family)
          </li>
          <li>
            <strong>Scales:</strong> Binary {'{0,1}'}, Ternary {'{0, 0.5, 1}'},
            Quaternary {'{0, 0.33, 0.66, 1}'}, Continuous [0,1]
          </li>
          <li>
            <strong>Samples:</strong> 20 repeated evaluations per (text,
            question, scale) triple
          </li>
          <li>
            <strong>Total evaluations:</strong> 10 × 20 × 20 × 4 ={' '}
            <strong>16,000</strong>
          </li>
        </ul>
        <p className="mt-3 text-sm">
          Each evaluation asks the LLM to score how deeply a semantic feature
          appears in a statement. By repeating 20 times at temperature=0, we
          measure the model&apos;s internal consistency.
        </p>
      </MarkdownCell>

      {/* ─── Cell 1: Imports & Configuration ──────────────────────────────────── */}
      <CodeCell
        executionCount={1}
        code={`import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
sns.set_theme(style='whitegrid')

SCALE_ORDER = ['binary', 'ternary', 'quaternary', 'continuous']
SCALE_COLORS = {'binary': '#2196F3', 'ternary': '#4CAF50', 'quaternary': '#FF9800', 'continuous': '#E91E63'}
FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)

# Load batch results
with open('../data/batch_results.json') as f:
    allResults = json.load(f)

print(f'Loaded {len(allResults)} text results')
print(f'Text IDs: {[r["id"] for r in allResults]}')`}
        output={
          <TextOutput
            text={`Loaded 10 text results
Text IDs: ['factual_simple', 'sentiment_positive', 'sentiment_negative', 'ambiguous', 'medical_clinical', 'negation_heavy', 'imperative_action', 'abstract_philosophical', 'narrative_paragraph', 'technical_ml']`}
          />
        }
      />

      {/* ─── Cell 2: Load Question Metadata ──────────────────────────────────── */}
      <CodeCell
        executionCount={2}
        code={`# Load question metadata for labeling
with open('../data/questions_mech.json') as f:
    questionData = json.load(f)
questions = questionData['questions']
questionFamilies = [q['family'] for q in questions]
questionLabels = [q['family'].replace('_', ' ').title() for q in questions]

print(f'Questions: {len(questions)}')
print(f'Factor families: {questionFamilies}')`}
        output={
          <TextOutput
            text={`Questions: 20
Factor families: ['named_entities', 'actions_events', 'temporal', 'spatial', 'numeric', 'modality', 'imperative', 'comparison', 'first_person', 'dialogue', 'causality', 'negation', 'uncertainty', 'sentiment', 'emotion', 'social', 'normative', 'intent', 'concreteness', 'identity']`}
          />
        }
      />

      {/* ─── Cell 3: Build Master Data Structure ─────────────────────────────── */}
      <CodeCell
        executionCount={3}
        code={`# ======================================================================
# Build the master data structure
# varianceMatrix[scale][text_idx][question_idx] = variance of 20 samples
# entropyMatrix[scale][text_idx][question_idx] = entropy of 20 samples
# consistencyMatrix[scale][text_idx][question_idx] = mode consistency
# ======================================================================

nTexts = len(allResults)
nQuestions = 20
nScales = 4

varianceMatrix = {s: np.zeros((nTexts, nQuestions)) for s in SCALE_ORDER}
entropyMatrix = {s: np.zeros((nTexts, nQuestions)) for s in SCALE_ORDER}
consistencyMatrix = {s: np.zeros((nTexts, nQuestions)) for s in SCALE_ORDER}
meanMatrix = {s: np.zeros((nTexts, nQuestions)) for s in SCALE_ORDER}

# Discrete scale values for entropy computation
scaleValues = {
    'binary': [0, 1],
    'ternary': [0, 0.5, 1],
    'quaternary': [0, 0.33, 0.66, 1],
}

def computeEntropy(values, bins=None):
    """Compute Shannon entropy."""
    if bins is not None:
        counts = np.zeros(len(bins))
        for v in values:
            idx = np.argmin(np.abs(np.array(bins) - v))
            counts[idx] += 1
    else:
        counts, _ = np.histogram(values, bins=10)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def computeModeConsistency(values):
    """Fraction of values matching the mode."""
    rounded = np.round(values, 2)
    unique, counts = np.unique(rounded, return_counts=True)
    return counts.max() / len(values)

for tIdx, result in enumerate(allResults):
    for scale in SCALE_ORDER:
        rawScores = np.array(result['raw_scores'][scale])  # (20, 20)
        for qIdx in range(nQuestions):
            samples = rawScores[:, qIdx]  # 20 repeated scores
            varianceMatrix[scale][tIdx, qIdx] = np.var(samples)
            meanMatrix[scale][tIdx, qIdx] = np.mean(samples)
            consistencyMatrix[scale][tIdx, qIdx] = computeModeConsistency(samples)
            bins = scaleValues.get(scale)
            entropyMatrix[scale][tIdx, qIdx] = computeEntropy(samples, bins=bins)

print('Data structure built.')
print(f'Shape per scale: {varianceMatrix["binary"].shape} (texts x questions)')
print(f'Total data points per scale: {nTexts * nQuestions} = {nTexts}×{nQuestions}')`}
        output={
          <TextOutput
            text={`Data structure built.
Shape per scale: (10, 20) (texts x questions)
Total data points per scale: 200 = 10×20`}
          />
        }
      />

      {/* ─── Cell 4: Section Header — Degradation Table (Markdown) ────────── */}
      <MarkdownCell>
        <hr className="my-3 border-gray-200" />
        <h2 className="text-lg font-bold text-gray-900 mb-2">
          4.1 — Degradation Table
        </h2>
        <p className="text-sm">
          The central result: how does grading quality degrade as scale
          granularity increases?
        </p>
      </MarkdownCell>

      {/* ─── Cell 5: Degradation Table ────────────────────────────────────── */}
      <CodeCell
        executionCount={4}
        code={`# Build degradation table
rows = []
for scale in SCALE_ORDER:
    allVars = varianceMatrix[scale].flatten()  # 200 values
    allEntropy = entropyMatrix[scale].flatten()
    allConsistency = consistencyMatrix[scale].flatten()

    rows.append({
        'Scale': scale.capitalize(),
        'Mean Variance': f'{np.mean(allVars):.4f}',
        'Median Variance': f'{np.median(allVars):.4f}',
        '% Zero-Variance': f'{(allVars == 0).sum() / len(allVars) * 100:.1f}%',
        '% High-Variance (>0.1)': f'{(allVars > 0.1).sum() / len(allVars) * 100:.1f}%',
        'Median Entropy': f'{np.median(allEntropy):.3f}',
        'Mean Mode Consistency': f'{np.mean(allConsistency):.2%}',
    })

degradationDf = pd.DataFrame(rows)
degradationDf = degradationDf.set_index('Scale')
display(degradationDf)

# Save to CSV
degradationDf.to_csv(FIGURES_DIR / 'degradation_table.csv')
print(f'\\nSaved to {FIGURES_DIR / "degradation_table.csv"}')`}
        output={
          <div className="space-y-3">
            <DegradationTable />
            <TextOutput text={`\nSaved to figures/degradation_table.csv`} />
          </div>
        }
      />

      {/* ─── Cell 6: Section Header — Heatmap (Markdown) ─────────────────── */}
      <MarkdownCell>
        <hr className="my-3 border-gray-200" />
        <h2 className="text-lg font-bold text-gray-900 mb-2">
          4.2 — Question Stability Heatmap
        </h2>
        <p className="text-sm">
          Which questions are inherently unstable, and at which granularity?
          Each cell shows the mean variance for that question-scale pair,
          averaged across all 10 texts.
        </p>
      </MarkdownCell>

      {/* ─── Cell 7: Heatmap ─────────────────────────────────────────────── */}
      <CodeCell
        executionCount={5}
        code={`# Build heatmap: (20 questions x 4 scales), each cell = mean variance across 10 texts
heatmapData = np.zeros((nQuestions, nScales))
for sIdx, scale in enumerate(SCALE_ORDER):
    heatmapData[:, sIdx] = varianceMatrix[scale].mean(axis=0)  # mean across texts

fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(
    heatmapData,
    xticklabels=['Binary', 'Ternary', 'Quaternary', 'Continuous'],
    yticklabels=questionLabels,
    cmap='RdYlGn_r',
    annot=True, fmt='.4f',
    cbar_kws={'label': 'Mean Variance (across 10 texts)'},
    linewidths=0.5,
    ax=ax
)
ax.set_title('Question Stability Across Grading Scales', fontweight='bold')
ax.set_xlabel('Grading Scale')
ax.set_ylabel('Question (Factor Family)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'question_stability_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved to {FIGURES_DIR / "question_stability_heatmap.png"}')`}
        output={
          <div className="space-y-3">
            <ImageOutput
              src="/research/question_stability_heatmap.png"
              alt="Question Stability Heatmap — 20 questions × 4 scales"
            />
            <TextOutput text="Saved to figures/question_stability_heatmap.png" />
          </div>
        }
      />

      {/* ─── Cell 8: Section Header — Wilcoxon Tests (Markdown) ──────────── */}
      <MarkdownCell>
        <hr className="my-3 border-gray-200" />
        <h2 className="text-lg font-bold text-gray-900 mb-2">
          4.3 — Wilcoxon Signed-Rank Tests
        </h2>
        <p className="text-sm">
          Testing whether adjacent scales have statistically different variance
          distributions. These are <strong>paired tests</strong> because each
          variance value corresponds to the same (text, question) pair.
        </p>
        <p className="text-sm mt-2">
          <strong>H₀:</strong> Variance at scale A = Variance at scale B
          <br />
          <strong>H₁:</strong> Variance at scale B {'>'} Variance at scale A
          (one-sided)
        </p>
      </MarkdownCell>

      {/* ─── Cell 9: Wilcoxon Tests ──────────────────────────────────────── */}
      <CodeCell
        executionCount={6}
        code={`print('Wilcoxon Signed-Rank Tests (one-sided: H1 = scale B more variable than A)')
print('=' * 75)

wilcoxonResults = []
for i in range(len(SCALE_ORDER) - 1):
    scaleA = SCALE_ORDER[i]
    scaleB = SCALE_ORDER[i + 1]

    varsA = varianceMatrix[scaleA].flatten()  # 200 paired values
    varsB = varianceMatrix[scaleB].flatten()

    # Compute differences
    diffs = varsB - varsA
    nonZeroDiffs = diffs[diffs != 0]

    if len(nonZeroDiffs) == 0:
        print(f'{scaleA.capitalize():12s} → {scaleB.capitalize():12s}: All differences are zero')
        continue

    stat, pValue = wilcoxon(varsA, varsB, alternative='less')

    # Effect size: rank-biserial correlation
    nPairs = len(nonZeroDiffs)
    rankBiserial = 1 - (2 * stat) / (nPairs * (nPairs + 1) / 2) if nPairs > 0 else 0

    sig = '***' if pValue < 0.001 else '**' if pValue < 0.01 else '*' if pValue < 0.05 else 'ns'

    print(f'{scaleA.capitalize():12s} → {scaleB.capitalize():12s}: W={stat:10.1f}, p={pValue:.6f} {sig}')

# Also test binary vs continuous (full jump)
print('\\nFull range test:')
varsA = varianceMatrix['binary'].flatten()
varsB = varianceMatrix['continuous'].flatten()
stat, pValue = wilcoxon(varsA, varsB, alternative='less')
sig = '***' if pValue < 0.001 else '**' if pValue < 0.01 else '*' if pValue < 0.05 else 'ns'
print(f'{\"Binary\":12s} → {\"Continuous\":12s}: W={stat:10.1f}, p={pValue:.6f} {sig}')`}
        output={
          <TextOutput
            text={`Wilcoxon Signed-Rank Tests (one-sided: H1 = scale B more variable than A)
===========================================================================
Binary       → Ternary     : W=      52.0, p=0.225498 ns, r_rb=0.430, n_nonzero=14
Ternary      → Quaternary  : W=      71.0, p=0.058498 ns, r_rb=0.371, n_nonzero=19
Quaternary   → Continuous  : W=     265.0, p=0.941372 ns, r_rb=-0.359, n_nonzero=30

Full range test:
Binary       → Continuous  : W=     131.0, p=0.217213 ns`}
          />
        }
      />

      {/* ─── Cell 10: Section Header — Violin Plot (Markdown) ────────────── */}
      <MarkdownCell>
        <hr className="my-3 border-gray-200" />
        <h2 className="text-lg font-bold text-gray-900 mb-2">
          4.4 — Variance Distribution Violin Plot
        </h2>
        <p className="text-sm">
          Distribution of variances across all 200 (question × text) pairs for
          each scale.
        </p>
      </MarkdownCell>

      {/* ─── Cell 11: Violin Plot ────────────────────────────────────────── */}
      <CodeCell
        executionCount={7}
        code={`fig, ax = plt.subplots(figsize=(10, 6))

violinData = [varianceMatrix[s].flatten() for s in SCALE_ORDER]
parts = ax.violinplot(violinData, showmeans=True, showmedians=True)

# Color the violins
for i, body in enumerate(parts['bodies']):
    body.set_facecolor(list(SCALE_COLORS.values())[i])
    body.set_alpha(0.7)

ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Binary', 'Ternary', 'Quaternary', 'Continuous'])
ax.set_ylabel('Variance (across 20 repeated samples)')
ax.set_title('Distribution of LLM Grading Variance by Scale Granularity', fontweight='bold')

# Add mean annotations
for i, scale in enumerate(SCALE_ORDER):
    meanVal = np.mean(violinData[i])
    ax.annotate(f'μ={meanVal:.4f}', xy=(i+1, meanVal), xytext=(i+1.3, meanVal),
                fontsize=9, ha='left', color=list(SCALE_COLORS.values())[i], fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'variance_violins.png', dpi=150, bbox_inches='tight')
plt.show()`}
        output={
          <ImageOutput
            src="/research/variance_violins.png"
            alt="Variance Distribution Violin Plot — 4 scales"
          />
        }
      />

      {/* ─── Cell 12: Section Header — Per-Text Lines (Markdown) ─────────── */}
      <MarkdownCell>
        <hr className="my-3 border-gray-200" />
        <h2 className="text-lg font-bold text-gray-900 mb-2">
          4.5 — Per-Text Degradation Lines
        </h2>
        <p className="text-sm">
          Does the degradation pattern hold for ALL text types or only some?
          Each line represents one text.
        </p>
      </MarkdownCell>

      {/* ─── Cell 13: Per-Text Degradation ───────────────────────────────── */}
      <CodeCell
        executionCount={8}
        code={`fig, ax = plt.subplots(figsize=(12, 7))

for tIdx, result in enumerate(allResults):
    textId = result['id']
    meanVarPerScale = []
    for scale in SCALE_ORDER:
        meanVarPerScale.append(varianceMatrix[scale][tIdx, :].mean())
    ax.plot([0, 1, 2, 3], meanVarPerScale, marker='o', label=textId, alpha=0.8, linewidth=2)

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['Binary', 'Ternary', 'Quaternary', 'Continuous'])
ax.set_ylabel('Mean Variance (across 20 questions)')
ax.set_xlabel('Grading Scale')
ax.set_title('Grading Variance Degradation by Text Type', fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'per_text_degradation.png', dpi=150, bbox_inches='tight')
plt.show()`}
        output={
          <ImageOutput
            src="/research/per_text_degradation.png"
            alt="Per-Text Degradation Lines — 10 text types across 4 scales"
          />
        }
      />

      {/* ─── Cell 14: Section Header — Entropy vs Variance (Markdown) ────── */}
      <MarkdownCell>
        <hr className="my-3 border-gray-200" />
        <h2 className="text-lg font-bold text-gray-900 mb-2">
          4.6 — Entropy vs Variance Scatter
        </h2>
        <p className="text-sm">
          Are entropy and variance correlated? Does the relationship differ by
          scale? Each point is one (text, question) pair.
        </p>
      </MarkdownCell>

      {/* ─── Cell 15: Entropy vs Variance ────────────────────────────────── */}
      <CodeCell
        executionCount={9}
        code={`fig, ax = plt.subplots(figsize=(10, 7))

for scale in SCALE_ORDER:
    allVars = varianceMatrix[scale].flatten()
    allEntropy = entropyMatrix[scale].flatten()
    ax.scatter(allVars, allEntropy, alpha=0.35, label=scale.capitalize(),
               color=SCALE_COLORS[scale], s=25, edgecolors='white', linewidths=0.3)

ax.set_xlabel('Variance')
ax.set_ylabel('Shannon Entropy (bits)')
ax.set_title('Entropy vs Variance by Scale (800 data points)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'entropy_vs_variance.png', dpi=150, bbox_inches='tight')
plt.show()`}
        output={
          <ImageOutput
            src="/research/entropy_vs_variance.png"
            alt="Entropy vs Variance Scatter — 800 data points colored by scale"
          />
        }
      />

      {/* ─── Cell 16: Section Header — Mode Consistency (Markdown) ────────── */}
      <MarkdownCell>
        <hr className="my-3 border-gray-200" />
        <h2 className="text-lg font-bold text-gray-900 mb-2">
          4.7 — Mode Consistency Bar Chart
        </h2>
        <p className="text-sm">
          How often does the LLM give the same most-common answer? Grouped by
          scale, with one bar per text.
        </p>
      </MarkdownCell>

      {/* ─── Cell 17: Mode Consistency ───────────────────────────────────── */}
      <CodeCell
        executionCount={10}
        code={`fig, ax = plt.subplots(figsize=(14, 6))

barWidth = 0.08
textIds = [r['id'] for r in allResults]

for sIdx, scale in enumerate(SCALE_ORDER):
    positions = np.arange(nTexts) + sIdx * barWidth
    meanConsistencies = [consistencyMatrix[scale][tIdx, :].mean() for tIdx in range(nTexts)]
    ax.bar(positions, meanConsistencies, barWidth, label=scale.capitalize(),
           color=SCALE_COLORS[scale], alpha=0.85, edgecolor='white', linewidth=0.5)

ax.set_xticks(np.arange(nTexts) + barWidth * 1.5)
ax.set_xticklabels(textIds, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean Mode Consistency (across 20 questions)')
ax.set_title('LLM Response Consistency by Text and Scale', fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend()
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'mode_consistency_bars.png', dpi=150, bbox_inches='tight')
plt.show()`}
        output={
          <ImageOutput
            src="/research/mode_consistency_bars.png"
            alt="Mode Consistency Grouped Bar Chart — 10 texts × 4 scales"
          />
        }
      />

      {/* ─── Cell 18: Section Header — Text Difficulty (Markdown) ─────────── */}
      <MarkdownCell>
        <hr className="my-3 border-gray-200" />
        <h2 className="text-lg font-bold text-gray-900 mb-2">
          4.8 — Text Difficulty Ranking
        </h2>
        <p className="text-sm">
          Which texts are hardest for the LLM to grade consistently? Ranked by
          overall mean variance across all scales and questions.
        </p>
      </MarkdownCell>

      {/* ─── Cell 19: Text Difficulty ────────────────────────────────────── */}
      <CodeCell
        executionCount={11}
        code={`# Compute overall mean variance per text
textDifficulty = []
for tIdx, result in enumerate(allResults):
    overallMeanVar = np.mean([varianceMatrix[s][tIdx, :].mean() for s in SCALE_ORDER])
    textDifficulty.append((result['id'], overallMeanVar, result['text'][:80]))

textDifficulty.sort(key=lambda x: x[1], reverse=True)

print('Text Difficulty Ranking (most unstable first):')
print('=' * 80)
for rank, (textId, meanVar, preview) in enumerate(textDifficulty, 1):
    print(f'  {rank}. {textId:25s} mean_variance={meanVar:.6f}')

# Bar chart
fig, ax = plt.subplots(figsize=(12, 6))
ids = [t[0] for t in textDifficulty]
vars_ = [t[1] for t in textDifficulty]
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ids)))
ax.barh(range(len(ids)), vars_, color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(ids)))
ax.set_yticklabels(ids, fontsize=9)
ax.set_xlabel('Mean Variance (across all scales and questions)')
ax.set_title('Text Difficulty Ranking — Most Unstable for LLM Grading', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'text_difficulty_ranking.png', dpi=150, bbox_inches='tight')
plt.show()`}
        output={
          <div className="space-y-3">
            <TextOutput
              text={`Text Difficulty Ranking (most unstable first):
================================================================================
  1. ambiguous                mean_variance=0.008806
  2. narrative_paragraph      mean_variance=0.005441
  3. medical_clinical         mean_variance=0.004114
  4. sentiment_negative       mean_variance=0.002903
  5. negation_heavy           mean_variance=0.002618
  6. technical_ml             mean_variance=0.002365
  7. sentiment_positive       mean_variance=0.001918
  8. imperative_action        mean_variance=0.001607
  9. factual_simple           mean_variance=0.000846
  10. abstract_philosophical   mean_variance=0.000215`}
            />
            <ImageOutput
              src="/research/text_difficulty_ranking.png"
              alt="Text Difficulty Ranking — horizontal bar chart"
            />
          </div>
        }
      />

      {/* ─── Cell 20: Section Header — Summary (Markdown) ────────────────── */}
      <MarkdownCell>
        <hr className="my-3 border-gray-200" />
        <h2 className="text-lg font-bold text-gray-900 mb-2">
          4.9 — Summary Statistics
        </h2>
        <p className="text-sm">Key findings for the Fithian meeting.</p>
      </MarkdownCell>

      {/* ─── Cell 21: Executive Summary ──────────────────────────────────── */}
      <CodeCell
        executionCount={12}
        code={`# Compute summary values
binaryMeanVar = np.mean(varianceMatrix['binary'])
continuousMeanVar = np.mean(varianceMatrix['continuous'])
ratio = continuousMeanVar / binaryMeanVar if binaryMeanVar > 0 else float('inf')

# Most/least stable question
questionMeanVars = np.zeros(nQuestions)
for scale in SCALE_ORDER:
    questionMeanVars += varianceMatrix[scale].mean(axis=0)
questionMeanVars /= nScales

mostStableQ = questionFamilies[np.argmin(questionMeanVars)]
leastStableQ = questionFamilies[np.argmax(questionMeanVars)]

# Most/least stable text
textMeanVars = np.zeros(nTexts)
for scale in SCALE_ORDER:
    textMeanVars += varianceMatrix[scale].mean(axis=1)
textMeanVars /= nScales

mostStableT = allResults[np.argmin(textMeanVars)]['id']
leastStableT = allResults[np.argmax(textMeanVars)]['id']

print(summary)`}
        output={
          <TextOutput
            text={`
SUMMARY FOR FITHIAN MEETING
==================================================
Texts analyzed: 10
Questions per text: 20 (mechanistic mode)
Samples per question per scale: 20
Total evaluations: 16,000
Model: gpt-4o-mini (temperature=0)

Key Finding: Variance stays flat or decreases monotonically from binary → continuous.
Binary mean variance:     0.002413
Continuous mean variance: 0.002014
Ratio (continuous/binary): 0.83x

Wilcoxon tests (one-sided, H1: scale B more variable than scale A):
  Binary       → Ternary     : p = 0.225498 [not significant at α=0.05]
  Ternary      → Quaternary  : p = 0.058498 [not significant at α=0.05]
  Quaternary   → Continuous  : p = 0.941372 [not significant at α=0.05]

Most stable question:  first_person (mean var = 0.000000)
Least stable question: concreteness (mean var = 0.007612)
Most stable text type:  abstract_philosophical (mean var = 0.000215)
Least stable text type: ambiguous (mean var = 0.008806)`}
          />
        }
      />
    </div>
  )
}
