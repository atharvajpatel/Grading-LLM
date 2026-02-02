# GRADING-LLM

Measure how consistent an LLM is when grading statements across different answer granularities.

## What This Does

Given a statement, this tool:
1. Asks 20 questions about it (choose from 2 question modes)
2. Repeats each question 20 times under 4 different grading scales:
   - **Binary**: {0, 1}
   - **Ternary**: {0, 0.5, 1}
   - **Quaternary**: {0, 0.33, 0.66, 1}
   - **Continuous**: [0, 1]
3. Computes stability metrics (variance, entropy, mode consistency)
4. Performs PCA across question dimensions
5. Produces a 3D visualization showing where the model "changes its mind"

## Question Modes

Choose between two distinct question sets:

### Mechanistic Mode (`mech`)

Probes **explicit linguistic and semantic features**:
- Named entities, actions, causality
- Temporal/spatial references, numbers
- Negation, uncertainty, modality
- Sentiment, emotion, social dynamics
- First-person, imperative, comparison
- Normative judgments, intent, concreteness, identity

**Best for:** Understanding how models process surface-level linguistic features.

### Interpretability Mode (`interp`)

Probes **implicit meaning, inference, and social understanding**:
- Implicit judgments and implied problems
- Unstated reasons and tone ambiguity
- Pragmatic meaning beyond literal text
- Mental state inference and social tension
- Emotional subtext and context dependence
- Power dynamics, omissions, social norms
- Explicit vs implicit meaning, indirection

**Best for:** Understanding how models handle nuanced, context-dependent interpretation.

See `data/questions_mech.json` and `data/questions_interp.json` for full question lists.

## Why This Matters

If an LLM were a perfectly consistent grader, it would give the same answer every time for the same question. In reality:
- Models exhibit **variance** even at temperature=0
- Finer-grained scales (continuous) typically show more instability than coarse ones (binary)
- PCA reveals **which semantic dimensions** drive this instability

The loadings on each principal component tell you which questions the model is most uncertain about—exposing where interpretability probes might be unreliable.

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Set API Key

Paste your OpenAI API key in `src/grading_llm/config.py`:

```python
OPENAI_API_KEY = "sk-..."
```

Or set the environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run

```bash
python -m grading_llm.run_single --text "The Earth orbits the Sun."
```

Results are saved to `logs/YYYYMMDD_HHMMSS_<hash>/`:
- `pca_plot.png` - 3D visualization
- `report.md` - Full analysis with auto-generated interpretation
- `metrics.json` - Raw metrics data
- `responses.jsonl` - All API responses

## Web UI

### Local Development

```bash
# Terminal 1: Backend
pip install -r requirements.txt
python main.py

# Terminal 2: Frontend
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173` in your browser.

### Deploy to Production

**Frontend (Vercel):**
1. Push repo to GitHub
2. Import at [vercel.com](https://vercel.com)
3. Set **Root Directory** to `frontend`
4. Add environment variable: `VITE_API_URL=https://your-backend.onrender.com`
5. Deploy (uses `frontend/vercel.json` config)

**Backend (Render):**
1. Import repo at [render.com](https://render.com)
2. Deploy (uses `render.yaml` config)

**Note:** Users provide their own OpenAI API key via the frontend UI. No server-side API key is required for deployment.

### Features

- **Analyze Tab**: Enter a statement and view:
  - **The Problem** section explaining LLM-as-a-judge calibration
  - **Question mode selector** (Mechanistic or Interpretability)
  - Real-time progress bar with current scale and sample
  - Cancel button to stop running analysis
  - **Value Distribution** with metric definitions and per-scale histograms
  - Interactive 3D PCA plot (rotate, zoom, pan)
  - Per-scale heatmaps showing response patterns
  - Detailed stability metrics table

- **Documentation Tab**:
  - Methodology explanation (1600 evaluations breakdown)
  - Browse all 20 questions by factor family
  - Learn how PCA reveals model uncertainty
  - Understand limitations and future directions

### Job Management

The backend uses an async job-based architecture:
- Jobs run in background with progress tracking
- Cancel anytime - no ghost jobs left running
- Page close/refresh automatically cancels the job
- Progress updates every second via polling

---

## CLI Commands

### Single Statement

```bash
python -m grading_llm.run_single --text "Your statement here"
python -m grading_llm.run_single --text "..." --samples 20  # More samples
```

### Batch Processing

```bash
python -m grading_llm.run_batch --input statements.jsonl
```

Input format (JSONL):
```json
{"id": "stmt_001", "text": "First statement"}
{"id": "stmt_002", "text": "Second statement"}
```

### Re-generate Question Bank (Advanced)

```bash
python -m grading_llm.prune_questions \
    --candidates data/questions_100.json \
    --out data/questions_20.json \
    --k 20 --corr_threshold 0.85
```

## Why 20 Questions?

The default question bank contains 20 carefully selected questions:
- **1 per factor family** (20 families × 1 = 20)
- **Low redundancy**: Selected via correlation-based pruning
- **Cheaper**: Fewer API calls per experiment
- **Cleaner PCA**: Interpretable components

The 100-question candidate pool (`data/questions_100.json`) is available for broader analysis.

## Factor Families

Questions cover 20 orthogonal semantic dimensions:

| Family | Example Question |
|--------|-----------------|
| named_entities | Does the statement mention a specific person by name? |
| actions_events | Does the statement describe an action being performed? |
| causality | Does the statement express a cause-and-effect relationship? |
| temporal | Does the statement reference a specific time or date? |
| spatial | Does the statement use spatial prepositions? |
| numeric | Does the statement contain explicit numbers? |
| negation | Does the statement contain explicit negation? |
| uncertainty | Does the statement express uncertainty (might, maybe)? |
| modality | Does the statement express obligation (should, must)? |
| sentiment | Does the statement contain evaluative language? |
| emotion | Does the statement name a specific emotion? |
| social | Does the statement describe interaction between people? |
| dialogue | Does the statement contain quoted speech? |
| first_person | Does the statement use first-person pronouns? |
| imperative | Does the statement give instructions or commands? |
| comparison | Does the statement compare two or more things? |
| normative | Does the statement make moral or ethical judgments? |
| intent | Does the statement describe intention or purpose? |
| concreteness | Does the statement describe tangible objects? |
| identity | Does the statement describe someone's identity or role? |

## Output Structure

```
logs/
└── 20240115_143022_abc123/
    ├── config.json          # Run configuration
    ├── responses.jsonl      # Raw API responses
    ├── embeddings.json      # Mean embeddings per scale
    ├── metrics.json         # Variance, entropy, PCA stats
    ├── pca_plot.png         # 3D visualization
    ├── variance_plot.png    # Variance by scale
    └── report.md            # Auto-generated analysis
```

## Understanding PCA: A Deep Dive

### What is PCA?

**Principal Component Analysis (PCA)** is a technique that finds the directions of maximum variance in high-dimensional data. In our case, we have 80 embedding samples (20 samples × 4 scales), each with 20 dimensions (one per question). PCA reduces this to 3 dimensions for visualization while preserving the most important patterns.

### The Math Behind PCA

#### Step 1: Input Data Matrix

We start with an **80×20 matrix**:
- **Rows**: 80 samples (20 per scale × 4 scales)
- **Columns**: 20 questions
- **Each cell**: LLM's score (0-1) for that question

```
         Q1    Q2    Q3   ...  Q20
Sample1  0.0   1.0   0.5  ...  0.0
Sample2  0.0   1.0   0.5  ...  0.0
...
Sample80 0.3   0.8   0.4  ...  0.2
```

#### Step 2: Center the Data

Subtract the mean of each column (question) so the data is centered around zero.

#### Step 3: Compute Covariance Matrix

Calculate a **20×20 covariance matrix** showing how questions co-vary with each other:
- High positive covariance: questions tend to be scored similarly
- High negative covariance: questions tend to be scored oppositely
- Near-zero covariance: questions are independent

#### Step 4: Eigendecomposition

Find the **eigenvectors** and **eigenvalues** of the covariance matrix:
- **Eigenvectors**: The directions (principal components) in 20D question space
- **Eigenvalues**: How much variance is captured by each direction

#### Step 5: Sort by Importance

Order principal components by their eigenvalues (largest first):
- **PC1**: Captures the most variance (largest eigenvalue)
- **PC2**: Captures second-most variance (orthogonal to PC1)
- **PC3**: Captures third-most variance (orthogonal to PC1 and PC2)

### Key Concepts Explained

#### Explained Variance Ratio

The **explained variance ratio** tells you what percentage of total data spread each PC captures:

```
Explained Variance = eigenvalue_i / sum(all eigenvalues)
```

| Component | Eigenvalue | Explained Variance |
|-----------|------------|-------------------|
| PC1 | 2.5 | 2.5 / 5.0 = **50%** |
| PC2 | 1.5 | 1.5 / 5.0 = **30%** |
| PC3 | 0.5 | 0.5 / 5.0 = **10%** |
| PC4-PC20 | 0.5 | Combined **10%** |

**Interpretation**: If PC1 explains 50% of variance, half of all the variation in your data can be explained by movement along this single direction.

#### Loadings (Coefficients)

Each PC is a **linear combination of all 20 questions**. The **loading** is the coefficient for each question:

```
PC1 = 0.40×Q3 + 0.35×Q7 + 0.25×Q12 + 0.15×Q1 + 0.10×Q5 + ... (all 20 terms)
      ↑         ↑         ↑
   loading   loading   loading
```

**What loadings tell you**:
- **Large positive loading** (+0.4): High scores on this question push samples in the positive PC direction
- **Large negative loading** (-0.4): High scores push samples in the negative PC direction
- **Small loading** (±0.05): Question has little influence on this PC

#### Top Contributing Questions

When we show "Top 5 Questions" for each PC, we're showing the questions with the **highest absolute loadings**—the ones that most strongly influence that principal component.

**Important**: PCA does NOT select or filter questions. All 20 questions contribute to every PC. We just highlight the dominant ones.

### Reading the 3D Visualization

The 3D plot shows 80 points projected onto (PC1, PC2, PC3) space:

| Visual Pattern | Meaning |
|----------------|---------|
| **Tight cluster** | Model gives consistent answers across samples |
| **Spread points** | Model is uncertain, answers vary |
| **Separated clusters by color** | Different scales produce different embeddings |
| **Overlapping clusters** | Scales produce similar embeddings |

#### What Each Element Shows

- **Spheres**: Individual embedding samples (20 per scale)
- **Octahedrons**: Centroids (mean position) for each scale
- **Colors**: Scale type (blue=binary, green=ternary, orange=quaternary, pink=continuous)
- **Axes**: PC1, PC2, PC3 with explained variance percentages
- **Grid**: -1 to 1 normalized coordinate system (centered at origin)

#### Understanding the Coordinate Scale

**Raw PCA coordinates are NOT bounded to a fixed range.** Here's why and how we handle it:

**How raw PCA coordinates are calculated:**
```
PC1_coordinate = Σ (centered_value × loading)  for all 20 questions
```

Each coordinate is the dot product of your centered data point with the eigenvector. Since:
- Centered values range roughly ±0.5 (original 0-1 minus mean ~0.5)
- Loadings are unit-normalized (each ~0.2-0.3 for top contributors)
- 20 terms sum together

**Typical raw ranges: approximately -2 to +2**

| Raw PC Value | Meaning |
|--------------|---------|
| **+1.5 to +2** | Sample scores FAR above average on questions with positive loadings |
| **+0.5 to +1** | Sample is moderately above average |
| **-0.5 to +0.5** | Sample is near average (most points cluster here) |
| **-1 to -0.5** | Sample is moderately below average |
| **-2 to -1.5** | Sample scores FAR below average |

**Why we normalize to -1 to 1 for visualization:**

We apply min-max normalization to each PC axis, then scale to -1 to 1:
```
normalized = ((raw_value - min) / (max - min)) * 2 - 1
```

This ensures:
- All axes are visually comparable (same -1 to 1 scale)
- Points are centered around the origin
- The visualization is symmetric, making it easier to see deviations from center
- Negative and positive directions have equal visual weight

**Important:** The -1 to 1 grid shows **relative positions within this dataset**, not absolute values. A point at 0 means "middle of the observed range," -1 means "minimum of observed range," and +1 means "maximum of observed range."

### Practical Interpretation

#### Example Analysis

If your results show:
- **PC1 (45% variance)**: Top questions are about "actions" and "causality"
- **PC2 (25% variance)**: Top questions are about "sentiment" and "emotion"
- **PC3 (15% variance)**: Top questions are about "temporal" references

This means:
1. The biggest source of variation is how the model scores action/causality questions
2. Sentiment/emotion questions are the second biggest source of disagreement
3. Temporal questions contribute less to overall instability

#### What This Reveals About Model Uncertainty

- **Questions with high loadings on PC1**: Most unstable—model "changes its mind" most on these
- **Questions with low loadings everywhere**: Most stable—model is consistent
- **Questions loading on multiple PCs**: Maximally confusing to the model

### Summary Table

| Term | Definition | Higher Value Means |
|------|------------|-------------------|
| **Eigenvalue** | Variance captured by a PC | More important PC |
| **Explained Variance** | % of total variance | PC captures more data patterns |
| **Loading** | Question's coefficient in PC | Question influences PC more |
| **Top 5 Questions** | Highest absolute loadings | Most influential questions for that PC |

### Why We Use 3 PCs

We reduce from 20 dimensions to 3 because:
1. **Visualization**: Humans can see 3D but not 20D
2. **Information retention**: First 3 PCs typically capture 60-80% of variance
3. **Interpretability**: Easier to understand 3 directions than 20

## Understanding Stability Metrics

### Consistency (Mode Matching)

**Consistency** measures what percentage of responses match the most common value (the mode).

```
Consistency = count(mode) / total_responses
```

| Consistency | Interpretation |
|-------------|----------------|
| **100%** | Perfect agreement—all responses identical |
| **75-99%** | High agreement—one answer dominates |
| **50-74%** | Moderate disagreement—answers split |
| **<50%** | High disagreement—no clear consensus |

**Example**: If 20 samples for a question score [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]:
- Mode = 0 (appears 15 times)
- Consistency = 15/20 = **75%**

### Entropy (Shannon Entropy)

**Entropy** measures the spread/unpredictability of values in bits. Higher entropy = more disorder.

```
Entropy = -Σ p(x) × log₂(p(x))  for all unique values x
```

| Entropy | Interpretation |
|---------|----------------|
| **0.0** | Perfect agreement—all responses identical |
| **0.5-1.0** | Low spread—one option strongly dominates |
| **1.0-1.5** | Moderate spread—values distributed across options |
| **1.5-2.0+** | High spread—values scattered unpredictably |

**Maximum entropy** depends on the number of options:
- Binary (2 options): max = 1.0 bits
- Ternary (3 options): max = 1.58 bits
- Quaternary (4 options): max = 2.0 bits
- Continuous (many values): can exceed 2.0 bits

### Consistency vs Entropy

These metrics are related but capture different things:

| Metric | Measures | Good Value |
|--------|----------|------------|
| **Consistency** | How dominant is the most common answer? | Higher is better |
| **Entropy** | How spread out are all answers? | Lower is better |

**Edge case**: You can have 50% consistency but low entropy if responses split evenly between just 2 values. Conversely, low consistency with high entropy means responses are scattered across many values.

### Variance

**Variance** measures the average squared deviation from the mean:

```
Variance = Σ(x - mean)² / n
```

For 0-1 bounded data:
- **0.0**: All values identical
- **0.25**: Maximum variance (half 0s, half 1s)
- **<0.05**: Very consistent responses

## API Usage

Each run makes approximately:
- Single statement: `20 samples × 4 scales × 1 batch = 80 API calls`
- Questions are batched 20 at a time for efficiency

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
