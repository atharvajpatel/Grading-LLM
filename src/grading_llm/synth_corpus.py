"""
Synthetic Corpus Generation

Generates ~300 factor-controlled statements for question pruning.
Each factor has positive and negative cases, designed to vary independently.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from itertools import product


# Base templates with factor toggles
# Each tuple: (factor_on_template, factor_off_template)
FACTOR_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "named_entities": [
        ("John went to the park.", "Someone went to the park."),
        ("Microsoft released an update.", "A company released an update."),
        ("The event was held in Tokyo.", "The event was held in a city."),
        ("She bought a Nike shirt.", "She bought a shirt."),
        ("They attended the Olympics.", "They attended an event."),
    ],
    "actions_events": [
        ("She ran across the field.", "The field was green."),
        ("The storm destroyed the building.", "The building was old."),
        ("He typed the report quickly.", "The report was long."),
        ("The ball bounced twice.", "The ball was red."),
        ("They built a new house.", "The house was large."),
    ],
    "causality": [
        ("The rain caused flooding.", "There was rain and flooding."),
        ("She was late because of traffic.", "She was late. There was traffic."),
        ("Therefore, we need to act.", "We need to act."),
        ("The fire led to evacuations.", "There was a fire and evacuations."),
        ("Due to costs, the project ended.", "The project ended. Costs were high."),
    ],
    "temporal": [
        ("The meeting is at 3pm.", "The meeting is scheduled."),
        ("She visited last summer.", "She visited recently."),
        ("They will arrive tomorrow.", "They plan to arrive."),
        ("After lunch, we left.", "We had lunch and left."),
        ("It happens every Monday.", "It happens regularly."),
    ],
    "spatial": [
        ("The book is on the shelf.", "The book is available."),
        ("She stood under the tree.", "She was near the tree."),
        ("Turn left at the corner.", "The corner is ahead."),
        ("It is five miles away.", "It is nearby."),
        ("The cat is behind the sofa.", "The cat is somewhere."),
    ],
    "numeric": [
        ("There are 47 students.", "There are many students."),
        ("Sales grew by 25%.", "Sales grew significantly."),
        ("She has three cats.", "She has some cats."),
        ("The table is 2 meters long.", "The table is quite long."),
        ("More than 100 people came.", "Many people came."),
    ],
    "negation": [
        ("She never eats meat.", "She rarely eats meat."),
        ("There is no water left.", "There is little water left."),
        ("This is not acceptable.", "This is unusual."),
        ("Nobody came to help.", "Few people came to help."),
        ("The result was unexpected.", "The result was surprising."),
    ],
    "uncertainty": [
        ("It might rain tomorrow.", "It will rain tomorrow."),
        ("The answer seems correct.", "The answer is correct."),
        ("Perhaps we should wait.", "We should wait."),
        ("I'm not sure about this.", "I know about this."),
        ("There is a 60% chance.", "This will happen."),
    ],
    "modality": [
        ("You should finish this.", "You can finish this."),
        ("She must leave now.", "She is leaving now."),
        ("They may attend the event.", "They are attending."),
        ("We ought to help.", "We want to help."),
        ("He can solve this.", "He will solve this."),
    ],
    "sentiment": [
        ("This is a wonderful idea.", "This is an idea."),
        ("The service was terrible.", "The service was slow."),
        ("I love this product.", "I bought this product."),
        ("The weather is beautiful.", "The weather is warm."),
        ("That was disappointing.", "That was unexpected."),
    ],
    "emotion": [
        ("She felt happy about it.", "She reacted to it."),
        ("He was angry and upset.", "He was affected."),
        ("The news made them sad.", "The news reached them."),
        ("I am anxious about this.", "I am thinking about this."),
        ("They were excited.", "They were ready."),
    ],
    "social": [
        ("They discussed the plan.", "They knew the plan."),
        ("She called to explain.", "She understood it."),
        ("We are close friends.", "We know each other."),
        ("The team collaborated.", "The team worked."),
        ("They argued about it.", "They disagreed."),
    ],
    "dialogue": [
        ('She said, "I will come."', "She planned to come."),
        ("According to him, it works.", "It works."),
        ("He mentioned the deadline.", "The deadline exists."),
        ("She asked about the price.", "She knew the price."),
        ("They replied immediately.", "They responded."),
    ],
    "first_person": [
        ("I went to the store.", "Someone went to the store."),
        ("We completed the project.", "The project was completed."),
        ("I believe this is right.", "This is right."),
        ("My opinion is different.", "The opinion is different."),
        ("I saw it happen.", "It happened."),
    ],
    "imperative": [
        ("Press the button now.", "The button can be pressed."),
        ("Close the door.", "The door should be closed."),
        ("Please submit the form.", "The form is due."),
        ("Make sure to save.", "Saving is important."),
        ("Turn right here.", "The turn is here."),
    ],
    "comparison": [
        ("This is better than that.", "This is good."),
        ("She runs faster than him.", "She runs fast."),
        ("The best option is clear.", "A good option exists."),
        ("They are similar in style.", "They have style."),
        ("Like a lion, he roared.", "He roared loudly."),
    ],
    "normative": [
        ("It is wrong to lie.", "People sometimes lie."),
        ("This is unacceptable behavior.", "This is unusual behavior."),
        ("Honesty is important.", "Honesty is valued."),
        ("That violates policy.", "That is not typical."),
        ("Everyone should help.", "Everyone can help."),
    ],
    "intent": [
        ("She plans to expand.", "She expanded."),
        ("The goal is to reduce costs.", "Costs were reduced."),
        ("He studies to pass.", "He studies and passes."),
        ("We hope to succeed.", "We succeeded."),
        ("To save time, use this.", "This saves time."),
    ],
    "concreteness": [
        ("The wooden table is heavy.", "The concept is complex."),
        ("The loud crash echoed.", "Something happened."),
        ("The smooth metal feels cold.", "The material is interesting."),
        ("Red flowers bloomed.", "Ideas flourished."),
        ("Smoke rose from the chimney.", "The situation improved."),
    ],
    "identity": [
        ("She is a doctor.", "She works in healthcare."),
        ("He is intelligent.", "He completed the task."),
        ("The engineer designed it.", "Someone designed it."),
        ("As a member, he voted.", "He voted."),
        ("I consider myself creative.", "I do creative work."),
    ],
}


def generate_synthetic_corpus(
    n_base: int = 20,
    seed: int = 42
) -> List[Dict[str, any]]:
    """
    Generate a synthetic corpus of factor-controlled statements.

    Creates statements where factors are systematically toggled on/off
    to enable correlation-based question pruning.

    Args:
        n_base: Number of base statements per factor toggle combination
        seed: Random seed for reproducibility

    Returns:
        List of statement dictionaries with factor labels
    """
    random.seed(seed)
    corpus = []
    statement_id = 0

    # Generate pure factor examples (one factor at a time)
    for factor, templates in FACTOR_TEMPLATES.items():
        for template_on, template_off in templates:
            # Factor ON
            corpus.append({
                "id": f"synth_{statement_id:04d}",
                "text": template_on,
                "factors": {factor: True},
                "source": "pure_positive"
            })
            statement_id += 1

            # Factor OFF
            corpus.append({
                "id": f"synth_{statement_id:04d}",
                "text": template_off,
                "factors": {factor: False},
                "source": "pure_negative"
            })
            statement_id += 1

    # Generate composite statements (multiple factors combined)
    composite_templates = _generate_composite_statements(n_base, seed)
    for item in composite_templates:
        item["id"] = f"synth_{statement_id:04d}"
        corpus.append(item)
        statement_id += 1

    return corpus


def _generate_composite_statements(n: int, seed: int) -> List[Dict]:
    """Generate statements with multiple factors combined."""
    random.seed(seed + 1)
    composites = []

    # Select random factor combinations
    factors = list(FACTOR_TEMPLATES.keys())

    for _ in range(n):
        # Pick 2-4 factors to combine
        n_factors = random.randint(2, 4)
        selected_factors = random.sample(factors, n_factors)

        # Randomly decide which are on/off
        factor_states = {f: random.choice([True, False]) for f in selected_factors}

        # Build composite text
        parts = []
        for factor, is_on in factor_states.items():
            templates = FACTOR_TEMPLATES[factor]
            template_on, template_off = random.choice(templates)
            parts.append(template_on if is_on else template_off)

        # Combine (simplified - just concatenate)
        text = " ".join(parts[:2])  # Use at most 2 parts for readability

        composites.append({
            "text": text,
            "factors": factor_states,
            "source": "composite"
        })

    return composites


def generate_minimal_pair_corpus(
    questions: List[Dict]
) -> List[Dict[str, any]]:
    """
    Generate corpus from question minimal pairs.

    Args:
        questions: List of question dictionaries with minimal_pairs

    Returns:
        List of statement dictionaries
    """
    corpus = []
    statement_id = 0

    for q in questions:
        minimal_pairs = q.get("minimal_pairs", {})
        family = q.get("family", "unknown")
        qid = q.get("id", "unknown")

        if "positive" in minimal_pairs:
            corpus.append({
                "id": f"mp_{statement_id:04d}",
                "text": minimal_pairs["positive"],
                "question_id": qid,
                "family": family,
                "expected": 1,
                "source": "minimal_pair_positive"
            })
            statement_id += 1

        if "negative" in minimal_pairs:
            corpus.append({
                "id": f"mp_{statement_id:04d}",
                "text": minimal_pairs["negative"],
                "question_id": qid,
                "family": family,
                "expected": 0,
                "source": "minimal_pair_negative"
            })
            statement_id += 1

    return corpus


def save_corpus(corpus: List[Dict], path: Path) -> None:
    """Save corpus to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item) + "\n")


def load_corpus(path: Path) -> List[Dict]:
    """Load corpus from JSONL file."""
    corpus = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    return corpus
