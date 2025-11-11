# Data Structure Guide

## Directory Organization

The `lib/` directory contains all pre-processed dataset files organized by experiment type:

```
lib/
├── plain/                      # Plain baseline questions (no opinions)
│   └── mmlu_plain.pkl         # Baseline plain questions
├── opinion_only/               # Opinion-only experiments
│   ├── prefix/                # Opinion before question
│   │   └── mmlu_opinion_only.pkl
│   └── suffix/                # Opinion after question
│       └── mmlu_opinion_only.pkl
└── pov/                       # Point-of-view (perspective) experiments
    ├── prefix/                # Prefix-based prompts
    │   ├── first_pov/         # First-person ("I believe...")
    │   │   ├── mmlu_academic_opinion_beginner.pkl
    │   │   ├── mmlu_academic_opinion_intermediate.pkl
    │   │   └── mmlu_academic_opinion_advanced.pkl
    │   └── third_pov/         # Third-person ("They believe...")
    │       ├── mmlu_academic_opinion_beginner.pkl
    │       ├── mmlu_academic_opinion_intermediate.pkl
    │       └── mmlu_academic_opinion_advanced.pkl
    └── suffix/                # Suffix-based prompts
        ├── first_pov/
        │   ├── mmlu_academic_opinion_beginner.pkl
        │   ├── mmlu_academic_opinion_intermediate.pkl
        │   └── mmlu_academic_opinion_advanced.pkl
        └── three_pov/         # Note: typo preserved from original
            ├── mmlu_academic_opinion_beginner.pkl
            ├── mmlu_academic_opinion_intermediate.pkl
            └── mmlu_academic_opinion_advanced.pkl
```

## File Naming Convention

- **Plain**: `mmlu_plain.pkl` - Baseline questions without opinions
- **Opinion-only**: `mmlu_opinion_only.pkl` - Questions with incorrect user opinions
- **POV with expertise**: `mmlu_academic_opinion_{level}.pkl` where level is:
  - `beginner` - Novice/Learner level
  - `intermediate` - Practitioner/Junior level  
  - `advanced` - Expert/Authority level

## Experiment Type Mapping

| Paper Section | Data Location |
|--------------|---------------|
| Plain baseline (Figure 2) | `lib/plain/mmlu_plain.pkl` |
| Opinion-only (Figure 2) | `lib/opinion_only/prefix/mmlu_opinion_only.pkl` |
| First-person + Expertise (Figure 3) | `lib/pov/prefix/first_pov/mmlu_academic_opinion_{level}.pkl` |
| Third-person comparison (Figure 8) | `lib/pov/prefix/third_pov/mmlu_academic_opinion_{level}.pkl` |

## Usage Examples

### Load Plain Baseline
```python
import pandas as pd
df = pd.read_pickle("lib/plain/mmlu_plain.pkl")
```

### Load Opinion-Only
```python
df = pd.read_pickle("lib/opinion_only/prefix/mmlu_opinion_only.pkl")
```

### Load First-Person Advanced
```python
df = pd.read_pickle("lib/pov/prefix/first_pov/mmlu_academic_opinion_advanced.pkl")
```

### Load Third-Person Beginner
```python
df = pd.read_pickle("lib/pov/prefix/third_pov/mmlu_academic_opinion_beginner.pkl")
```

## Data Format

All `.pkl` files contain pandas DataFrames with the following columns:

- `question`: The original MMLU question
- `A`, `B`, `C`, `D`: Multiple choice options
- `answer`: Correct answer letter
- `full_question`: Complete prompt with prefix/opinion if applicable
- `user_opinion`: The incorrect answer stated by the user (if applicable)
- `subject`: MMLU subject category

