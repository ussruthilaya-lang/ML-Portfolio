# Assignment 4 - Association Rule Mining (Apriori)
### CS6140 Machine Learning | Northeastern University
**Name:** Sruthilaya

---

## What is this?

This is my fourth assignment for CS6140. The goal was to build a market basket analysis pipeline from scratch using the Apriori algorithm — going from raw, messy transaction data all the way to a publication-quality visualization dashboard. Along the way the assignment pushed us to write our own data preprocessing (instead of relying on a given snippet), systematically study how Apriori's parameters affect the rules it finds, generalize a broken helper function, and build a custom scoring system for ranking rules.

---

## Dataset

Market_Basket_Optimisation.csv from the CS6140 course repo. 7,501 grocery store transactions, each with up to 20 purchased items, stored with **no header row**. The data is sparse - most cells in a row are empty (NaN) because customers typically buy far fewer than 20 items.

- 119 unique items across the dataset
- Average of ~3.91 items per transaction
- ~120,657 NaN/empty cells across the whole file — this sparsity is exactly why robust preprocessing mattered

---

## Tasks Completed

**Q1 - Robust Data Preprocessing**

Built `clean_transactions(filepath)` from scratch (the reference code intentionally omitted this step). It reads the raw CSV with Python's `csv` module (handles ragged rows that break `pd.read_csv` without a header), dynamically detects the number of rows/columns instead of hardcoding 7501/20, strips out NaN/empty cells, and removes duplicate items within a transaction while preserving order.

The function prints a summary report after loading:
- Total transactions: 7,501
- Detected columns: 20
- Average items per transaction: 3.91
- Total unique items across dataset: 119
- NaN/empty cells removed: 120,657

Wrote 3 unit tests with `unittest` covering the required edge cases: a row that's entirely NaN (→ empty transaction), a row with duplicate items (→ deduplicated, order preserved), and a completely empty CSV file (→ returns `[]` without crashing on `EmptyDataError`/`ZeroDivisionError`).

**Q2 - Parameter Sensitivity Experiment**

`run_apriori_experiment(transactions, support_values, confidence_values, min_lift, min_length, max_length)` systematically loops over 5 support values (`[0.001, 0.002, 0.003, 0.005, 0.01]`) and 4 confidence values (`[0.1, 0.2, 0.3, 0.5]`), runs Apriori for every combination, and returns a DataFrame of `min_support`, `min_confidence`, `num_rules`, `avg_lift`, `max_lift`.

Visualized the results with a heatmap (support × confidence → number of rules) and a line plot (avg_lift vs min_support, one line per confidence level). Key findings:
- min_support=0.001, min_confidence=0.1 → 77 rules, avg lift 4.49 — largest set, but with weaker associations near the lift=3 cutoff
- Raising confidence to 0.2 trims this to 29 rules while *raising* avg lift to 4.67 — the dropped rules were dragging quality down
- Beyond min_support=0.003, rule count collapses to 5-12 regardless of confidence
- min_confidence=0.5 produces **zero** rules at every support level — too strict for this sparse data
- Recommendation: min_support=0.001, min_confidence=0.2 — best balance of rule quantity (29) and quality (highest avg lift of any non-empty configuration)

**Q3 - Generalized inspect() and Rule Analysis Pipeline**

`parse_rules(results)` rewrites the broken pairwise-only `inspect()` into a generalized parser that handles rules of *any* length — looping over every ordered statistic and the full LHS/RHS itemsets (not just `[0][0]`). Returns a DataFrame with `LHS`, `RHS` (comma-separated strings), `num_items_lhs`, `num_items_rhs`, `Support`, `Confidence`, `Lift`. Tested with `max_length=3`, producing 74 rules including multi-item rules like `mineral water, whole wheat pasta → olive oil`.

`filter_rules(df, min_support, min_confidence, min_lift, contains_item)` filters the parsed DataFrame on any combination of thresholds, plus an item-containment check across both LHS and RHS. Demonstrated by finding all rules involving "mineral water" with lift > 3.

`compute_custom_score(df, w_support, w_confidence, w_lift)` min-max normalizes Support, Confidence, and Lift to [0, 1] and computes a weighted score. With weights (1.0, 1.5, 2.0), the top 10 by `weighted_score` and top 10 by `Lift` alone overlap substantially but not completely — the weighted score surfaces rules that balance all three metrics rather than over-indexing on lift alone (e.g., `tomato sauce → ground beef, spaghetti` ranks higher by lift than by weighted score because its support and confidence are comparatively low).

**Q4 - Visualization Dashboard**

Built three visualizations aimed at a non-technical store manager:
- A scatter plot of Support vs Confidence with point size proportional to Lift, color-coded (crimson for lift > 4), and the top 5 rules by lift annotated directly on the plot
- A horizontal bar chart of the top 15 rules by lift, labeled with readable `LHS → RHS` rule names and color-coded by Confidence using a `coolwarm` colormap with a colorbar legend
- `item_frequency_chart(transactions, top_n=20)` which counts and plots the most common items, then discusses whether high-frequency items tend to appear in high-lift rules

The discussion's conclusion: **no, they mostly don't**. Very common items like mineral water co-occur with almost everything by sheer volume, which pulls their lift toward 1 (statistical independence). The strongest-lift rules instead pair comparatively rare, niche items (e.g., `fromage blanc → honey` at lift 5.16) — pairings that happen for a real behavioral reason (like a recipe habit) rather than coincidence. So frequency drives common, high-support/high-confidence rules, while lift highlights meaningful — often niche — associations.

---

## Tools Used

- Python (Pandas, NumPy, Matplotlib, Seaborn, csv, collections.Counter)
- apyori (Apriori algorithm implementation)
- unittest
- Jupyter Notebook

---

## How to Run

1. Place `Market_Basket_Optimisation.csv` in a `dataset/` folder inside this directory (or load directly from the course URL)
2. `pip install apyori` if not already installed
3. Run all cells top to bottom

---

## Some things I noticed along the way

- The raw CSV has ragged rows (varying numbers of items per transaction with no header), which makes `pd.read_csv` throw a `ParserError` — switching to Python's built-in `csv` module sidestepped this entirely and felt more "from scratch" anyway.
- Lift is a much more interesting metric than support or confidence alone — it's the one that separates "this pair shows up a lot" from "these two things are *actually* related." Frequent items dilute their own lift just by being everywhere.
- Tightening confidence from 0.1 to 0.2 didn't just reduce rule count — it actually *improved* average rule quality, which was a nice illustration that "more rules" isn't automatically "better analysis."
- Generalizing `inspect()` to handle multi-item rules required rethinking the indexing entirely — the original `[0][0]` assumptions baked in a pairwise-only design that silently breaks (or crashes) the moment `max_length > 2`.

---

## Files Submitted
- `association.ipynb` - main notebook
- `submission_association.pdf` - PDF export of notebook with all outputs
