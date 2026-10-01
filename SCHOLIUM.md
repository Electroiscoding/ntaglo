# Scholium on the Feed Ranking Algorithm

[← Back to README](README.md)

**Contents:** [1. What it is](#1-what-it-is) · [2. The signals](#2-the-signals) · [3. Design character](#3-design-character) · [4. Findings](#4-findings-where-behavior-departs-from-intent) · [5. Tightening it](#5-if-you-want-to-tighten-it) · [6. Encouraged](#6-encouraged-behavior-and-content) · [7. Discouraged](#7-discouraged-behavior-and-content) · [8. Neither](#8-what-it-neither-rewards-nor-punishes) · [9. Second-order effects](#9-second-order-effects) · [10. What moves the ranking](#10-what-actually-moves-the-ranking) · [11. Summary](#11-summary)

---

## 1. What it is

A **per-viewer, content-based re-ranker**. It takes a candidate set of posts, one user and that user's interaction events, and returns a score-sorted list. Its method is a **weighted linear blend of normalized signals**, followed by small **multiplicative modulators**. Meaning comes from one source: unit-normalized MiniLM sentence embeddings (384-d), compared by cosine similarity.

```
final    = additive × (1 + location) × exposureCap
additive = 0.22·quality + 0.15·wR·roadmap + 0.15·wT·taste
         + 0.10·masteryNorm + 0.08·freshness + 0.03·exploration
```

`wR` is `roadmapWeight` (default 0.25) and `wT` is `tasteWeight` (default 0.30).

---

## 2. The signals

### Content quality (the prior)
- `raw = 0.55·metadata×2.5 + 0.45·0.75·((cos(post, trendCentroid)+1)/2) + 0.2·artifact − grammarPenalty`, clipped to [0, 1.5].
- It is then clamped into `[0.45, 0.95]` as the starting quality. This is the Bayesian prior mean: every post begins with at least moderate credit.

### Engagement (the evidence)
- Only views with at least 50% visibility and at least 1 s of dwell count, once per (user, post).
- Dwell scoring:

| Dwell | Signal |
|---|---|
| more than 15 s | +1.0 |
| 5 to 15 s | +0.2 |
| 1.5 to 5 s | 0 |
| 1.0 to 1.5 s | −0.10 |

- A like adds +0.15 and a reply adds +1.0.
- Posterior: `(w·prior + Σsignal) / (w + n)`.
- The prior weight `w` is half the expected consumption time (video length, or word count at 238 wpm), clipped to 5 to 60. Longer content demands more evidence before it moves.

### Roadmap and taste (intent)
- Roadmap similarity is `0.65·cos(roadmap, post) + 0.35·cos(roadmap, authorCentroid)`, so the author's history acts as a credibility prior.
- Taste is the best match across the user's tastes.
- Both are rescaled from the cosine band [0.15, 0.55] to [0, 1], which turns "mildly related" into 0 and "clearly related" into 1.

### Mastery (behavioral modulation)
A product of five factors, squashed by a logistic centered at 1:

| Factor | Meaning |
|---|---|
| E | `0.5·accessibility + 1.2·dwellPercentile` |
| M | dwell ÷ session length |
| L | "topic distance" bump (1.0 to 1.56) that favors newer items |
| X | lexical rejection penalty on reply text, down to 0.55 |
| G | device and bandwidth fit (heavy media favored when charging and fast, light media on old or slow devices) |

### Freshness
`1/(1+(age/48h)^1.2)`, damped when three or more repeat impressions occurred.

### Exploration
Trend similarity × a mild view-count boost (up to ×1.1) + uniform noise up to 0.05, scaled by a quality hint.

### Location
At most +3%, with exponential decay at a 500 km scale.

### Exposure governor
A per-(user, post) rate limiter. If a post's cap is 5%, it is shown at most about 5% of the time.

---

## 3. Design character

- **Smoothing before judgment.** The shrinkage estimator prevents one view from deciding a post's fate. This is the right instinct for cold start.
- **Percentile-based dwell.** Comparing dwell to the user's own history removes the bias between fast and slow readers.
- **Semantic, not lexical, personalization.** Roadmap and taste work across vocabulary, not by keyword.
- **Soft everywhere.** There are no hard filters. All effects are bounded and composable.
- **Cost.** Texts are embedded in one batch with caching. Per-request work is O(P·d), plus O(P log P) for the sort.

---

## 4. Findings: where behavior departs from intent

These come from tracing the code as written, with `algo.py` unchanged.

1. **The roadmap/taste lift is dead code.** `relevanceLift` (up to ×2.2) is computed, then the next line overwrites `combinedRank` without it.
2. **Personalization is weak at default weights.** The effective maxima are roadmap 0.15×0.25 = 0.0375 and taste 0.15×0.30 = 0.045, totaling 0.0825. Quality's floor alone contributes at least 0.079. Content quality therefore outweighs a user's stated goals unless the weights are raised, up to 2.0 via the API.
3. **The exposure cap never fires through the API.** With one deduplicated impression per post, the lowest reachable quality is about (5×0.45−0.10)/6 ≈ 0.36, far above the 0.10 trigger. Only the unit test (50 synthetic impressions) reaches it. Spam is ranked down by score but never throttled.
4. **Unused inputs.** `likeCount`, `replyCount` and `sessionActiveTimeScore` never affect ranking. `impressionCount` only feeds the exploration boost. `roadmapMinimumSimilarity` is defined but never read. `regencyScore` is returned but not used in the final score.
5. **Double counting.** The artifact bonus and grammar penalty each enter once in `contentQualityScore` and again in `bayesianStartingQuality`, so both effectively count twice.
6. **`funKeyDensity` is user-level**, so it shifts all of a user's posts equally and barely changes their order.
7. **Sub-second skips are invisible.** The eligibility gate (at least 1 s) excludes them, so the −0.10 penalty only reaches dwells of 1.0 to 1.5 s.
8. **Rejection scoring is lexical.** It counts distinct patterns, so "no problem, love it" is penalized and "this is awful" is not.
9. **Run-to-run variance.** The exploration noise is unseeded, so scores differ slightly between identical calls (at most about 0.0015 in the final score).
10. **Unbounded state.** The governor's history grows per (user, post) forever. Cap it or move it to Redis at scale.

---

## 5. If you want to tighten it

None of these require restructuring `algo.py`. They could ship as a patched copy:

- Delete the overwriting `combinedRank` line to restore the lift.
- Feed global like and reply rates into the posterior.
- Pass a seed to the exploration function for reproducible ranks.
- Count sub-second skips as negative evidence, so the cap can trigger.

---

## 6. Encouraged behavior and content

| Signal | What it rewards | Strength |
|---|---|---|
| Trend-aligned topics | Posts close to the trending set (default: Python, ML, phones, cooking, fitness, travel) | Moderate |
| Craft | Clean grammar, rich metadata, attached artifacts (each counted twice) | Moderate, capped by the 0.95 ceiling |
| Sustained attention | Dwell over 15 s is worth +1, the biggest passive signal | Strong |
| Conversation | A reply is worth +1, as much as 15 s of attention, whatever its tone | Strong |
| Newness | Freshness, plus an age bump (×1.2 under 10 days) | Strong |
| Niche consistency | An author whose history matches the viewer's roadmap (35% of the match) | Weak to moderate |
| Device fit | Video for charging, fast devices; light posts for old or slow ones | Weak |
| Newcomers | A 0.45 starting floor plus exploration noise gives every post a first look | Gentle |

---

## 7. Discouraged behavior and content

- **Staleness.** Freshness halves at 48 hours and falls to about 15% by day 8.
- **Repetition.** From the third repeat impression, freshness is multiplied by 0.45 and then shrinks further as 3/n.
- **Hostile replies.** Words like *hate*, *worst* and *boring* can cut that post's mastery factor by up to 45% for that viewer.
- **Glances.** Dwell of 1.0 to 1.5 s costs −0.10.
- **Off-trend content.** It is only mildly disadvantaged.
- **Heavy media on weak devices.** It takes ×0.8.
- **Distance.** At most a 3% loss.
- **Spam.** The 5% exposure cap exists, but through the API it rarely engages (finding 3), so spam is ranked down rather than throttled.

---

## 8. What it neither rewards nor punishes

- **Skipping.** Views under 1 s or under 50% visible are ignored. A post everyone scrolls past only decays through age.
- **Popularity.** `likeCount`, `replyCount` and `impressionCount` don't enter quality, so there is no snowball effect.
- **Video completion.** `watchedFraction` and the watch curve are unused. Dwell is what counts.
- **Reading effort.** 16 s on a 10-word post equals 16 s on a 2,000-word one.
- **Truth, sentiment, harm and diversity.** Nothing measures these.
- **Accessibility.** The score is the same for every post, so it never reorders anything.

---

## 9. Second-order effects

1. **Evidence speed depends on length.** A short post (prior weight 5) is judged within a handful of views, while a long one (weight up to 60) needs far more evidence. Short, punchy content gets fast verdicts both ways.
2. **Reply-bait pays.** Reply credit ignores sentiment, so controversy that provokes replies is rewarded. The rejection penalty only dents the replier's own view of the post.
3. **Likes are nearly worthless to farm** (+0.15), while replies and dwell are the levers for gaming. Dwell can be inflated by leaving a screen open, and the one-count-per-user rule only slows sock accounts.
4. **Mainstream gravity.** Trend similarity enters both quality and exploration, nudging the feed toward whatever the trend set contains. Changing `trendingTexts` is the knob that steers culture.
5. **Bubble risk is a dial.** At defaults, roadmap plus taste contribute at most 0.0825. At weight 2.0 each contributes up to 0.30 and personalization dominates. Exploration (3% of the blend) offers little serendipity at any setting.
6. **Anti-snowball by design.** Evidence is per viewer, so no post gets globally louder just because others engaged.

---

## 10. What actually moves the ranking

Approximate maximum swing in the additive score, at default weights:

| Rank | Signal | Swing |
|---|---|---|
| 1 | Content quality | about 0.11 |
| 2 | Freshness | about 0.08 (can outweigh perfect alignment with the user's goals) |
| 3 | Mastery (dwell habits, device, age bump, rejection language) | about 0.05 |
| 4 | Taste | about 0.045 |
| 5 | Roadmap | about 0.04 |
| 6 | Exploration | about 0.02 |
| 7 | Location | about 1%, multiplicative |

---

## 11. Summary

The algorithm favors **fresh, clean, on-trend, attention-holding, conversation-provoking posts from topically consistent authors**, judged per viewer with cautious Bayesian smoothing. It leans toward *newness and quality over personal intent*, and it measures *engagement but not value*.

- **Main strengths:** fair cold start, no popularity snowball, and bounded, hard-to-dominate scoring.
- **Main risks:** reply-bait, mainstream drift and dwell inflation.

[← Back to README](README.md)
