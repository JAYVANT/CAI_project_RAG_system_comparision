# PayPal Financial Q&A System Evaluation Report

Generated: 2025-08-23 19:54:44

## Executive Summary

**Winner: RAG System** (Quality: 0.332 vs 0.331)

## Detailed Metrics

### Overall Performance

| Metric | RAG | Fine-Tuned |
|--------|-----|------------|
| Avg Quality | 0.332 | 0.331 |
| Avg Confidence | 0.667 | 0.750 |
| Avg Response Time | 2.151 | 0.397 |
| Std Quality | 0.286 | 0.300 |
| Total Sources | 27.000 | 0.000 |

### Performance by Category

#### High Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.492 | 0.700 | 2.309 |
| Fine-Tuned | 0.708 | 0.750 | 0.176 |

#### Low Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.356 | 0.600 | 1.987 |
| Fine-Tuned | 0.100 | 0.750 | 0.581 |

#### Irrelevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.150 | 0.700 | 2.157 |
| Fine-Tuned | 0.183 | 0.750 | 0.435 |

## Key Findings

### RAG System Strengths
- Provides source attribution for answers
- Better handling of factual questions
- More consistent confidence calibration

### Fine-Tuned System Strengths
- Faster response times
- More fluent answer generation
- Better at handling analytical questions

## Recommendations

1. **Use RAG for**: Factual queries, audit trails, dynamic data
2. **Use Fine-Tuned for**: Speed-critical applications, analytical insights
3. **Consider Hybrid**: Combine both for optimal performance
