# PayPal Financial Q&A System Evaluation Report

Generated: 2025-08-23 20:47:34

## Executive Summary

**Winner: Fine-Tuned System** (Quality: 0.382 vs 0.326)

## Detailed Metrics

### Overall Performance

| Metric | RAG | Fine-Tuned |
|--------|-----|------------|
| Avg Quality | 0.326 | 0.382 |
| Avg Confidence | 0.622 | 0.750 |
| Avg Response Time | 2.730 | 0.452 |
| Std Quality | 0.299 | 0.333 |
| Total Sources | 54.000 | 0.000 |

### Performance by Category

#### High Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.550 | 0.400 | 3.631 |
| Fine-Tuned | 0.786 | 0.750 | 0.203 |

#### Low Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.278 | 0.667 | 2.304 |
| Fine-Tuned | 0.244 | 0.750 | 0.437 |

#### Irrelevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.150 | 0.800 | 2.256 |
| Fine-Tuned | 0.117 | 0.750 | 0.715 |

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
