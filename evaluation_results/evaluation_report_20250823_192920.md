# PayPal Financial Q&A System Evaluation Report

Generated: 2025-08-23 19:29:21

## Executive Summary

**Winner: Fine-Tuned System** (Quality: 0.379 vs 0.270)

## Detailed Metrics

### Overall Performance

| Metric | RAG | Fine-Tuned |
|--------|-----|------------|
| Avg Quality | 0.270 | 0.379 |
| Avg Confidence | 0.700 | 0.750 |
| Avg Response Time | 2.292 | 0.420 |
| Std Quality | 0.284 | 0.316 |
| Total Sources | 27.000 | 0.000 |

### Performance by Category

#### High Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.433 | 0.700 | 2.561 |
| Fine-Tuned | 0.786 | 0.750 | 0.206 |

#### Low Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.228 | 0.700 | 1.942 |
| Fine-Tuned | 0.100 | 0.750 | 0.610 |

#### Irrelevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.150 | 0.700 | 2.371 |
| Fine-Tuned | 0.250 | 0.750 | 0.445 |

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
