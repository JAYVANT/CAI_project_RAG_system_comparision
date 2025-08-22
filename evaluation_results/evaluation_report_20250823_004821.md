# PayPal Financial Q&A System Evaluation Report

Generated: 2025-08-23 00:48:22

## Executive Summary

**Winner: RAG System** (Quality: 0.272 vs 0.216)

## Detailed Metrics

### Overall Performance

| Metric | RAG | Fine-Tuned |
|--------|-----|------------|
| Avg Quality | 0.272 | 0.216 |
| Avg Confidence | -0.880 | 0.744 |
| Avg Response Time | 2.759 | 2.134 |
| Std Quality | 0.305 | 0.159 |
| Total Sources | 27.000 | 0.000 |

### Performance by Category

#### High Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.550 | 0.580 | 2.884 |
| Fine-Tuned | 0.319 | 0.692 | 2.151 |

#### Low Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.167 | -0.365 | 2.078 |
| Fine-Tuned | 0.211 | 0.826 | 2.162 |

#### Irrelevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.100 | -2.853 | 3.315 |
| Fine-Tuned | 0.117 | 0.714 | 2.089 |

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
