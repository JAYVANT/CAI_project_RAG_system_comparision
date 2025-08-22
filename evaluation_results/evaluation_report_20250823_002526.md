# PayPal Financial Q&A System Evaluation Report

Generated: 2025-08-23 00:25:27

## Executive Summary

**Winner: Fine-Tuned System** (Quality: 0.403 vs 0.195)

## Detailed Metrics

### Overall Performance

| Metric | RAG | Fine-Tuned |
|--------|-----|------------|
| Avg Quality | 0.195 | 0.403 |
| Avg Confidence | -3.072 | 0.826 |
| Avg Response Time | 3.006 | 2.074 |
| Std Quality | 0.203 | 0.233 |
| Total Sources | 27.000 | 0.000 |

### Performance by Category

#### High Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.292 | -0.291 | 3.308 |
| Fine-Tuned | 0.658 | 0.815 | 2.085 |

#### Low Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.194 | -1.241 | 2.859 |
| Fine-Tuned | 0.383 | 0.835 | 2.100 |

#### Irrelevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.100 | -7.684 | 2.851 |
| Fine-Tuned | 0.167 | 0.829 | 2.037 |

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
