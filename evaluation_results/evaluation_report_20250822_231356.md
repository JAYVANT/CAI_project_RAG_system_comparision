# PayPal Financial Q&A System Evaluation Report

Generated: 2025-08-22 23:13:57

## Executive Summary

**Winner: Fine-Tuned System** (Quality: 0.216 vs 0.193)

## Detailed Metrics

### Overall Performance

| Metric | RAG | Fine-Tuned |
|--------|-----|------------|
| Avg Quality | 0.193 | 0.216 |
| Avg Confidence | -3.072 | 0.775 |
| Avg Response Time | 2.908 | 1.887 |
| Std Quality | 0.195 | 0.174 |
| Total Sources | 27.000 | 0.000 |

### Performance by Category

#### High Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.283 | -0.291 | 3.110 |
| Fine-Tuned | 0.347 | 0.765 | 2.026 |

#### Low Confidence Relevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.194 | -1.241 | 2.713 |
| Fine-Tuned | 0.167 | 0.803 | 1.590 |

#### Irrelevant

| System | Quality | Confidence | Time (s) |
|--------|---------|------------|----------|
| RAG | 0.100 | -7.684 | 2.902 |
| Fine-Tuned | 0.133 | 0.757 | 2.044 |

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
