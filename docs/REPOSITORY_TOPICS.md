# Repository Topics for SEO Optimization

This document provides guidance on setting up GitHub repository topics to optimize search engine visibility and discoverability.

## Overview

Repository topics are keywords that help categorize your GitHub repository and make it easier for users to find. They appear on your repository's main page and are indexed by GitHub's search engine and external search engines.

## Recommended Topics

The following 20 topics have been carefully selected to maximize SEO impact and discoverability:

1. **llm-training** - Primary use case
2. **gpu-memory-calculator** - Unique identifier
3. **deep-learning** - Broad category
4. **pytorch** - Main framework
5. **deepspeed** - Key technology
6. **large-language-models** - Target domain
7. **machine-learning** - Broad category
8. **transformer-models** - Architecture type
9. **distributed-training** - Key feature
10. **memory-optimization** - Core problem solved
11. **megatron-lm** - Supported framework
12. **zero-optimization** - Technical feature
13. **pytorch-fsdp** - Supported technology
14. **llm** - Short form keyword
15. **gpu-computing** - Hardware category
16. **tensor-parallelism** - Technical feature
17. **ai-infrastructure** - Infrastructure category
18. **developer-tools** - Tool classification
19. **nlp** - Application domain
20. **model-training** - General use case

## How to Add Topics

### Method 1: GitHub Web Interface (Recommended)

1. Navigate to the repository: https://github.com/George614/gpu-mem-calculator
2. Look for the "About" section on the right sidebar
3. Click the gear icon (⚙️) to edit repository details
4. In the "Topics" field, add the recommended topics listed above
5. Click "Save changes"

### Method 2: GitHub CLI

If you have the GitHub CLI installed and authenticated:

```bash
gh repo edit George614/gpu-mem-calculator \
  --add-topic "llm-training" \
  --add-topic "gpu-memory-calculator" \
  --add-topic "deep-learning" \
  --add-topic "pytorch" \
  --add-topic "deepspeed" \
  --add-topic "large-language-models" \
  --add-topic "machine-learning" \
  --add-topic "transformer-models" \
  --add-topic "distributed-training" \
  --add-topic "memory-optimization" \
  --add-topic "megatron-lm" \
  --add-topic "zero-optimization" \
  --add-topic "pytorch-fsdp" \
  --add-topic "llm" \
  --add-topic "gpu-computing" \
  --add-topic "tensor-parallelism" \
  --add-topic "ai-infrastructure" \
  --add-topic "developer-tools" \
  --add-topic "nlp" \
  --add-topic "model-training"
```

### Method 3: GitHub API

Using curl with a personal access token:

```bash
curl -X PUT \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/George614/gpu-mem-calculator/topics \
  -d '{
    "names": [
      "llm-training",
      "gpu-memory-calculator",
      "deep-learning",
      "pytorch",
      "deepspeed",
      "large-language-models",
      "machine-learning",
      "transformer-models",
      "distributed-training",
      "memory-optimization",
      "megatron-lm",
      "zero-optimization",
      "pytorch-fsdp",
      "llm",
      "gpu-computing",
      "tensor-parallelism",
      "ai-infrastructure",
      "developer-tools",
      "nlp",
      "model-training"
    ]
  }'
```

## SEO Benefits

Adding these topics provides several SEO benefits:

### 1. **Improved GitHub Search Ranking**
   - Topics are indexed by GitHub's search algorithm
   - Higher visibility in GitHub search results
   - Better matching with user queries

### 2. **Enhanced Discoverability**
   - Appears in topic pages (e.g., github.com/topics/llm-training)
   - Shows up in "Similar repositories" recommendations
   - Featured in GitHub Explore for relevant topics

### 3. **External Search Engine Optimization**
   - Topics appear in page metadata
   - Indexed by Google, Bing, and other search engines
   - Improves organic search rankings

### 4. **Community Engagement**
   - Users can follow specific topics
   - Increases repository visibility to interested developers
   - Builds community around shared interests

### 5. **Cross-Repository Discovery**
   - Links repository to related projects
   - Appears in topic-based repository lists
   - Facilitates collaboration opportunities

## Topic Selection Strategy

Our topic selection follows these principles:

### 1. **Relevance** (40% weight)
   - Must accurately describe repository functionality
   - Should match user search intent
   - Aligned with repository content

### 2. **Search Volume** (30% weight)
   - Topics with high search frequency
   - Balance between popular and niche terms
   - Consider trending technologies

### 3. **Competition** (20% weight)
   - Mix of broad and specific topics
   - Unique identifiers (e.g., "gpu-memory-calculator")
   - Lower competition for niche terms

### 4. **Longevity** (10% weight)
   - Technologies with staying power
   - Avoid overly trendy/temporary terms
   - Focus on established standards

## Topic Categories Breakdown

### Primary Categories (Core Function)
- `llm-training` - Main use case
- `gpu-memory-calculator` - Tool identity
- `memory-optimization` - Problem solved
- `model-training` - General application

### Technology Stack
- `pytorch` - Main framework
- `deepspeed` - Key optimization library
- `megatron-lm` - Parallelism framework
- `pytorch-fsdp` - Distributed training
- `cuda` - GPU programming
- `nvidia-gpu` - Hardware platform

### ML/AI Domain
- `deep-learning` - Field category
- `machine-learning` - Broader field
- `large-language-models` - Specific domain
- `transformer-models` - Architecture type
- `nlp` - Application area

### Technical Features
- `distributed-training` - Key capability
- `zero-optimization` - Specific technique
- `tensor-parallelism` - Parallelism method
- `gpu-computing` - Compute paradigm

### Tool Classification
- `developer-tools` - Tool category
- `ai-infrastructure` - Infrastructure type

## Monitoring & Analytics

### Tracking Topic Performance

1. **GitHub Insights**
   - Monitor traffic sources from topics
   - Check referral traffic from topic pages
   - Track repository views and clones

2. **Search Rankings**
   - Monitor Google search rankings for key terms
   - Track GitHub search positioning
   - Analyze click-through rates

3. **Community Engagement**
   - Track stars and forks growth
   - Monitor issue and PR activity
   - Observe community discussions

### Regular Review

- **Quarterly**: Review topic performance and trends
- **Semi-annually**: Update topics based on new features
- **Annually**: Comprehensive SEO audit and optimization

## Best Practices

### Do's ✅
- Use lowercase and hyphens (e.g., `large-language-models`)
- Keep topics relevant to repository content
- Mix broad and specific topics
- Include framework/technology names
- Add domain-specific terms

### Don'ts ❌
- Don't exceed 20 topics (GitHub limit)
- Avoid spaces in topics
- Don't use misleading or unrelated topics
- Avoid overly generic single words (e.g., "code", "software")
- Don't spam with duplicate meanings

## Alternative Topics (Future Considerations)

If the repository evolves or you want to test different topics:

### Additional Technical Topics
- `activation-checkpointing`
- `gradient-accumulation`
- `mixed-precision-training`
- `multi-node-training`
- `inference-engine`

### Additional Use Case Topics
- `cost-optimization`
- `capacity-planning`
- `oom-error-prevention`
- `training-efficiency`
- `gpu-selection`

### Additional Platform Topics
- `huggingface`
- `wandb-integration`
- `mlops`
- `ml-engineering`
- `research-tools`

## Related Resources

- [GitHub Topics Documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics)
- [GitHub Search Documentation](https://docs.github.com/en/search-github/searching-on-github)
- [SEO Best Practices for GitHub](https://github.blog/2013-05-16-repository-metadata-and-plugin-support/)

## Questions?

If you have questions about repository topics or SEO optimization, please:
- Open an issue: https://github.com/George614/gpu-mem-calculator/issues
- Start a discussion: https://github.com/George614/gpu-mem-calculator/discussions
- Check the FAQ: docs/FAQ.md

---

Last Updated: January 2026
