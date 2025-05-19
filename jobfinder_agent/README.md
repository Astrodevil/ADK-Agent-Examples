# Resume Analyzer & Job Finder Agent

An AI-powered pipeline that analyzes resumes and finds relevant job listings. Full explainer video is available on [YouTube](https://www.youtube.com/watch?v=ji_hECcyTjs)



## Overview

This agent demonstrates a 4-agent sequential pipeline that:
- Extracts content from uploaded Resume PDFs using Mistral OCR 
- Prepares tailored job search queries for Hacker News and Wellfound (AngelList)
- Uses Linkup API to search for jobs based on the queries
- Compiles and formats relevant job listings, prioritizing those with higher chances of selection

## Technical Pattern

Uses a 4-agent sequential pipeline:
1. **MistralOCRAgent**: Extracts text from uploaded PDF resumes using Mistral OCR.
2. **QueryPrepAgent**: Prepares tailored job search queries based on the extracted resume content.
3. **LinkupSearchAgent**: Uses the Linkup API to search for relevant jobs on Hacker News and Wellfound.
4. **JobFilterAgent**: Compiles and formats the job listings, prioritizing them based on relevance and experience requirements.

## Add API Keys

```
NEBIUS_API_KEY="your_nebius_api_key_here"
NEBIUS_API_BASE="https://api.studio.nebius.ai/v1"
MISTRAL_API_KEY="your_mistral_api_key_here"
LINKUP_API_KEY="your_linkup_api_key_here"
```

## Usage

1. Upload your resume in PDF format.
2. Execute the `run_ai_analysis()` function in the notebook.
3. The agent pipeline will process your resume and display a list of relevant job postings.


## Required API Keys

- [Nebius AI](https://dub.sh/AIStudio) - For LLM inference (Qwen3-14B).
- [Mistral AI](https://mistral.ai) - For OCR processing.
- [Linkup](https://www.linkup.so/) - For job search.
