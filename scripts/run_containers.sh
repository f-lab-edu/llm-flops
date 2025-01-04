#!/bin/bash

# Embedding BentoML API 서버 실행 
docker run --rm -p 3001:3000 sentence_embedding_service:latest serve

# LLM BentoML API 서버 실행
docker run --rm -p 3000:3000 llm_service:latest serve