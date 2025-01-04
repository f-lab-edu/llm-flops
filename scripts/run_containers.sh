#!/bin/bash

# Embedding BentoML API 서버 실행 
docker run --rm -p 3001:3000 sentence_embedding_service:latest serve

# LLM BentoML API 서버 실행
docker run --rm -p 3000:3000 -e OPENAI_API_KEY=${OPENAI_API_KEY} -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=${OPENSEARCH_INITIAL_ADMIN_PASSWORD} llm_service:latest serve
