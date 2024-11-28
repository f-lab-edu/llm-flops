#!/bin/bash

cd app/embedding_server

BENTO_TAG=$(bentoml build --output tag)
BENTO_CONTAINER_TAG=$(echo "$BENTO_TAG" | awk -F':' '{print $2":"$3}')

# system이 MacOS인지 확인합니다
if [[ "$(uname)" == "Darwin" ]]; then
    # Apple Silicon에서 돌고 있는지 확인합니다
    if [[ "$(sysctl -n machdep.cpu.brand_string)" == *"Apple"* ]]; then
        echo "This system is running macOS on Apple Silicon."
        bentoml containerize --opt platform=linux/amd64 $BENTO_CONTAINER_TAG
    else
        echo "This system is running macOS but not on Apple Silicon."
    fi
else
    echo "This system is not running macOS."
    bentoml containerize $BENTO_CONTAINER_TAG
fi

# Bento build한 후 생성된 container로 embedding API를 시작합니다
# docker run -it --rm -p 3000:3000 $BENTO_CONTAINER_TAG serve



