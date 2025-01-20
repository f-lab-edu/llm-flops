#!/bin/bash

cd app/llm_server

BENTO_TAG=$(bentoml build --platform linux --output tag)
BENTO_CONTAINER_TAG=$(echo "$BENTO_TAG" | grep "__tag__" | sed -n 's/^__tag__:\(.*\)/\1/p')
REPOSITORY_NAME=$(echo "$BENTO_CONTAINER_TAG" | cut -d':' -f1)

echo "------------------"
echo $BENTO_CONTAINER_TAG
echo $REPOSITORY_NAME
echo "------------------"

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

docker tag $BENTO_CONTAINER_TAG $REPOSITORY_NAME:latest
