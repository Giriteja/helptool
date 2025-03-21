#!/bin/bash

# Download and install Poplar SDK
POPLAR_SDK_URL="https://downloads.graphcore.ai/direct/..."  # Replace with the exact URL for your desired version
wget -q $POPLAR_SDK_URL -O poplar_sdk.tar.gz
mkdir -p poplar_sdk
tar -xf poplar_sdk.tar.gz -C poplar_sdk --strip-components=1
cd poplar_sdk
./install.sh
cd ..

# Set up environment variables
export POPLAR_SDK_ENABLED=true
