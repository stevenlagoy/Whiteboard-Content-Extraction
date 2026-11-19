#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAFFE_ROOT="$PROJECT_ROOT/accessmath-textboxes"

export PYTHONPATH="$CAFFE_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CAFFE_ROOT/.build_release/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "TextBoxes environment configured."
echo "Caffe directory: $CAFFE_ROOT"
