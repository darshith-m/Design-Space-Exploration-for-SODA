#!/bin/bash

set -e
set -o pipefail

docker run -u $(id -u):$(id -g) -v $(pwd):/working_dir --rm agostini01/soda \
tf-mlir-translate \
  --graphdef-to-mlir \
  --tf-input-arrays=x1 \
  --tf-input-data-types=DT_FLOAT \
  --tf-input-shapes=1,32,32,1 \ # Change input shape according to the model
  --tf-output-arrays=Identity \
  $1 \
  -o output/tf.mlir

docker run -u $(id -u):$(id -g) -v $(pwd):/working_dir --rm agostini01/soda \
tf-opt \
  --tf-executor-to-functional-conversion \
  --tf-region-control-flow-to-functional \
  --tf-shape-inference \
  --tf-to-tosa-pipeline \
  output/tf.mlir \
  -o $2