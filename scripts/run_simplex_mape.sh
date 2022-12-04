#!/bin/bash

# Results on GPU is identical as CPU
build/eval-simplex-mape --gpu Flat
build/eval-simplex-mape --gpu IVF32,Flat
build/eval-simplex-mape --gpu IVF64,Flat
build/eval-simplex-mape --gpu IVF128,Flat
build/eval-simplex-mape --gpu IVF256,Flat
build/eval-simplex-mape --gpu --log2-min-n 11 IVF512,Flat
build/eval-simplex-mape --gpu --log2-min-n 12 IVF1024,Flat
build/eval-simplex-mape HNSW32
build/eval-simplex-mape NSG32
build/eval-simplex-mape LSH
