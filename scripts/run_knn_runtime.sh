#!/bin/bash

build/eval-knn-runtime Flat
build/eval-knn-runtime IVF32,Flat
build/eval-knn-runtime IVF64,Flat
build/eval-knn-runtime IVF128,Flat
build/eval-knn-runtime IVF256,Flat
build/eval-knn-runtime --log2-min-n 11 IVF512,Flat
build/eval-knn-runtime --log2-min-n 12 IVF1024,Flat
build/eval-knn-runtime --gpu Flat
build/eval-knn-runtime --gpu IVF32,Flat
build/eval-knn-runtime --gpu IVF64,Flat
build/eval-knn-runtime --gpu IVF128,Flat
build/eval-knn-runtime --gpu IVF256,Flat
build/eval-knn-runtime --gpu --log2-min-n 11 IVF512,Flat
build/eval-knn-runtime --gpu --log2-min-n 12 IVF1024,Flat
build/eval-knn-runtime HNSW32
build/eval-knn-runtime NSG32
build/eval-knn-runtime LSH
