#!/bin/bash

# Results on GPU is identical as CPU
build/eval-knn-recall --gpu Flat
build/eval-knn-recall --gpu IVF32,Flat
build/eval-knn-recall --gpu IVF64,Flat
build/eval-knn-recall --gpu IVF128,Flat
build/eval-knn-recall --gpu IVF256,Flat
build/eval-knn-recall --gpu --log2-min-n 11 IVF512,Flat
build/eval-knn-recall --gpu --log2-min-n 12 IVF1024,Flat
build/eval-knn-recall HNSW32
build/eval-knn-recall NSG32
build/eval-knn-recall LSH
