#pragma once
#include "cuda_runtime.h"

class Intersection {
public:
	float t;
	int shapeIndex;
	float u = 0, v = 0;

	__host__ __device__ Intersection() : t(0), shapeIndex(-1) {};
	__host__ __device__ Intersection(float t, int index) : t(t), shapeIndex(index) {}
};