#pragma once
#include "math.cuh"

class Ray {
public:
	Entity origin, direction;

	__host__ __device__
		Ray() : origin(), direction() {}

	__host__ __device__
		Ray(const Entity& o, const Entity& d) : origin(o), direction(d) {}

	__host__ __device__ inline Entity Position_At_t(float t) const {
		return Add(origin, Multiply(direction, t));
	}
};