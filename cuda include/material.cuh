#pragma once
#include "color.cuh"

class Material {
public:
	float ambient, diffuse, specular, shininess;
	float reflective, transparency, refractive_index;
	Color color;
	__host__ __device__ Material() :
        color(1, 1, 1),
        ambient(0.1f),
        diffuse(0.9f),
        specular(0.9f),
        shininess(200.0f),
        reflective(0.0f),
        transparency(0.0f),
        refractive_index(1.0f) 
    {}
};