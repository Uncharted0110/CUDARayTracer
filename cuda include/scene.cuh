#pragma once

#include "canvas.cuh"
#include "shape.cuh"
#include "ray.cuh"
#include "color.cuh"
#include "matrices.cuh"
#include "lighting.cuh"

struct GPUCamera {
	int hsize, vsize;
	float field_of_view, half_width, half_height, pixel_size;
	Matrix4x4 transform;
};

struct GPUWorld {
	PointLight light;
	Shape* objects;
	int numObjects;
};

__global__ 
inline void RenderKernel(GPUWorld world, GPUCamera cam, Color* pixels) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= cam.hsize || py >= cam.vsize) return;
    int idx = py * cam.hsize + px;

    // Compute primary ray from camera (deviceCam must contain inverse transform, precomputed)
    Ray ray = RayForPixel_device(cam, px, py);

    // Find nearest hit
    float closestT = 1e30f;
    int hitIdx = -1;
    for (int s = 0; s < world.numObjects; ++s) {
        Intersection h = IntersectShape(world.objects[s], ray, s);
        if (h.t > 0.0f && h.t < closestT) {
            closestT = h.t;
            hitIdx = s;
        }
    }

    Color pixelColor = Color(0, 0, 0);
    if (hitIdx >= 0) {
        Entity hitPoint = ray.Position_At_t(closestT);                      // implement device version
        Entity normal = NormalAtShape(world.objects[hitIdx], hitPoint);
		Entity eye = Negate(ray.direction);
        pixelColor = Lighting(world.objects[hitIdx].mat,
            world.objects[0],
            world.light,
            hitPoint, eye, normal, false);
    }

    pixels[idx] = pixelColor;
}

__device__ 
inline Ray RayForPixel_device(GPUCamera cam, int x, int y) {
    float xoffset = (x + 0.5f) * cam.pixel_size;
    float yoffset = (y + 0.5f) * cam.pixel_size;

    float world_x = cam.half_width - xoffset;
    float world_y = cam.half_height - yoffset;

    Entity pixelPoint = CreatePoint(world_x, world_y, -1.0f);
    Entity origin = CreatePoint(0.0f, 0.0f, 0.0f);
    Entity direction = Normalize(Subtract(pixelPoint, origin));

    return Ray(origin, direction);
}

__device__ 
inline Color TraceRay(GPUWorld world, Ray ray) {
    // Simplified: you can later port your IntersectWorld, ShadeHit, etc. here
    return Color(0.8f, 0.4f, 0.6f);
}