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
	Matrix4x4 inverse;  // Add inverse for ray transformation
};

struct GPUWorld {
	PointLight light;
	Shape* objects;
	int numObjects;
};

__device__
inline Ray RayForPixel_device(GPUCamera cam, int x, int y) {
    float xoffset = (x + 0.5f) * cam.pixel_size;
    float yoffset = (y + 0.5f) * cam.pixel_size;

    float world_x = cam.half_width - xoffset;
    float world_y = cam.half_height - yoffset;

    Entity pixelPoint = CreatePoint(world_x, world_y, -1.0f);
    Entity origin = CreatePoint(0.0f, 0.0f, 0.0f);
    
    // Transform ray by camera's inverse transform
    Entity worldPixel = MultiplyMatrixEntity(cam.inverse, pixelPoint);
    Entity worldOrigin = MultiplyMatrixEntity(cam.inverse, origin);
    Entity direction = Normalize(Subtract(worldPixel, worldOrigin));

    return Ray(worldOrigin, direction);
}

// Check if a point is in shadow
__device__
inline bool IsShadowed(const GPUWorld& world, const Entity& point) {
    // Vector from point to light source
    Entity v = Subtract(world.light.position, point);
    
    // Magnitude of that vector and its direction
    float distance = Magnitude(v);
    Entity direction = Normalize(v);
    
    // Create a ray from the point to the light source
    Ray shadowRay(point, direction);
    
    // Check if any object blocks the light
    for (int s = 0; s < world.numObjects; ++s) {
        Intersection h = IntersectShape(world.objects[s], shadowRay, s);
        
        // If we hit something between the point and the light, we're in shadow
        if (h.t > 0.007f && h.t < distance) {  // 0.005f epsilon to avoid self-intersection
            return true;
        }
    }
    
    return false;
}

__global__ 
inline void RenderKernel(GPUWorld world, GPUCamera cam, Color* pixels) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= cam.hsize || py >= cam.vsize) return;
    int idx = py * cam.hsize + px;

    // Compute primary ray from camera
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

    Color pixelColor = Color(0.1f, 0.1f, 0.2f);  // Dark blue background
    if (hitIdx >= 0) {
        Entity hitPoint = ray.Position_At_t(closestT);
        Entity normal = NormalAtShape(world.objects[hitIdx], hitPoint);
        Entity eye = Negate(ray.direction);
        
        // Compute shadow point slightly offset from surface to avoid self-intersection
        Entity over_point = Add(hitPoint, Multiply(normal, 0.0001f));
        
        // Check if point is in shadow
        bool in_shadow = IsShadowed(world, over_point);
        
        pixelColor = Lighting(world.objects[hitIdx].mat,
            world.objects[hitIdx],
            world.light,
            over_point, eye, normal, in_shadow);
    }

    pixels[idx] = pixelColor;
}
