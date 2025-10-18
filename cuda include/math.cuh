#pragma once
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// -----------------------------
// CUDA-compatible Entity class
// -----------------------------
class Entity {
public:
    float x, y, z, w;

    __host__ __device__
        Entity() : x(0), y(0), z(0), w(0) {}

    __host__ __device__
        Entity(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    __host__ __device__
        Entity operator+(const Entity& other) const {
        return Entity(x + other.x, y + other.y, z + other.z, w + other.w);
    }

    __host__ __device__
        Entity operator-(const Entity& other) const {
        return Entity(x - other.x, y - other.y, z - other.z, w - other.w);
    }

    __host__ __device__
        Entity operator*(float scalar) const {
        return Entity(x * scalar, y * scalar, z * scalar, w * scalar);
    }
};

// -----------------------------
// Basic math functions
// -----------------------------
__host__ __device__
inline Entity CreateVector(float x, float y, float z) {
    return Entity(x, y, z, 0.0f);
}

__host__ __device__
inline Entity CreatePoint(float x, float y, float z) {
    return Entity(x, y, z, 1.0f);
}

__host__ __device__
inline bool equal(float a, float b) {
    return fabsf(a - b) < 1e-5f;
}

__host__ __device__
inline Entity Add(Entity a, Entity b) {
    if (a.w + b.w != 2)
        return Entity(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    else
        return Entity(0.0f, 0.0f, 0.0f, 0.0f);

}

__host__ __device__
inline Entity Subtract(Entity a, Entity b) {
    if (a.w - b.w >= 0)
        return Entity(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
    else
        return Entity(0.0f, 0.0f, 0.0f, 0.0f);
}

__host__ __device__
inline Entity Negate(Entity a) { return a * -1.0f; }

__host__ __device__
inline Entity Multiply(Entity a, float scalar) { return a * scalar; }

__host__ __device__
inline Entity Divide(Entity a, float scalar) {
    return Entity(a.x / scalar, a.y / scalar, a.z / scalar, a.w / scalar);
}

__host__ __device__
inline float Magnitude(Entity a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__
inline Entity Normalize(Entity a) {
    float mag = Magnitude(a);
    if (mag == 0) return a;
    return Entity(a.x / mag, a.y / mag, a.z / mag, 0.0f);
}

__host__ __device__
inline float DotProduct(Entity a, Entity b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline Entity CrossProduct(Entity a, Entity b) {
    return Entity(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
        0.0f
    );
}