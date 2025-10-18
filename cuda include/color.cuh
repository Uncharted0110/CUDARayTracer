#pragma once
#include "cuda_runtime.h"
#include <cmath>

struct Color {
    float red, green, blue;

    __host__ __device__
        Color() : red(0), green(0), blue(0) {}

    __host__ __device__
        Color(float r, float g, float b) : red(r), green(g), blue(b) {}

    __host__ __device__
        Color operator+(const Color& o) const {
        return Color(red + o.red, green + o.green, blue + o.blue);
    }

    __host__ __device__
        Color operator-(const Color& o) const {
        return Color(red - o.red, green - o.green, blue - o.blue);
    }

    __host__ __device__
        Color operator*(const Color& o) const {
        return Color(red * o.red, green * o.green, blue * o.blue);
    }

    __host__ __device__
        Color operator*(float scalar) const {
        return Color(red * scalar, green * scalar, blue * scalar);
    }

    __host__ __device__
        void clamp() {
        red = fminf(fmaxf(red, 0.0f), 1.0f);
        green = fminf(fmaxf(green, 0.0f), 1.0f);
        blue = fminf(fmaxf(blue, 0.0f), 1.0f);
    }
};
