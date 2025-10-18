#pragma once
#include "Color.cuh"

#include <iostream>
#include <fstream>

struct Canvas {
    int width, height;
    Color* pixels; // device pointer

    __host__ Canvas(int w, int h) : width(w), height(h) {
        cudaMallocManaged(&pixels, width * height * sizeof(Color));
    }

    __host__ ~Canvas() {
        cudaFree(pixels);
    }

    __host__ __device__
        inline Color& PixelAt(int x, int y) {
        return pixels[y * width + x];
    }
};

// Function declarations
inline void SaveCanvasToPPM(Canvas& c, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) { std::cerr << "Cannot open file\n"; return; }

    file << "P3\n" << c.width << " " << c.height << "\n255\n";

    for (int y = 0; y < c.height; ++y) {
        for (int x = 0; x < c.width; ++x) {
            Color col = c.PixelAt(x, y);
            col.clamp();
            int r = static_cast<int>(col.red * 255 + 0.5f);
            int g = static_cast<int>(col.green * 255 + 0.5f);
            int b = static_cast<int>(col.blue * 255 + 0.5f);
            file << r << " " << g << " " << b << " ";
        }
        file << "\n";
    }

    file.close();
}

__global__
inline void FillBackgroundKernel(Color* pixels, int width, int height, Color bg) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        pixels[y * width + x] = bg;
    }
}
