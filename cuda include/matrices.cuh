#pragma once
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "math.cuh"

// -----------------------------
// 4x4 Matrix structure
// -----------------------------
struct Matrix4x4 {
    float m[4][4];

    __host__ __device__ inline Matrix4x4() {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    __host__ __device__ inline float* operator[](int row) { return m[row]; }
    __host__ __device__ inline const float* operator[](int row) const { return m[row]; }
};

// -----------------------------
// Basic matrix operations
// -----------------------------
__host__ __device__ inline Matrix4x4 IdentityMatrix() {
    Matrix4x4 M;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            M.m[i][j] = (i == j ? 1.0f : 0.0f);
    return M;
}

__host__ __device__ inline Matrix4x4 Multiply(const Matrix4x4& A, const Matrix4x4& B) {
    Matrix4x4 R;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            R[i][j] = A[i][0] * B[0][j] +
                A[i][1] * B[1][j] +
                A[i][2] * B[2][j] +
                A[i][3] * B[3][j];
        }
    return R;
}

__host__ __device__ inline Entity MultiplyMatrixEntity(const Matrix4x4& mat, const Entity& e) {
    return Entity(
        mat.m[0][0] * e.x + mat.m[0][1] * e.y + mat.m[0][2] * e.z + mat.m[0][3] * e.w,
        mat.m[1][0] * e.x + mat.m[1][1] * e.y + mat.m[1][2] * e.z + mat.m[1][3] * e.w,
        mat.m[2][0] * e.x + mat.m[2][1] * e.y + mat.m[2][2] * e.z + mat.m[2][3] * e.w,
        mat.m[3][0] * e.x + mat.m[3][1] * e.y + mat.m[3][2] * e.z + mat.m[3][3] * e.w
    );
}

__host__ __device__ inline Matrix4x4 Transpose(const Matrix4x4& mat) {
    Matrix4x4 R;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            R.m[i][j] = mat.m[j][i];
    return R;
}

// -----------------------------
// Determinant, Cofactor & Inverse
// -----------------------------

__host__ __device__ inline float Minor3x3(
    float a1, float a2, float a3,
    float b1, float b2, float b3,
    float c1, float c2, float c3)
{
    return a1 * (b2 * c3 - b3 * c2)
        - a2 * (b1 * c3 - b3 * c1)
        + a3 * (b1 * c2 - b2 * c1);
}

__host__ __device__ inline float Determinant(const Matrix4x4& m) {
    // Compute 4x4 determinant using expansion by minors
    float det = 0.0f;

    float sub0 = Minor3x3(
        m.m[1][1], m.m[1][2], m.m[1][3],
        m.m[2][1], m.m[2][2], m.m[2][3],
        m.m[3][1], m.m[3][2], m.m[3][3]);

    float sub1 = Minor3x3(
        m.m[1][0], m.m[1][2], m.m[1][3],
        m.m[2][0], m.m[2][2], m.m[2][3],
        m.m[3][0], m.m[3][2], m.m[3][3]);

    float sub2 = Minor3x3(
        m.m[1][0], m.m[1][1], m.m[1][3],
        m.m[2][0], m.m[2][1], m.m[2][3],
        m.m[3][0], m.m[3][1], m.m[3][3]);

    float sub3 = Minor3x3(
        m.m[1][0], m.m[1][1], m.m[1][2],
        m.m[2][0], m.m[2][1], m.m[2][2],
        m.m[3][0], m.m[3][1], m.m[3][2]);

    det = m.m[0][0] * sub0 - m.m[0][1] * sub1 + m.m[0][2] * sub2 - m.m[0][3] * sub3;
    return det;
}

__host__ __device__ inline Matrix4x4 CofactorMatrix(const Matrix4x4& m) {
    Matrix4x4 cof;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float sub[3][3];
            int subi = 0;
            for (int i = 0; i < 4; ++i) {
                if (i == row) continue;
                int subj = 0;
                for (int j = 0; j < 4; ++j) {
                    if (j == col) continue;
                    sub[subi][subj] = m.m[i][j];
                    subj++;
                }
                subi++;
            }

            float minor = Minor3x3(
                sub[0][0], sub[0][1], sub[0][2],
                sub[1][0], sub[1][1], sub[1][2],
                sub[2][0], sub[2][1], sub[2][2]
            );

            float sign = ((row + col) % 2 == 0) ? 1.0f : -1.0f;
            cof.m[row][col] = sign * minor;
        }
    }
    return cof;
}

__host__ __device__ inline Matrix4x4 InvertMatrix(const Matrix4x4& m) {
    Matrix4x4 cof = CofactorMatrix(m);
    Matrix4x4 adj = Transpose(cof);
    float det = Determinant(m);

    if (fabsf(det) < 1e-8f) return Matrix4x4(); // Return identity if singular

    Matrix4x4 inv;
    float invDet = 1.0f / det;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            inv.m[i][j] = adj.m[i][j] * invDet;

    return inv;
}


// -----------------------------
// Transformation matrices
// -----------------------------
__host__ __device__ inline Matrix4x4 Translation(float x, float y, float z) {
    Matrix4x4 R;
    R.m[0][3] = x; R.m[1][3] = y; R.m[2][3] = z;
    return R;
}

__host__ __device__ inline Matrix4x4 Scaling(float x, float y, float z) {
    Matrix4x4 R;
    R.m[0][0] = x; R.m[1][1] = y; R.m[2][2] = z;
    return R;
}

__host__ __device__ inline Matrix4x4 Rotation_X(float rad) {
    Matrix4x4 R;
    R.m[1][1] = cosf(rad); R.m[1][2] = -sinf(rad);
    R.m[2][1] = sinf(rad); R.m[2][2] = cosf(rad);
    return R;
}

__host__ __device__ inline Matrix4x4 Rotation_Y(float rad) {
    Matrix4x4 R;
    R.m[0][0] = cosf(rad); R.m[0][2] = sinf(rad);
    R.m[2][0] = -sinf(rad); R.m[2][2] = cosf(rad);
    return R;
}

__host__ __device__ inline Matrix4x4 Rotation_Z(float rad) {
    Matrix4x4 R;
    R.m[0][0] = cosf(rad); R.m[0][1] = -sinf(rad);
    R.m[1][0] = sinf(rad); R.m[1][1] = cosf(rad);
    return R;
}

__host__ __device__ inline Matrix4x4 Shearing(float xy, float xz, float yx, float yz, float zx, float zy) {
    Matrix4x4 R;
    R.m[0][1] = xy; R.m[0][2] = xz;
    R.m[1][0] = yx; R.m[1][2] = yz;
    R.m[2][0] = zx; R.m[2][1] = zy;
    return R;
}

__host__ __device__ inline float DegToRad(float deg) {
    return deg * 3.14159265359f / 180.0f;
}
