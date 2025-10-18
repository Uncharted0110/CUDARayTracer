#pragma once

#include "math.cuh"
#include "color.cuh"
#include "material.cuh"
#include "shape.cuh"

#define BLACK Color(0.0f, 0.0f, 0.0f)

struct PointLight {
	Entity position;
	Color intensity;
};

__device__ Entity Reflect(Entity in, Entity normal) {
	return Subtract(in, Multiply(normal, 2.0f * DotProduct(in, normal)));
}

__device__ Color Lighting(Material mat, const Shape& s, const PointLight& light,
    const Entity& pointPos, const Entity& eye_v,
    const Entity& normal_v, bool in_shadow)
{
    Color color = mat.color;
    Color ambient, diffuse, specular;

    Color effectiveColor = color * light.intensity;
    Entity light_v = Normalize(Subtract(light.position, pointPos));

    ambient = effectiveColor * mat.ambient;

    if (in_shadow)
        return ambient;

    float lightv_dot_normalv = DotProduct(light_v, normal_v);

    if (lightv_dot_normalv < 0.0f) {
        diffuse = BLACK;
        specular = BLACK;
    }
    else {
        diffuse = effectiveColor * mat.diffuse * lightv_dot_normalv;

        Entity reflect_v = Reflect(Negate(light_v), normal_v);
        float reflectv_dot_eyev = DotProduct(reflect_v, eye_v);

        if (reflectv_dot_eyev <= 0.0f)
            specular = BLACK;
        else {
            float factor = powf(reflectv_dot_eyev, mat.shininess);
            specular = light.intensity * mat.specular * factor;
        }
    }

    return ambient + diffuse + specular;
}