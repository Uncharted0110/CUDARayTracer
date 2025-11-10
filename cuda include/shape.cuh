#pragma once

#include "ray.cuh"
#include "material.cuh"
#include "intersection.cuh"
#include "matrices.cuh"

enum ShapeType {
	SPHERE,
	PLANE,
	CUBE,
	CYLINDER
};

struct Shape {
	ShapeType type;
	Material mat;
	Matrix4x4 transform;
	Matrix4x4 inverse;
	Matrix4x4 inv_transpose;
	float minimum, maximum;
	bool closed;

	__host__ __device__
		Shape() :
		type(SPHERE),
		minimum(-INFINITY),
		maximum(INFINITY),
		closed(false)
	{}
};

// transform ray by 4x4 matrix (world->object)
__host__ __device__
inline Ray TransformRayBy(const Ray& r, const Matrix4x4& inv) {
	Entity newOrigin = MultiplyMatrixEntity(inv, r.origin);     // implement MultiplyMatrixEntity(mat, entity)
	Entity newDir = MultiplyMatrixEntity(inv, r.direction);
	return Ray(newOrigin, newDir);
}

__host__ __device__
inline float SphereIntersect(const Ray& ray) {
	Entity sphere_to_ray = Subtract(ray.origin, CreatePoint(0.0f, 0.0f, 0.0f));
	float a = DotProduct(ray.direction, ray.direction);
	float b = 2 * DotProduct(ray.direction, sphere_to_ray);
	float c = DotProduct(sphere_to_ray, sphere_to_ray) - 1;
	float discriminant = b * b - 4 * a * c;

	if(discriminant < 0)
		return -1.0f;
	float t1 = (-b - sqrtf(discriminant)) / (2.0f * a);
	float t2 = (-b + sqrtf(discriminant)) / (2.0f * a);
	
	// Return closest positive t value
	if (t1 >= 0) return t1;
	if (t2 >= 0) return t2;
	return -1.0f;
}

__host__ __device__
inline Entity SphereNormalAt(const Entity& point) {
	return Normalize(Subtract(point, CreatePoint(0.0f, 0.0f, 0.0f)));
}

__host__ __device__
inline float PlaneIntersect(const Ray& ray) {
	if (fabs(ray.direction.y) < 1e-6)
		return -1.0f;
	float t = -ray.origin.y / ray.direction.y;
	return (t >= 0) ? t : -1.0f;
}

__host__ __device__
inline Entity PlaneNormalAt(const Entity& point) {
	return CreateVector(0.0f, 1.0f, 0.0f);
}

__host__ __device__
inline float CubeIntersect(const Ray& ray) {
	auto checkAxis = [](float origin, float direction) {
		float tmin, tmax;
		float t1 = (-1 - origin) / direction;
		float t2 = (1 - origin) / direction;
		if (t1 > t2) {
			float temp = t1;
			t1 = t2;
			t2 = temp;
		}
		tmin = t1;
		tmax = t2;
		struct { float first; float second; } result;
		result.first = tmin;
		result.second = tmax;
		return result;
		};

	auto x = checkAxis(ray.origin.x, ray.direction.x);
	auto y = checkAxis(ray.origin.y, ray.direction.y);
	auto z = checkAxis(ray.origin.z, ray.direction.z);

	float tmin = fmaxf(fmaxf(x.first, y.first), z.first);
	float tmax = fminf(fminf(x.second, y.second), z.second);
	if (tmin > tmax) return -1.0f;
	return tmin;
}

__host__ __device__
inline Entity CubeNormalAt(const Entity& point) {
	float absX = fabsf(point.x);
	float absY = fabsf(point.y);
	float absZ = fabsf(point.z);
	float maxc = fmaxf(fmaxf(absX, absY), absZ);

	if (maxc == absX)
		return CreateVector(point.x, 0.0f, 0.0f);
	else if (maxc == absY)
		return CreateVector(0.0f, point.y, 0.0f);
	else
		return CreateVector(0.0f, 0.0f, point.z);
}


__host__ __device__
inline float CylinderIntersect(const Ray& ray, float minY, float maxY, bool closed)
{
	float a = ray.direction.x * ray.direction.x + ray.direction.z * ray.direction.z;
	if (fabs(a) < 1e-6) 
		return -1.0f;
	
	float b = 2 * (ray.origin.x * ray.direction.x + ray.origin.z * ray.direction.z);
	float c = ray.origin.x * ray.origin.x + ray.origin.z * ray.origin.z - 1;
	float discriminant = b * b - 4 * a * c;
	if(discriminant < 0)
		return -1.0f;

	float t0 = (-b - sqrtf(discriminant)) / (2 * a);
	float t1 = (-b + sqrtf(discriminant)) / (2 * a);

	float y0 = ray.origin.y + t0 * ray.direction.y;
	float y1 = ray.origin.y + t1 * ray.direction.y;

	if (y0 < minY && y1 < minY)	return -1.0f;
	if (y0 > maxY && y1 > maxY)	return -1.0f;

	return (t0 > 0) ? t0 : ((t1 > 0) ? t1 : -1.0f);
}

__host__ __device__
inline Entity CylinderNormalAt(const Entity& point) {
	return CreateVector(point.x, 0.0f, point.z);
}

__host__ __device__
inline Intersection IntersectShape(const Shape& shape, const Ray& worldRay, int index) {
	Ray localRay = TransformRayBy(worldRay, shape.inverse);
	
	float t = -1.0f;
	switch (shape.type)
	{
		case SPHERE:
			t = SphereIntersect(localRay);
			break;
		case PLANE:
			t = PlaneIntersect(localRay);
			break;
		case CUBE:
			t = CubeIntersect(localRay);
			break;
		case CYLINDER:
			t = CylinderIntersect(localRay, shape.minimum, shape.maximum, shape.closed);
			break;
		default:
			break;
	}
	return (t > 0.0f) ? Intersection(t, index) : Intersection(-1.0f, -1);
}

// Compute normal at world-space point on shape
__host__ __device__
inline Entity NormalAtShape(const Shape& shape, const Entity& worldPoint) {
	// transform point to object space
	Entity objectPoint = MultiplyMatrixEntity(shape.inverse, worldPoint);
	Entity objectNormal;
	switch (shape.type) {
	case SPHERE: objectNormal = Subtract(objectPoint, CreatePoint(0, 0, 0)); break;
	case PLANE:  objectNormal = CreateVector(0, 1, 0); break;
	case CUBE:   objectNormal = CubeNormalAt(objectPoint); break;
	case CYLINDER: objectNormal = CylinderNormalAt(objectPoint); break;
	default: objectNormal = CreateVector(0, 0, 0); break;
	}
	// transform normal back to world space using inverse-transpose
	Entity worldNormal = MultiplyMatrixEntity(shape.inv_transpose, objectNormal);
	worldNormal.w = 0.0f;
	return Normalize(worldNormal);
}