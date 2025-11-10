#include "cuda include/canvas.cuh"
#include "cuda include/scene.cuh"
#include "cuda include/shape.cuh"
#include "cuda include/lighting.cuh"
#include "cuda include/material.cuh"
#include "cuda include/matrices.cuh"

#include <iostream>

int main()
{
	const int width = 512;
	const int height = 512;
	Canvas canvas(width, height);

	// Create camera
	GPUCamera camera;
	camera.hsize = width;
	camera.vsize = height;
	camera.field_of_view = PI / 3.0f;  // 60 degrees
	
	float half_view = tanf(camera.field_of_view / 2.0f);
	float aspect = static_cast<float>(width) / static_cast<float>(height);
	
	if (aspect >= 1.0f) {
		camera.half_width = half_view;
		camera.half_height = half_view / aspect;
	} else {
		camera.half_width = half_view * aspect;
		camera.half_height = half_view;
	}
	
	camera.pixel_size = (camera.half_width * 2.0f) / width;
	
	// Set camera transform (look from, look at, up)
	camera.transform = View_Transform(
		CreatePoint(0.0f, 1.5f, -5.0f),   // from
		CreatePoint(0.0f, 1.0f, 0.0f),    // to
		CreateVector(0.0f, 1.0f, 0.0f)    // up
	);
	camera.inverse = InvertMatrix(camera.transform);

	// Allocate GPU memory for 6 shapes
	const int numShapes = 6;
	Shape* d_objects;
	cudaMallocManaged(&d_objects, numShapes * sizeof(Shape));

	// Floor
	d_objects[0].type = SPHERE;
	d_objects[0].transform = Scaling(10.0f, 0.01f, 10.0f);
	d_objects[0].inverse = InvertMatrix(d_objects[0].transform);
	d_objects[0].inv_transpose = Transpose(d_objects[0].inverse);
	d_objects[0].mat.color = Color(1.0f, 0.9f, 0.9f);
	d_objects[0].mat.ambient = 0.1f;
	d_objects[0].mat.diffuse = 0.9f;
	d_objects[0].mat.specular = 0.0f;
	d_objects[0].mat.shininess = 200.0f;

	// Left wall
	d_objects[1].type = SPHERE;
	d_objects[1].transform = Translation(0.0f, 0.0f, 5.0f) * 
	                          Rotation_Y(-PI / 4.0f) * 
	                          Rotation_X(PI / 2.0f) * 
	                          Scaling(10.0f, 0.01f, 10.0f);
	d_objects[1].inverse = InvertMatrix(d_objects[1].transform);
	d_objects[1].inv_transpose = Transpose(d_objects[1].inverse);
	d_objects[1].mat = d_objects[0].mat;  // Same as floor

	// Right wall
	d_objects[2].type = SPHERE;
	d_objects[2].transform = Translation(0.0f, 0.0f, 5.0f) * 
	                          Rotation_Y(PI / 4.0f) * 
	                          Rotation_X(PI / 2.0f) * 
	                          Scaling(10.0f, 0.01f, 10.0f);
	d_objects[2].inverse = InvertMatrix(d_objects[2].transform);
	d_objects[2].inv_transpose = Transpose(d_objects[2].inverse);
	d_objects[2].mat = d_objects[0].mat;  // Same as floor

	// Middle sphere (green)
	d_objects[3].type = SPHERE;
	d_objects[3].transform = Translation(-0.5f, 1.0f, 0.5f);
	d_objects[3].inverse = InvertMatrix(d_objects[3].transform);
	d_objects[3].inv_transpose = Transpose(d_objects[3].inverse);
	d_objects[3].mat.color = Color(0.1f, 1.0f, 0.5f);
	d_objects[3].mat.ambient = 0.1f;
	d_objects[3].mat.diffuse = 0.7f;
	d_objects[3].mat.specular = 0.3f;
	d_objects[3].mat.shininess = 200.0f;

	// Right sphere (yellow-green, smaller)
	d_objects[4].type = SPHERE;
	d_objects[4].transform = Translation(1.5f, 0.5f, -0.5f) * 
	                          Scaling(0.5f, 0.5f, 0.5f);
	d_objects[4].inverse = InvertMatrix(d_objects[4].transform);
	d_objects[4].inv_transpose = Transpose(d_objects[4].inverse);
	d_objects[4].mat.color = Color(0.5f, 1.0f, 0.1f);
	d_objects[4].mat.ambient = 0.1f;
	d_objects[4].mat.diffuse = 0.7f;
	d_objects[4].mat.specular = 0.3f;
	d_objects[4].mat.shininess = 200.0f;

	// Left sphere (yellow, smallest)
	d_objects[5].type = SPHERE;
	d_objects[5].transform = Translation(-1.5f, 0.33f, -0.75f) * 
	                          Scaling(0.33f, 0.33f, 0.33f);
	d_objects[5].inverse = InvertMatrix(d_objects[5].transform);
	d_objects[5].inv_transpose = Transpose(d_objects[5].inverse);
	d_objects[5].mat.color = Color(1.0f, 0.8f, 0.1f);
	d_objects[5].mat.ambient = 0.1f;
	d_objects[5].mat.diffuse = 0.7f;
	d_objects[5].mat.specular = 0.3f;
	d_objects[5].mat.shininess = 200.0f;

	// Create light source
	PointLight light;
	light.position = CreatePoint(-10.0f, 10.0f, -10.0f);
	light.intensity = Color(1.0f, 1.0f, 1.0f);

	// Create world
	GPUWorld world;
	world.light = light;
	world.objects = d_objects;
	world.numObjects = numShapes;

	// Launch kernel
	dim3 threads(16, 16);
	dim3 blocks((width + 15) / 16, (height + 15) / 16);

	std::cout << "Rendering scene with " << numShapes << " objects...\n";
	RenderKernel<<<blocks, threads>>>(world, camera, canvas.pixels);
	
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}

	// Save output
	SaveCanvasToPPM(canvas, "ShadowsTrial.ppm");
	std::cout << "Canvas saved to ShadowsTrial.ppm\n";

	// Cleanup
	cudaFree(d_objects);

	return 0;
}