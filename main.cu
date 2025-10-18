#include "cuda include/canvas.cuh"

#include <iostream>

int main()
{
	Canvas canvas(800, 600);
	Color bg(0.8f, 0.4f, 0.6f);

	dim3 threads(16, 16);
	dim3 blocks((canvas.width + 15) / 16, (canvas.height + 15) / 16);

	FillBackgroundKernel << <blocks, threads >> > (canvas.pixels, canvas.width, canvas.height, bg);
	cudaDeviceSynchronize();

	SaveCanvasToPPM(canvas, "output.ppm");

	std::cout << "Canvas saved to output.ppm\n";
	return 0;
}