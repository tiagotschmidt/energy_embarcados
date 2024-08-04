#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define pixel(x, y, s) pixels[((y) * s) + (x)]

// #define DEBUG
#define THREADS_PER_BLOCK 256

__global__ void haar(int *pixels, long long size, double SQRT_2) {
  // Calculate thread indices
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Check for boundary conditions
  if (x < size && y < size) {
    long long int s = size;
    while (s > 1) {
      long long int mid = s / 2;

      if (y < mid) { // Process only the upper half in row transformation
        int a = pixel(x, y, s);
        a = (a + pixel(mid + x, y, s)) / SQRT_2;
        int d = pixel(x, y, s);
        d = (d - pixel(mid + x, y, s)) / SQRT_2;
        pixel(x, y, s) = a;
        pixel(mid + x, y, s) = d;
      }

      if (x < mid) { // Process only the left half in column transformation
        int a = pixel(x, y, s);
        a = (a + pixel(x, mid + y, s)) / SQRT_2;
        int d = pixel(x, y, s);
        d = (d - pixel(x, mid + y, s)) / SQRT_2;
        pixel(x, y, s) = a;
        pixel(x, mid + y, s) = d;
      }

      s /= 2;
    }
  }
}

int main(int argc, char *argv[]) {
  long long size;
  double SQRT_2;

  FILE *in;
  FILE *out;

  in = fopen("image.in", "rb");
  if (in == NULL) {
    perror("image.in");
    exit(EXIT_FAILURE);
  }

  out = fopen("image.out", "wb");
  if (out == NULL) {
    perror("image.out");
    exit(EXIT_FAILURE);
  }

  fread(&size, sizeof(size), 1, in);

  fwrite(&size, sizeof(size), 1, out);

  int *pixels = (int *)malloc(size * size * sizeof(int));
  int *pixels_d = nullptr;

  if (!fread(pixels, size * size * sizeof(int), 1, in)) {
    perror("read error");
    exit(EXIT_FAILURE);
  }
  cudaMalloc((void **)pixels_d, sizeof(int) * size * size);
  cudaMemcpy(pixels_d, pixels, size * size, cudaMemcpyHostToDevice);

  SQRT_2 = sqrt(2);
  int blocksPerGridX = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int blocksPerGridY = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  haar<<<blocksPerGridX, blocksPerGridY>>>(pixels_d, size, SQRT_2);

  cudaMemcpy(pixels, pixels_d, size * size, cudaMemcpyDeviceToHost);
  cudaFree(pixels_d);

  fwrite(pixels, size * size * sizeof(int), 1, out);

  free(pixels);

  fclose(out);
  fclose(in);

  return EXIT_SUCCESS;
}
