#ifndef BRESENHAM_H
#define BRESENHAM_H
#include <helper_cuda.h>
#include <helper_math.h>

__host__ __device__
bool outside(const float2 &pos, const uint2 dim)
{
return pos.x < 0 || pos.x >= dim.x || pos.y < 0 || pos.y >= dim.y || pos.x != pos.x || pos.y != pos.y;
}

template<class FieldType>
__global__
void drawline(
	FieldType *canvas,
	FieldType *omega,
             const float2 * _p1,
                  const float2 * _p2,
				FieldType *field) {

	uint tid = (blockDim.x * blockIdx.x + threadIdx.x ) + blockDim.x*gridDim.x * (blockDim.y * blockIdx.y + threadIdx.y);
	float2 p1 = _p1[tid];
	float2 p2 = _p2[tid];

    if (!outside(p1, make_uint2(blockDim.x * gridDim.x, blockDim.y*gridDim.y)) && !outside(p2, make_uint2(blockDim.x * gridDim.x, blockDim.y*gridDim.y))){

		FieldType val = canvas[tid]; 
			float2 p = p1;
			float2 d = p2 - p;

			float N = max(fabs(d.x), fabs(d.y));
			if (N < 1e-6) {
				N = 1;
			}

			const float2 s = make_float2(d.x / N, d.y / N);

			for (int i = 0; i<N; i++) {
				if (!outside(make_float2(round(p.x), roundf(p.y)), make_uint2(blockDim.x * gridDim.x, blockDim.y*gridDim.y))) {
					size_t idx = static_cast<size_t>(roundf(p.y))*blockDim.x*gridDim.x + static_cast<size_t>(roundf(p.x));
          //field[idx] += val;//color(255,255,255);
          //omega[idx]++;
          atomicAdd(field + idx, val);
          atomicAdd(omega + idx, 1);
				}
				p += s;
			}

	}
}






#endif
