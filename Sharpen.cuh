#ifndef SHARPEN_H
#define SHARPEN_H

__host__ __device__
size_t getIdx(uint x, uint y,
	const uint2 dim)
{
		return max(min(y, dim.y - 1), static_cast<uint>(0)) * dim.x
		+ max(min(x, dim.x - 1), static_cast<uint>(0));
}

template<class FieldType, class FieldOutType>
__global__
void sharpen(
	const FieldType *data,
	FieldOutType *reval
	) 
{
	uint x, y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	uint2 dim = make_uint2(blockDim.x * gridDim.x, blockDim.y * gridDim.y);

	FieldOutType left_up = data[getIdx(x - 1, y + 1, dim)];// * float(ttl);
	FieldOutType up = data[getIdx(x, y + 1, dim)];// / float(ttl);
	FieldOutType right_up = data[getIdx(x + 1, y + 1, dim)];// / float(ttl);
	FieldOutType left = data[getIdx(x - 1, y, dim)];// / float(ttl);
	FieldOutType right = data[getIdx(x + 1, y, dim)];// / float(ttl);
	FieldOutType left_down = data[getIdx(x - 1, y - 1, dim)];// / float(ttl);
	FieldOutType down = data[getIdx(x, y - 1, dim)];// / float(ttl);
	FieldOutType right_down = data[getIdx(x + 1, y - 1, dim)];// / float(ttl);
	FieldOutType center = data[getIdx(x, y, dim)];// / float(ttl);


	reval[getIdx(x, y, dim)] =static_cast<FieldOutType>(9)*center - left_up - up - right_up - left - right - left_down - down - right_down;
}

#endif
