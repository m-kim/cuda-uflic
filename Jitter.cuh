#ifndef Jitter_H
#define Jitter_H

#include <thrust/tuple.h>

template<class T>
class Jitter
{
public:
	Jitter(uint2 &_d,
				T bitsize,
				T clampLow,
				T clampHigh)
		:dim(_d),
		BitSize(bitsize),
		clampingLowerBound(clampLow),
		clampingUpperBound(clampHigh)
	{

	}

	__device__ __host__
		size_t getIdx(uint x, uint y) const
	{
		return max(min(y, dim.x - 1), static_cast<uint>(0)) * dim.x
			+ max(min(x, dim.x - 1), static_cast<uint>(0));
	}


	__host__ __device__
	T operator()(const thrust::tuple<uint, T,T> &idx_data_tex) const
	{
		uint x, y;
    x = thrust::get<0>(idx_data_tex) / dim.x;
    y = thrust::get<0>(idx_data_tex) % dim.x;

    T reval = thrust::get<1>(idx_data_tex);
		if (y >= 0 && y < dim.y && x >= 0 && x < dim.x) {
      T rnd = thrust::get<2>(idx_data_tex);
			if (rnd > BitSize / 2)
				rnd -= BitSize / 2;

			if (reval > BitSize / 2)
				reval = BitSize / 2 + rnd;
			else
				reval = rnd;

			if (reval > clampingUpperBound)
				reval = BitSize;
			else if (reval < clampingLowerBound)
				reval = 0;
			//else
			//	reval = (reval - clampingLowerBound) / (clampingUpperBound - clampingLowerBound);
		}
		return reval;
	}

	uint2 dim;
	T clampingLowerBound, clampingUpperBound;
	T BitSize;
};

#endif
