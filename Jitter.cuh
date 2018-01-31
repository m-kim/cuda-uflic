#ifndef Jitter_H
#define Jitter_H

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
		x = idx_data_tex.get<0>() / dim.x;
		y = idx_data_tex.get<0>() % dim.x;

		T reval = idx_data_tex.get<1>();
		if (y >= 0 && y < dim.y && x >= 0 && x < dim.x) {
			T rnd = idx_data_tex.get<2>();
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
