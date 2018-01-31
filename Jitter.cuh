#ifndef Jitter_H
#define Jitter_H

#include <vtkm/Types.h>
#include <vtkm/Math.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/Field.h>
#include <vtkm/cont/ArrayHandle.h>

class Jitter : public vtkm::worklet::WorkletMapField
{
public:
	Jitter(vtkm::Id2 &_d,
				vtkm::IdComponent bitsize,
				vtkm::Float32 clampLow,
				vtkm::Float32 clampHigh)
		:dim(_d),
		BitSize(bitsize),
		clampingLowerBound(clampLow),
		clampingUpperBound(clampHigh)
	{

	}

	typedef void ControlSignature(FieldIn<>,
																WholeArrayInOut<>,
																WholeArrayInOut<>,

																FieldOut<>);


	typedef void ExecutionSignature(_1, _2, _3, _4);


	VTKM_EXEC
		vtkm::Id getIdx(vtkm::Id x, vtkm::Id y) const
	{
		return vtkm::Max(vtkm::Min(y, dim[1] - 1), static_cast<vtkm::Id>(0)) * dim[0]
			+ vtkm::Max(vtkm::Min(x, dim[0] - 1), static_cast<vtkm::Id>(0));
	}

  template<typename WholeArrayInType, typename FieldOutType>
  VTKM_EXEC
	void operator()(const vtkm::Id &idx,
		const WholeArrayInType &data,
		const WholeArrayInType &tex,
		FieldOutType &reval
		) const
	{
		vtkm::Id x, y;
		y = idx / dim[0];
		x = idx % dim[0];
		reval = data[idx];
		if (y >= 0 && y < dim[1] && x >= 0 && x < dim[0]) {
			FieldOutType rnd = tex[idx];
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
	}

	vtkm::Id2 dim;
	vtkm::IdComponent clampingLowerBound, clampingUpperBound;
	vtkm::IdComponent BitSize;
};

template<typename FieldType, typename DeviceAdapter>
class DoJitter
{
public:
	DoJitter(vtkm::Id2 _d)
		:dim(_d)
	{

	}

	void Run(
		vtkm::cont::ArrayHandle<FieldType> &in,
		vtkm::cont::ArrayHandle<FieldType> &tex,
		vtkm::cont::ArrayHandle<FieldType> &out) {
		typedef typename vtkm::worklet::DispatcherMapField<Jitter>
			JitterWorkletDispatchType;

		Jitter JitterWorklet(dim, 256, 256 * 0.1, 256 * 0.9);
		JitterWorkletDispatchType dispatch(JitterWorklet);
		in.PrepareForInPlace(DeviceAdapter());
		out.PrepareForInPlace(DeviceAdapter());

		dispatch.Invoke(vtkm::cont::ArrayHandleIndex(dim[0] * dim[1]),
			in, tex, out);

	}

	vtkm::Id2 dim;
};
#endif
