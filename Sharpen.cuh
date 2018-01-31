#ifndef SHARPEN_H
#define SHARPEN_H

#include <vtkm/Types.h>
#include <vtkm/Math.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/Field.h>
#include <vtkm/cont/ArrayHandle.h>

class Sharpen : public vtkm::worklet::WorkletMapField
{
public:
	Sharpen(vtkm::Id2 &_d) 
		:dim(_d)
	{

	}

	typedef void ControlSignature(FieldIn<>,
																WholeArrayInOut<>,
																FieldOut<>);


	typedef void ExecutionSignature(_1, _2, _3);


	VTKM_EXEC
	vtkm::Id getIdx(vtkm::Id x, vtkm::Id y) const
	{
		return vtkm::Max(vtkm::Min(y, dim[1]-1),static_cast<vtkm::Id>(0)) * dim[0] 
			+ vtkm::Max(vtkm::Min(x, dim[0]-1), static_cast<vtkm::Id>(0));
	}

  template<typename WholeArrayInType, typename FieldOutType>
  VTKM_EXEC
	void operator()(const vtkm::Id &idx,
									const WholeArrayInType &data,
									FieldOutType &reval
									) const
	{
		vtkm::Id x, y;
		y = idx / dim[0];
		x = idx % dim[0];

		FieldOutType left_up = data[getIdx(x - 1, y + 1)];// * float(ttl);
		FieldOutType up = data[getIdx(x, y + 1)];// / float(ttl);
		FieldOutType right_up = data[getIdx(x + 1, y + 1)];// / float(ttl);
		FieldOutType left = data[getIdx(x - 1, y)];// / float(ttl);
		FieldOutType right = data[getIdx(x + 1, y)];// / float(ttl);
		FieldOutType left_down = data[getIdx(x - 1, y - 1)];// / float(ttl);
		FieldOutType down = data[getIdx(x, y - 1)];// / float(ttl);
		FieldOutType right_down = data[getIdx(x + 1, y - 1)];// / float(ttl);
		FieldOutType center = data[getIdx(x, y)];// / float(ttl);


		reval =static_cast<FieldOutType>(9)*center - left_up - up - right_up - left - right - left_down - down - right_down;
	}

	vtkm::Id2 dim;

};

template<typename FieldType, typename DeviceAdapter>
class DoSharpen
{
public:
	DoSharpen(vtkm::Id2 _d)
		:dim(_d)
	{

	}

	void Run(
					vtkm::cont::ArrayHandle<FieldType> &in,
					vtkm::cont::ArrayHandle<FieldType> &out) {
		typedef typename vtkm::worklet::DispatcherMapField<Sharpen>
			SharpenWorkletDispatchType;

		Sharpen sharpenWorklet(dim);
		SharpenWorkletDispatchType dispatch(sharpenWorklet);
		in.PrepareForInPlace(DeviceAdapter());
		out.PrepareForInPlace(DeviceAdapter());
		
		dispatch.Invoke(vtkm::cont::ArrayHandleIndex(dim[0] * dim[1]),
																		in, out);

	}

	vtkm::Id2 dim;
};
#endif
