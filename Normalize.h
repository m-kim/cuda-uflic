#ifndef Normalize_H
#define Normalize_H




class Normalize : public vtkm::worklet::WorkletMapField
{
public:
	Normalize(vtkm::Id2 &_d) 
		:dim(_d)
	{

	}

	typedef void ControlSignature(FieldInOut<>,
																FieldInOut<>,
																FieldInOut<>);


	typedef void ExecutionSignature(_1, _2, _3);


	VTKM_EXEC
	vtkm::Id getIdx(vtkm::Id x, vtkm::Id y) const
	{
		return vtkm::Max(vtkm::Min(y, dim[1]-1),static_cast<vtkm::Id>(0)) * dim[0] 
			+ vtkm::Max(vtkm::Min(x, dim[0]-1), static_cast<vtkm::Id>(0));
	}

  template<typename FieldInType, typename FieldOutType>
  VTKM_EXEC
	void operator()(FieldInType &canvas,
									FieldInType &omega,
									FieldOutType &reval
									) const
	{
		reval = 0;
		if (omega > 0)
			reval = canvas / omega;
	}

	vtkm::Id2 dim;

};

template<typename FieldType, typename DeviceAdapter>
class DoNormalize
{
public:
	DoNormalize(vtkm::Id2 _d)
		:dim(_d)
	{

	}

	void Run(
				  vtkm::cont::ArrayHandle<FieldType> &canvas,
					vtkm::cont::ArrayHandle<FieldType> &omega,
					vtkm::cont::ArrayHandle<FieldType> &out) {
		typedef typename vtkm::worklet::DispatcherMapField<Normalize>
			NormalizeWorkletDispatchType;

		
		Normalize NormalizeWorklet(dim);
		NormalizeWorkletDispatchType dispatch(NormalizeWorklet);
		canvas.PrepareForInPlace(DeviceAdapter());
		omega.PrepareForInPlace(DeviceAdapter());
		out.PrepareForInPlace(DeviceAdapter());

		dispatch.Invoke(canvas,omega, out);

	}

	vtkm::Id2 dim;
};
#endif
