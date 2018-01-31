#include <vtkm/Math.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/io/writer/VTKDataSetWriter.h>
#include <vtkm/io/reader/VTKDataSetReader.h>
#include <vtkm/rendering/View2D.h>
#include <vtkm/rendering/Canvas.h>
#include <vtkm/rendering/Mapper.h>
#include <vtkm/rendering/Scene.h>
#include <fstream>
#include <sstream>

#include "Evaluator.h"
#include "Integrator.h"
#include "UFLIC.h"
#include "Draw.h"
#include "Sharpen.h"
#include "Normalize.h"
#include "Jitter.h"
#include "Convolve.h"

#include "Reader.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <type_traits>

typedef VTKM_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;

class zero_voxel : public vtkm::worklet::WorkletMapField
{
public:
  typedef void ControlSignature(FieldIn<>, FieldOut<>);
  typedef void ExecutionSignature(_1, WorkIndex, _2);
  //
  VTKM_CONT
  zero_voxel() {}

  template <typename T>
  VTKM_EXEC_CONT void operator()(const vtkm::Id&,
                                 const vtkm::Id& vtkmNotUsed(index),
                                 T& voxel_value) const
  {
    voxel_value = T(0);
  }
};

template<typename FieldType, vtkm::Id Size>
class ResetParticles : public vtkm::worklet::WorkletMapField
{
public:
  typedef  vtkm::Vec<FieldType, Size> VecType;
  typedef void ControlSignature(FieldIn<>, FieldOut<>);
  typedef void ExecutionSignature(_1, _2);
  //
  VTKM_CONT
  ResetParticles(vtkm::Id &d) : dim(d) {}

  VTKM_EXEC_CONT void operator()(const vtkm::Id&idx,
                                 VecType& out) const
  {
    vtkm::Id y = idx / dim;
    vtkm::Id x = idx % dim;
    out = VecType(x + 0.5, y + 0.5);
  }

  vtkm::Id dim;
};


template<typename VecComponentType>
void saveAs(std::string fileName, 
	vtkm::cont::ArrayHandle<VecComponentType > canvasArray, 
	vtkm::Id Width, vtkm::Id Height) {
	std::ofstream of(fileName.c_str(), std::ios_base::binary | std::ios_base::out);
	of << "P6" << std::endl << Width << " " << Height << std::endl << 255 << std::endl;
	//ColorBufferType::PortalConstControl colorPortal = this->ColorBuffer.GetPortalConstControl();
	for (vtkm::Id yIndex = Height - 1; yIndex >= 0; yIndex--)
	{
		for (vtkm::Id xIndex = 0; xIndex < Width; xIndex++)
		{
			VecComponentType val = canvasArray.GetPortalConstControl().Get(yIndex * Width + xIndex);
			
			vtkm::Vec<VecComponentType, 4> tuple(val, val, val, val);
			of << (unsigned char)(tuple[0]);
			of << (unsigned char)(tuple[1]);
			of << (unsigned char)(tuple[2]);
		}
	}
	of.close();
}


int main(int argc, char **argv)
{
  const vtkm::IdComponent ttl = 4, loop_cnt = 12;
  const int Size = 2;
  typedef VTKM_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;
  typedef vtkm::Float32 VecType;
	typedef vtkm::Int32 FieldType;

  typedef vtkm::cont::ArrayHandle<vtkm::Vec<VecType, Size>> VecHandle;


  std::vector<vtkm::Vec<VecType, Size>> vecs;

  //std::shared_ptr<Reader<VecType, Size, ReaderVTK<VecType, Size>>> reader(new ReaderVTK<VecType, Size>("BField_2d.vtk"));
  //std::shared_ptr<Reader<VecType, Size,  ReaderPS<VecType, Size,ReaderXGC<VecType,Size>>>> reader(new ReaderPS<VecType, Size, ReaderXGC<VecType,Size>>("/home/mkim/vtkm-uflic/psi2q/2D_packed/psi2D_packed_normalized_256_99.vec", vtkm::Id2(256,256), Bounds(0,256,0,256)));
  std::shared_ptr<ReaderPS<VecType, Size, ReaderXGC<VecType, Size>>> reader(new ReaderXGC<VecType, Size>("/home/mkim/vtkm-uflic/psi2q/2D_packed/psi2D_packed_512_", vtkm::Id2(512,512), Bounds(0,512,0,512), loop_cnt));
  //std::shared_ptr<ReaderPS<VecType, Size, ReaderXGC<VecType, Size>>> reader(new ReaderXGC<VecType, Size>("XGC_", vtkm::Id2(96,256), Bounds(0,96,0,256)));
  typedef VectorField<VecType,Size> EvalType;


//  int x = 512;
//  int y = 256;
//  if (argc > 1){
//    x = atoi(argv[1]);
//    y = atoi(argv[2]);
//  }
//  std::shared_ptr<Reader<VecType, Size,ReaderCalc<VecType,Size>>> reader(new ReaderCalc<VecType, Size>("XGC_", vtkm::Id2(x,y), Bounds(0,x,0,y)));
//  typedef DoubleGyreField<VecType,Size> EvalType;



  typedef RK4Integrator<EvalType, VecType, Size> IntegratorType;

  typedef ParticleAdvectionWorklet<IntegratorType, VecType, Size, DeviceAdapter> ParticleAdvectionWorkletType;
  typedef DrawLineWorklet<FieldType, VecType, Size, DeviceAdapter> DrawLineWorkletType;

  reader->read(vecs);

  auto t0 = std::chrono::high_resolution_clock::now();

  vtkm::Id2 dim = reader->dim;
  vtkm::Vec<VecType,Size> spacing = reader->spacing;
  Bounds bounds = reader->bounds;

	vtkm::cont::ArrayHandle<vtkm::Vec<VecType, Size>> vecArray;
	std::vector<vtkm::Vec<VecType, Size>> pl[ttl], pr[ttl];

  for (int i = 0; i < ttl; i++) {
    for (int y = 0; y<dim[0]; y++) {
      for (int x = 0; x<dim[1]; x++) {
        pl[i].push_back(vtkm::Vec<VecType, Size>(x + 0.5, y + 0.5));
        pr[i].push_back(vtkm::Vec<VecType, Size>(x + 0.5, y + 0.5));
      }
    }
  }
  std::vector<vtkm::cont::ArrayHandle<vtkm::Vec<VecType, Size>>> sl(ttl), sr(ttl);
	for (int i = 0; i<ttl; i++) {
		sl[i] = vtkm::cont::make_ArrayHandle(pl[i]);
    sr[i] = vtkm::cont::make_ArrayHandle(pr[i]);
	}

	
	vecArray = vtkm::cont::make_ArrayHandle(&vecs[0], vecs.size());


	std::vector<FieldType> canvas[ttl], propertyField[2], omega(dim[0] * dim[1], 0), tex(dim[0] * dim[1], 0);
  vtkm::Float32 t = 0;
  const vtkm::Float32 dt = 0.1;
		for (int i = 0; i < 2; i++) {
		propertyField[i].resize(dim[0] * dim[1], 0);
	}

	for (int i = 0; i < ttl; i++) {
		canvas[i].resize(dim[0] * dim[1], 0);
	}
	for (int i = 0; i < canvas[0].size(); i++) {
		tex[i] = canvas[0][i] = rand() % 255;
	}

	vtkm::cont::ArrayHandle<FieldType > canvasArray[ttl], propFieldArray[2], omegaArray, texArray;
	for (int i = 0; i < ttl; i++) {
		canvasArray[i] = vtkm::cont::make_ArrayHandle(&canvas[i][0], canvas[i].size());
	}
	propFieldArray[0] = vtkm::cont::make_ArrayHandle(&propertyField[0][0], propertyField[0].size());
	propFieldArray[1] = vtkm::cont::make_ArrayHandle(&propertyField[1][0], propertyField[1].size());
	omegaArray = vtkm::cont::make_ArrayHandle(&omega[0], omega.size());
	texArray = vtkm::cont::make_ArrayHandle(&tex[0], tex.size());

  DrawLineWorkletType drawline(bounds, dim);
	DoNormalize<FieldType, DeviceAdapter> donorm(dim);
	DoSharpen<FieldType, DeviceAdapter> dosharp(dim);
	DoJitter<FieldType, DeviceAdapter> dojitter(dim);
  vtkm::cont::ArrayHandleCounting<vtkm::Id> indexArray(vtkm::Id(0), 1, propFieldArray[0].GetNumberOfValues());

  for (int loop = 0; loop < loop_cnt; loop++) {
    EvalType eval(t, Bounds(0, dim[0], 0, dim[1]), spacing);
    IntegratorType integrator(eval, 3.0);
    ParticleAdvectionWorkletType advect(integrator);
    //std::cout << "t: " << t << std::endl;
//		for (int i = 0; i < sr[loop % ttl].GetNumberOfValues(); i++) {
//			vtkm::Id x, y;
//			y = i / dim[0];
//			x = i % dim[0];
//			sl[loop %ttl].GetPortalControl().Set(i, vtkm::Vec<VecType, 2>(x + 0.5, y + 0.5));
//		}
    vtkm::worklet::DispatcherMapField<ResetParticles<VecType,Size>> resetDispatcher(dim[0]);
    resetDispatcher.Invoke(indexArray, sl[loop%ttl]);
		//reset the current canvas
		for (int i = 0; i < canvasArray[loop % ttl].GetNumberOfValues(); i++) {
			canvasArray[loop % ttl].GetPortalControl().Set(i, rand() % 255);
		}

//		for (int i = 0; i < propFieldArray[0].GetNumberOfValues(); i++) {
//			propFieldArray[0].GetPortalControl().Set(i, 0);
//			propFieldArray[1].GetPortalControl().Set(i, 0);
//			omegaArray.GetPortalControl().Set(i, 0);
//		}

    vtkm::worklet::DispatcherMapField<zero_voxel> zeroDispatcher;
    zeroDispatcher.Invoke(indexArray, propFieldArray[0]);
    zeroDispatcher.Invoke(indexArray, propFieldArray[1]);
    zeroDispatcher.Invoke(indexArray, omegaArray);

		for (int i = 0; i < vtkm::Min(ttl, loop+1); i++) {
			advect.Run(sl[i], sr[i], vecArray);
			drawline.Run(canvasArray[i], propFieldArray[0], omegaArray, sl[i], sr[i]);
		}

		sr.swap(sl);

		donorm.Run(propFieldArray[0], omegaArray, propFieldArray[1]);
    std::stringstream fn;
    fn << "uflic-" << loop << ".pnm";
    saveAs(fn.str().c_str(), propFieldArray[1], dim[0], dim[1]);

    //REUSE omegaArray as a temporary cache to sharpen
    dosharp.Run(propFieldArray[1], omegaArray);
    dojitter.Run(omegaArray, texArray, canvasArray[(loop) % ttl]);

    t += dt;// / (vtkm::Float32)ttl + 1.0 / (vtkm::Float32)ttl;
    reader->next(vecs);
    vecArray = vtkm::cont::make_ArrayHandle(&vecs[0], vecs.size());

	}

  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Finished dt: " << dt << " cnt: " << loop_cnt << " time: " << std::chrono::duration<double>(t1-t0).count() << "s" << std::endl;
    std::stringstream fn;
    fn << "uflic-final" << ".pnm";
    saveAs(fn.str().c_str(), propFieldArray[1], dim[0], dim[1]);


	//vtkm::rendering::Mapper mapper;
	//vtkm::rendering::Canvas canvas(512, 512);
	//vtkm::rendering::Scene scene;

	//scene.AddActor(vtkm::rendering::Actor(
	//	ds.GetCellSet(), ds.GetCoordinateSystem(), ds.GetField(fieldNm), colorTable));
	//vtkm::rendering::Camera camera;
	//SetCamera<ViewType>(camera, ds.GetCoordinateSystem().GetBounds());

	//vtkm::rendering::View2D view;
}
