#include <fstream>
#include <sstream>

#include <helper_cuda.h>
#include <helper_math.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <type_traits>

#include "Draw.cuh"
#include "Sharpen.cuh"
#include "Jitter.cuh"
#include "Evaluator.cuh"
#include "Integrator.cuh"
#include "UFLIC.cuh"
#include "Reader.h"

template<class T>
struct normale {
	__host__ __device__ T operator()(const T &x, const T &y) const {
		T reval = 0;
		if (x > 0)
			reval = x / y;

		return reval;
	}
};


struct resetParticles {
	resetParticles(uint2 _d) { dim = _d; }
	__host__ __device__ float2 operator()(const uint &idx) {
		 uint y = idx / dim.x;
		 uint x = idx % dim.x;
		 return make_float2(x + 0.5, y + 0.5);

	}
	
	uint2 dim;
};

struct prg
{
	unsigned char a, b;

	__host__ __device__
		prg(unsigned char _a = 0.f, unsigned char _b = 1.f) : a(_a), b(_b) {};

	__host__ __device__
		float operator()(const unsigned int n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<unsigned char> dist(a, b);
		rng.discard(n);

		return dist(rng);
	}
};

template<typename VecComponentType>
void saveAs(std::string fileName, 
	thrust::host_vector<VecComponentType> canvas, 
	size_t Width, size_t Height) {
	std::ofstream of(fileName.c_str(), std::ios_base::binary | std::ios_base::out);
	of << "P6" << std::endl << Width << " " << Height << std::endl << 255 << std::endl;
	//ColorBufferType::PortalConstControl colorPortal = this->ColorBuffer.GetPortalConstControl();
	for (size_t yIndex = Height - 1; yIndex >= 0; yIndex--)
	{
		for (size_t xIndex = 0; xIndex < Width; xIndex++)
		{
			VecComponentType val = canvas[yIndex * Width + xIndex];
			
			uint4 tuple = make_uint4(val, val, val, val);
			of << (unsigned char)(tuple.x);
			of << (unsigned char)(tuple.y);
			of << (unsigned char)(tuple.z);
		}
	}
	of.close();
}


int main(int argc, char **argv)
{
	const size_t ttl = 4, loop_cnt = 12;
	typedef float VecType;
	typedef unsigned char FieldType;

  typedef float2 VecField;


  thrust::host_vector<VecField> vecs;
  thrust::device_vector<VecField> d_vecs;
  //std::shared_ptr<Reader<VecType, Size, ReaderVTK<VecType, Size>>> reader(new ReaderVTK<VecType, Size>("BField_2d.vtk"));
  //std::shared_ptr<Reader<VecType, Size,  ReaderPS<VecType, Size,ReaderXGC<VecType,Size>>>> reader(new ReaderPS<VecType, Size, ReaderXGC<VecType,Size>>("/home/mkim/vtkm-uflic/psi2q/2D_packed/psi2D_packed_normalized_256_99.vec", vtkm::Id2(256,256), Bounds(0,256,0,256)));
  //std::shared_ptr<ReaderPS<VecType, Size, ReaderXGC<VecType, Size>>> reader(new ReaderXGC<VecType, Size>("/home/mkim/vtkm-uflic/psi2q/2D_packed/psi2D_packed_512_", vtkm::Id2(512,512), Bounds(0,512,0,512), loop_cnt));
  //std::shared_ptr<ReaderPS<VecType, Size, ReaderXGC<VecType, Size>>> reader(new ReaderXGC<VecType, Size>("XGC_", vtkm::Id2(96,256), Bounds(0,96,0,256)));
  //typedef VectorField<VecType,Size> EvalType;


  int x = 512;
  int y = 256;
  if (argc > 1){
    x = atoi(argv[1]);
    y = atoi(argv[2]);
  }
  std::shared_ptr<Reader<VecType,ReaderCalc<VecType>>> reader(new ReaderCalc<VecType>("XGC_", make_uint2(x,y), make_float2(0,0),make_float2(x,y)));
  typedef DoubleGyreField<VecType> EvalType;



  typedef RK4Integrator<EvalType, VecType> IntegratorType;

  reader->read(vecs);

  auto t0 = std::chrono::high_resolution_clock::now();

  uint2 dim = { 256,256 };

  //vtkm::Id2 dim = reader->dim;
  float2 spacing = reader->spacing;
  //Bounds bounds = reader->bounds;

	std::vector<thrust::host_vector<VecField>> h_l(ttl), h_r(ttl);

  for (int i = 0; i < ttl; i++) {
    for (int y = 0; y<dim.y; y++) {
      for (int x = 0; x<dim.x; x++) {
        h_l[i].push_back(make_float2(x + 0.5, y + 0.5));
        h_r[i].push_back(make_float2(x + 0.5, y + 0.5));
      }
    }
  }
  std::vector<thrust::device_vector<VecField>> d_l(ttl), d_r(ttl);
	for (int i = 0; i<ttl; i++) {
		d_l[i] = h_l[i];
		d_r[i] = h_r[i];
	}

	
	//vecArray = vtkm::cont::make_ArrayHandle(&vecs[0], vecs.size());


	thrust::host_vector<FieldType> h_canvas[ttl], h_propertyField[2], h_omega(dim.x * dim.y, 0), h_tex(dim.x * dim.y, 0);
	VecType t = 0;
	const VecType dt = 0.1;
		for (int i = 0; i < 2; i++) {
		h_propertyField[i].resize(dim.x * dim.x, 0);
	}

	for (int i = 0; i < ttl; i++) {
		h_canvas[i].resize(dim.x * dim.y, 0);
	}
	for (int i = 0; i < h_canvas[0].size(); i++) {
		h_tex[i] = h_canvas[0][i] = rand() % 255;
	}

	thrust::device_vector<FieldType > d_canvas[ttl], d_propField[2], d_omega, d_tex;
	for (int i = 0; i < ttl; i++) {
		d_canvas[i] = h_canvas[i];
	}
	d_propField[0] = h_propertyField[0];
	d_propField[1] = h_propertyField[1];
	d_omega = h_omega;
	d_tex = h_tex;

  //DrawLineWorkletType drawline(bounds, dim);
	//DoSharpen<FieldType, DeviceAdapter> dosharp(dim);
	//DoJitter<FieldType, DeviceAdapter> dojitter(dim);
  //vtkm::cont::ArrayHandleCounting<vtkm::Id> indexArray(vtkm::Id(0), 1, propFieldArray[0].GetNumberOfValues());
	thrust::counting_iterator<uint> indexArray_begin(0), indexArray_end;
	indexArray_end = indexArray_begin + (dim.x * dim.y);

  for (int loop = 0; loop < loop_cnt; loop++) {
	EvalType eval(t, make_float2(0,0), make_float2(dim.x, dim.y), spacing);
    IntegratorType integrator(eval, 3.0);
    //ParticleAdvectionWorkletType advect(integrator);
    //std::cout << "t: " << t << std::endl;

	//vtkm::worklet::DispatcherMapField<ResetParticles<VecType,Size>> resetDispatcher(dim[0]);
    //resetDispatcher.Invoke(indexArray, sl[loop%ttl]);
	thrust::transform(indexArray_begin, indexArray_end, d_l[loop%ttl].begin(), resetParticles(dim));
		//reset the current canvas
		for (int i = 0; i < d_canvas[loop % ttl].size(); i++) {
			d_canvas[loop % ttl][i] = rand() % 255;
		}

    thrust::fill(d_propField[0].begin(), d_propField[0].end(), 0);
	thrust::fill(d_propField[1].begin(), d_propField[1].end(), 0);
	thrust::fill(d_omega.begin(), d_omega.end(), 0);

		for (int i = 0; i < min(ttl, static_cast<size_t>(loop)+1); i++) {
			//advect.Run(sl[i], sr[i], vecArray);
			
			advect<IntegratorType, VecField, VecField> << <dim.x*dim.y/32, 32 >> > (
				thrust::raw_pointer_cast(d_l[i].data()),
				thrust::raw_pointer_cast(vecs.data()),
				integrator,
				thrust::raw_pointer_cast(d_r[i].data())
				);
			//drawline.Run(canvasArray[i], propFieldArray[0], omegaArray, sl[i], sr[i]);
			drawline<<<dim.x*dim.y/32, 32>>>(thrust::raw_pointer_cast(d_canvas[i].data()),
				thrust::raw_pointer_cast(d_omega.data()),
				thrust::raw_pointer_cast(d_l[i].data()),
				thrust::raw_pointer_cast(d_r[i].data()),
				thrust::raw_pointer_cast(d_propField[0].data())
				);


		}

		//sr.swap(sl);
		d_r.swap(d_l);

		//donorm.Run(propFieldArray[0], omegaArray, propFieldArray[1]);
		thrust::transform(d_propField[0].begin(), d_propField[0].end(), d_omega.begin(), d_propField[1].begin(), normale<unsigned char>());
		
		h_propertyField[1] = d_propField[1];
    std::stringstream fn;
    fn << "uflic-" << loop << ".pnm";
    saveAs(fn.str().c_str(), h_propertyField[1], dim.x, dim.y);

    //REUSE omegaArray as a temporary cache to sharpen
    //dosharp.Run(propFieldArray[1], omegaArray);
	dim3 dimBlock(16, 16);
	dim3 dimGrid;
	dimGrid.x = (dim.x + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (dim.y + dimBlock.y - 1) / dimBlock.y;
	sharpen<unsigned char><<<dimGrid, dimBlock>>>(
		thrust::raw_pointer_cast(d_propField[1].data()),
		thrust::raw_pointer_cast(d_omega.data())
		);
    //dojitter.Run(omegaArray, texArray, canvasArray[(loop) % ttl]);

	auto data_tex_begin = thrust::make_zip_iterator(
		make_tuple(indexArray_begin, d_omega.begin(), d_tex.begin()));
	auto data_tex_end = thrust::make_zip_iterator(
		make_tuple(indexArray_end, d_omega.end(), d_tex.end()));
	thrust::transform(
		data_tex_begin,
		data_tex_end,
		d_canvas[(loop) % ttl].begin(),
		Jitter<FieldType>(dim, 256, 256 * 0.1, 256 * 0.9));


    t += dt;// / (vtkm::Float32)ttl + 1.0 / (vtkm::Float32)ttl;
    reader->next(vecs);
    //vecArray = vtkm::cont::make_ArrayHandle(&vecs[0], vecs.size());

	}

  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Finished dt: " << dt << " cnt: " << loop_cnt << " time: " << std::chrono::duration<double>(t1-t0).count() << "s" << std::endl;
    std::stringstream fn;
    fn << "uflic-final" << ".pnm";
	h_propertyField[1] = d_propField[1];
    saveAs(fn.str().c_str(), h_propertyField[1], dim.x, dim.y);


	//vtkm::rendering::Mapper mapper;
	//vtkm::rendering::Canvas canvas(512, 512);
	//vtkm::rendering::Scene scene;

	//scene.AddActor(vtkm::rendering::Actor(
	//	ds.GetCellSet(), ds.GetCoordinateSystem(), ds.GetField(fieldNm), colorTable));
	//vtkm::rendering::Camera camera;
	//SetCamera<ViewType>(camera, ds.GetCoordinateSystem().GetBounds());

	//vtkm::rendering::View2D view;
}
