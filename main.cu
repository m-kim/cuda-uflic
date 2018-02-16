#include <fstream>
#include <sstream>

#include <helper_cuda.h>
#include <helper_math.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
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

bool do_print = false;
template<class T>
struct normale {
	__host__ __device__ T operator()(const T &x, const T &y) const {
		T reval = 0;
    if (y > 0)
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
    thrust::uniform_int_distribution<unsigned char> dist(a, b);
		rng.discard(n);

		return dist(rng);
	}
};

template<typename VecComponentType>
void saveAs(std::string fileName, 
  thrust::device_vector<VecComponentType> d_canvas,
	size_t Width, size_t Height) {

  thrust::host_vector<VecComponentType> canvas = d_canvas;
	std::ofstream of(fileName.c_str(), std::ios_base::binary | std::ios_base::out);
	of << "P6\n" << Width << " " << Height << "\n" << 255 << "\n";
	//ColorBufferType::PortalConstControl colorPortal = this->ColorBuffer.GetPortalConstControl();
  for (int yIndex = (Height - 1); yIndex >= 0; yIndex--)
  {
    for (int xIndex = 0; xIndex < Width; xIndex++)
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




template<typename EvalType, typename VecType, typename FieldType, typename VecField>
void run( std::shared_ptr<Reader<VecField>> reader)
{
  const size_t ttl = 10;


  std::vector<thrust::host_vector<VecField, thrust::cuda::experimental::pinned_allocator<VecField>>> vecs(2);
  std::vector<thrust::device_vector<VecField>> d_vecs(2);
#ifdef RUN_BFIELD
  std::shared_ptr<Reader<VecField, ReaderVTK<VecField>>> reader(new ReaderVTK<VecField>("/home/mkim/vtkm-uflic/BField_2d.vtk", 34));
  typedef VectorField<VecType> EvalType;
  loop_cnt = 34;
#elif defined RUN_PSI2Q
  //std::shared_ptr<Reader<VecField,  ReaderPS<VecField,ReaderXGC<VecType,Size>>>> reader(new ReaderPS<VecField, ReaderXGC<VecField>>("/home/mkim/vtkm-uflic/psi2q/2D_packed/psi2D_packed_normalized_256_99.vec", uint2(256,256), float2(0,0), float2(256,256)));
  loop_cnt = 99;
  std::string fstr("/home/mkim/vtkm-uflic/psi2q/2D_packed/psi2D_packed_512_");
  std::shared_ptr<ReaderPS<VecField, ReaderVEL<VecField>>> reader(new ReaderVEL<VecField>(fstr, make_uint2(512,512), make_float2(0.0f,0.0f), make_float2(512.0f, 512.0f), loop_cnt));
  typedef VectorField<VecType> EvalType;

#elif defined RUN_XGC
  std::string fstr("/home/mkim/vtkm-uflic/xgc/XGC_");
  std::shared_ptr<ReaderPS<VecField, ReaderVEL<VecField>>> reader(new ReaderVEL<VecField>(fstr, make_uint2(96,256), make_float2(0,0), make_float2(96,256), 149));
  typedef VectorField<VecType> EvalType;
#else

#endif


  size_t loop_cnt = reader->iter_cnt;
  typedef RK4Integrator<EvalType, VecType> IntegratorType;

  reader->readFile();

  cudaFree(0);
  auto t0 = std::chrono::high_resolution_clock::now();


  uint2 dim = reader->dim;
  float2 spacing = reader->spacing;
  //Bounds bounds = reader->bounds;
  thrust::counting_iterator<uint> indexArray_begin(0), indexArray_end;
  indexArray_end = indexArray_begin + (dim.x * dim.y);


  std::vector<thrust::device_vector<VecField>> d_l(ttl), d_r(ttl);
	for (int i = 0; i<ttl; i++) {
//		d_l[i] = h_l[i];
//		d_r[i] = h_r[i];
    d_l[i].resize(dim.x*dim.y);
    d_r[i].resize(dim.x*dim.y);
    thrust::transform(indexArray_begin, indexArray_end, d_l[i].begin(), resetParticles(dim));
    thrust::transform(indexArray_begin, indexArray_end, d_r[i].begin(), resetParticles(dim));

	}

	
	//vecArray = vtkm::cont::make_ArrayHandle(&vecs[0], vecs.size());


  thrust::device_vector<FieldType > d_canvas[ttl], d_propField[2], d_omega(dim.x*dim.y,0), d_tex;
  VecType t = 0;
	const VecType dt = 0.1;
  for (int i = 0; i < 2; i++) {
    d_propField[i].resize(dim.x * dim.x, 0);
	}

  for (int i = 0; i < ttl; i++) {
    d_canvas[i].resize(dim.x * dim.y, 0);
  }
  d_tex.resize(dim.x*dim.y);
  thrust::transform(indexArray_begin, indexArray_end, d_tex.begin(), prg(0,255));
  d_canvas[0] = d_tex;

  for (int loop = 0; loop < loop_cnt; loop++) {


    reader->next(d_vecs[0]);

    EvalType eval(t, make_float2(0,0), make_float2(dim.x, dim.y), spacing);
    IntegratorType integrator(eval, 3.0);
    //std::cout << "t: " << t << std::endl;

    thrust::transform(indexArray_begin, indexArray_end, d_l[loop%ttl].begin(), resetParticles(dim));
		//reset the current canvas
    thrust::transform(indexArray_begin, indexArray_end, d_canvas[loop%ttl].begin(), prg(0,255));

    thrust::fill(d_propField[0].begin(), d_propField[0].end(), 0);
    thrust::fill(d_propField[1].begin(), d_propField[1].end(), 0);
    thrust::fill(d_omega.begin(), d_omega.end(), 0);

		for (int i = 0; i < min(ttl, static_cast<size_t>(loop)+1); i++) {
			//advect.Run(sl[i], sr[i], vecArray);
			
      advect<IntegratorType, VecField, VecField> << <dim.x*dim.y/128, 128 >> > (
				thrust::raw_pointer_cast(d_l[i].data()),
        thrust::raw_pointer_cast(d_vecs[0].data()),
				integrator,
				thrust::raw_pointer_cast(d_r[i].data())
				);
			//drawline.Run(canvasArray[i], propFieldArray[0], omegaArray, sl[i], sr[i]);
      dim3 dimBlock(16,8);
			dim3 dimGrid;
			dimGrid.x = (dim.x + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (dim.y + dimBlock.y - 1) / dimBlock.y;


      drawline<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(d_canvas[i].data()),
				thrust::raw_pointer_cast(d_omega.data()),
				thrust::raw_pointer_cast(d_l[i].data()),
				thrust::raw_pointer_cast(d_r[i].data()),
				thrust::raw_pointer_cast(d_propField[0].data())
				);


		}

		//sr.swap(sl);
		d_r.swap(d_l);

		thrust::transform(d_propField[0].begin(), d_propField[0].end(), d_omega.begin(), d_propField[1].begin(), normale<FieldType>());

    if (do_print){
      std::stringstream fn;
      fn << "uflic-" << loop << ".pnm";
      saveAs(fn.str().c_str(), d_propField[1], dim.x, dim.y);

    }

    //REUSE omegaArray as a temporary cache to sharpen
    //dosharp.Run(propFieldArray[1], omegaArray);
  dim3 dimBlock(16,8);
	dim3 dimGrid;
  dimGrid.x = (dim.x + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (dim.y + dimBlock.y - 1) / dimBlock.y;
  sharpen<FieldType, FieldType><<<dimGrid, dimBlock>>>(
		thrust::raw_pointer_cast(d_propField[1].data()),
		thrust::raw_pointer_cast(d_omega.data())
		);

	thrust::counting_iterator<uint> _begin(0), _end;
	_end = _begin + (dim.x * dim.y);

  auto data_tex_begin = thrust::make_zip_iterator(
    make_tuple(_begin, d_omega.begin(), d_tex.begin()));
  thrust::transform(
    data_tex_begin,
    data_tex_begin + d_omega.size(),
    d_canvas[(loop) % ttl].begin(),
    Jitter<FieldType>(dim, 255, 255 * 0.1, 255 * 0.9));

    t += dt;// / (vtkm::Float32)ttl + 1.0 / (vtkm::Float32)ttl;

    cudaStreamSynchronize(0);

  }
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << "Finished dt: " << dt << " cnt: " << loop_cnt << " time: " << std::chrono::duration<double>(t1-t0).count() << " s" << std::endl;
    std::stringstream fn;
    fn << "uflic-final" << ".pnm";
    saveAs(fn.str().c_str(), d_propField[1], dim.x, dim.y);
}

std::tuple<int,int,int>
parse(int argc, char **argv){

  int x = 512;
  int y = 256;
  int which = 0;

  for (int i=1; i<argc; i++){
    if (!strcmp(argv[i], "bfield")){
      which = 1;
    }
    else if (!strcmp(argv[i], "PSI")){
      which = 2;
    }
    else if (!strcmp(argv[i], "dims")){
      if (i+1 < argc && i+2 < argc){
        x = atoi(argv[i+1]);
        y = atoi(argv[i+2]);
        i += 2;
      }
    }
    else if (!strcmp(argv[i], "print")){
      do_print = true;
    }
  }

  return std::make_tuple(which,x,y);
}

int main(int argc, char **argv)
{
  typedef VTKM_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;
  typedef float VecType;
  typedef int FieldType;
  typedef float2 VecField;

  std::tuple<int, int, int> ret;

  ret = parse(argc,argv);

  std::shared_ptr<Reader<VecField>> reader;

  if (std::get<0>(ret) == 1){
    std::string fstr("/home/mkim/vtkm-uflic/BField_2d.vtk");
    reader = std::shared_ptr<ReaderVTK<VecField>>(new ReaderVTK<VecField>(fstr, 34));
    run<VectorField<VecType>,VecType,FieldType,VecField>(reader);
  }

  else if (std::get<0>(ret) == 2){
    //std::shared_ptr<Reader<VecType,  ReaderPS<VecType,ReaderVEL<VecType>>>> reader(new ReaderPS<VecType, ReaderVEL<VecType>>("/home/mkim/vtkm-uflic/psi2q/2D_packed/psi2D_packed_normalized_256_99.vec", vtkm::Id2(256,256), Bounds(0,256,0,256)));
    std::string fstr("/home/mkim/vtkm-uflic/psi2q/2D_packed/psi2D_packed_512_");
    reader = std::shared_ptr<ReaderVEL<VecField>>(new ReaderVEL<VecField>(fstr, make_uint2(512,512), make_float2(0,0), make_float2(512,512), 99));
    run<VectorField<VecType>,VecType,FieldType,VecField>(reader);
  }
  else{

    int x = std::get<1>(ret);
    int y = std::get<2>(ret);
    std::string fstr("XGC_");
    reader = std::shared_ptr<ReaderCalc<VecField>>(new ReaderCalc<VecField>(fstr, make_uint2(x,y), make_float2(0,0), make_float2(x,y), make_float2(2,1), 99));
    run<DoubleGyreField<VecType>,VecType,FieldType,VecField>(reader);
  }
}
