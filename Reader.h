#ifndef READER_H
#define READER_H
#include <iostream>
#include <sstream>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/io/reader/VTKDataSetReader.h>
template <typename VecType, class Derived>
class Reader
{
public:
  typedef Derived DerivedType;

  Reader(){}
  Reader(std::string fn,
         uint2 d,
		  const float2& _bb_min,
		  const float2 &_bb_max,
	  float2 sp)
    : filename(fn),
      dim(d),
      bb_min(_bb_min),
	  bb_max(_bb_max),
        spacing(sp)
  {

  }
  virtual void readFile() = 0;

  virtual void next(thrust::device_vector<VecType> &in) =0;
  uint2 dim;
  const float2 spacing;
  //vtkm::cont::DataSetBuilderUniform dataSetBuilder;
  //vtkm::cont::DataSet ds;
  std::stringstream filename;
  float2 bb_min, bb_max;
};

template <typename VecType, class Derived>
class ReaderPS : public Reader<VecType, ReaderPS<VecType, Derived>>
{
public:
  ReaderPS(std::string fn,
           uint2 d,
           const float2& _bb_min,
           const float2 &_bb_max)
    : Reader<VecType, ReaderPS>(fn,
                            d,
                            _bb_min,
                            _bb_max,
                            make_float2(1,1)
                            )
  {

    //this->ds = this->dataSetBuilder.Create(this->dim);
  }

  void readFile()
  {
    std::cout << this->filename.str() << std::endl;

    std::string line;
    std::ifstream file(this->filename.str());

    while (std::getline(file, line)) {
      std::stringstream ss;
      ss << line;
      std::string tok;
      float2 vec;
      //while (std::getline(ss, tok, ' ')) {
      std::getline(ss, tok, ' ');
      vec.x = atof(tok.c_str());
      std::getline(ss, tok, ' ');
      vec.y = atof(tok.c_str());

      buf.push_back(vec);
      //}
    }
    //	String text = null;

    //		String[] subtext = splitTokens(text, " ");

    //		vecs[cnt].x = float(subtext[0]);
    //		vecs[cnt].y = float(subtext[1]);
    //		cnt += 1;
    //	}
    //}
    //catch (IOException e) {
    //	e.printStackTrace();
    //}
  }
  virtual void next(thrust::device_vector<VecType> &in)
  {
    std::swap(in, buf);
  }

  thrust::host_vector<VecType,thrust::cuda::experimental::pinned_allocator<VecType>> buf;
};

template <typename VecType>
class ReaderVTK : public Reader<VecType, ReaderVTK<VecType>>
{
public:
  typedef VectorField<VecType> EvalType;
  typedef thrust::host_vector<VecType,thrust::cuda::experimental::pinned_allocator<VecType>> PinnedType;

  ReaderVTK(std::string fn, int _iter)
    : Reader<VecType, ReaderVTK>(fn,
                            make_uint2(512,512),
                                 make_float2(0,0),
                                 make_float2(512,512)
                                  ,make_float2(1,1)
                            )
    ,iter_cnt(_iter)
  {

  }
  void readFile()
  {
    std::cout << this->filename.str() << std::endl;
    ds = readVTK(this->filename.str());
  //  vtkm::Bounds vounds = ds.GetCoordinateSystem(0).GetBounds();
  //  Bounds bounds(vounds.X, vounds.Y);

    vtkm::cont::CellSetStructured<3> cells;
    ds.GetCellSet(0).CopyTo(cells);
    dim3 = cells.GetSchedulingRange(vtkm::TopologyElementTagPoint());
    this->dim = make_uint2(dim3[0], dim3[2]);
    vtkm::cont::Field fld = ds.GetField(0);
    dah = fld.GetData();
    ah = dah.Cast<vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>>>();
    std::cout << ah.GetNumberOfValues() << std::endl;
    loop = 20;
    buffer.resize(iter_cnt);
    vec_iter = buffer.begin();

    for (int i=0; i<iter_cnt; i++){
      buffer[i].resize(this->dim.x * this->dim.y);
      parse(buffer[i]);
    }
  }

  void next(thrust::device_vector<VecType> &in)
  {
    in = *vec_iter;
    vec_iter++;

  }

  void parse(PinnedType &in)
  {
    for (int z=0; z<this->dim.y; z++){
      for (int y=loop; y<loop+1; y++){
        for (int x=0; x<this->dim.x; x++){
          vtkm::Id idx = z*this->dim.x*128 + y*this->dim.x+x;
          vtkm::Vec<vtkm::Float64,3> vec  = ah.GetPortalConstControl().Get(idx);
          in[z*this->dim.x + x] = make_float2(vec[0],vec[2]);
        }
      }
    }
    loop++;

  }

  vtkm::cont::DataSet readVTK(std::string fn)
  {
    vtkm::cont::DataSet ds;
    vtkm::io::reader::VTKDataSetReader rdr(fn.c_str());
    try
    {
      ds = rdr.ReadDataSet();
    }
    catch (vtkm::io::ErrorIO &e) {
      std::string message("Error reading: ");
      message += fn.c_str();
      message += ", ";
      message += e.GetMessage();
      std::cerr << message << std::endl;
    }
    return ds;

  }


  vtkm::cont::DataSet ds;
  vtkm::cont::DynamicArrayHandle dah;
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64,3>> ah;

  vtkm::Id3 dim3;
  vtkm::Id loop;
  std::vector<PinnedType> buffer;
  typename std::vector<PinnedType>::iterator vec_iter;
  int iter_cnt;
};

template <typename VecType>
class ReaderCalc : public Reader<VecType, ReaderCalc<VecType>>
{
public:
  typedef DoubleGyreField<VecType> EvalType;
  ReaderCalc(std::string fn,
             uint2 d = make_uint2(512,512),
        const float2& _bb_min = make_float2(0,0),
      const float2 &_bb_max = make_float2(512,512),

             float2 sp = make_float2(2,1))
    : Reader<VecType, ReaderCalc>(fn,
                            d,
                            _bb_min,
							_bb_max,
                            sp
                            )
  {
  }
  void readFile()
  {
  }
  void next(thrust::device_vector<VecType> &in){}
};


template <typename VecType>
class ReaderVEL : public ReaderPS<VecType, ReaderVEL<VecType>>
{
public:

  ReaderVEL(std::string &fn,
           uint2 d,
           float2 _bb_min,
            float2 _bb_max,
            size_t frame_cnt)
    : ReaderPS<VecType, ReaderVEL>(fn,
                            d,
                            _bb_min,
                            _bb_max
                            ),
      base_fn(fn),
      frameCnt(frame_cnt)

  {
    //this->ds = this->dataSetBuilder.Create(this->dim);
    loop = 0;
    this->filename << base_fn << loop << ".vel";
    std::cout << this->filename.str() << std::endl;

    mem.resize(100 - loop);
    for (int i=loop; i<min(frameCnt, 100-loop); i++){
      this->filename.str("");
      this->filename << base_fn << i << ".vel";
      this->readFile();
      thrust::swap(this->buf, mem[i]);
      this->buf.resize(0);
    }
  }

  void next(thrust::device_vector<VecType> &in)
  {
    in = mem[loop++];
  }

  std::string base_fn;
  thrust::host_vector<thrust::host_vector<VecType>> mem;
  uint frameCnt, loop;
};
#endif
