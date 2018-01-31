#ifndef Evaluator_h
#define Evaluator_h
#include <vtkm/Types.h>
#include <vtkm/VectorAnalysis.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/CellSetStructured.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/cont/DynamicArrayHandle.h>
#include "Bounds2.h"

template <typename FieldType, vtkm::IdComponent Size = 2>
class VectorField
{
public:
  VTKM_CONT
  VectorField()
  {
  }

  VTKM_CONT
  VectorField(const vtkm::Float32 t,
              const Bounds& bb,
              const vtkm::Vec<FieldType, Size> s)
    : bounds( bb ),
      spacing(s),
      t(0.0)
  {
    dim[0] = bounds.X.Max - bounds.X.Min;
    dim[1] = bounds.Y.Max - bounds.Y.Min;
  }

  VTKM_EXEC
  void incrT(FieldType dt){
    t += dt;
  }
  template<typename VelFieldType>
  VTKM_EXEC
  bool Evaluate(const vtkm::Vec<FieldType, Size>& pos,
                const VelFieldType& vecData,
                vtkm::Vec<FieldType, Size>& outVel) const
  {
    //if (!bounds.Contains(pos)) Contains is inclusive
    if (pos[0] < 0 || pos[0] >= dim[0] || pos[1] < 0 || pos[1] >= dim[1] || pos[0] != pos[0] || pos[1] != pos[1])
      return false;

    outVel = vecData.Get(vtkm::Floor(pos[1]) * dim[0] + vtkm::Floor(pos[0]));
    return outVel.Dot(outVel) > 0;
  }


private:

  vtkm::Id2 dim;
  vtkm::Vec<FieldType,Size> spacing;
  Bounds bounds;
  FieldType omega, A, epsilon;
  FieldType t;

};
template <typename FieldType, vtkm::IdComponent Size = 2>
class DoubleGyreField
{
public:
  VTKM_CONT
  DoubleGyreField()
    : omega(2 * vtkm::Pi() / 10.0),
      A(0.1),
      epsilon(1e-6),
      t(0.0)
  {
  }

  VTKM_CONT
  DoubleGyreField(const vtkm::Float32 t,
                  const Bounds& bb,
                  const vtkm::Vec<FieldType, Size> s)
    : bounds( bb ),
      spacing(s),
      omega(2 * vtkm::Pi() / 10.0),
      A(0.1),
      epsilon(1e-6),
      t(0.0)
  {
		dim[0] = bounds.X.Max - bounds.X.Min;
		dim[1] = bounds.Y.Max - bounds.Y.Min;
  }

  VTKM_EXEC
  void incrT(FieldType dt){
    t += dt;
  }

	template<typename VelFieldType>
  VTKM_EXEC_CONT
  bool Evaluate(const vtkm::Vec<FieldType, Size>& pos,
                const VelFieldType& vtkmNotUsed(vecData),
                vtkm::Vec<FieldType, Size>& outVel) const
  {
    //if (!bounds.Contains(pos))
    //  return false;
    outVel[0] = calcU(spacing[0] * pos[0] / dim[0], spacing[1] * pos[1] / dim[1], t);
    outVel[1] = calcV(spacing[0] * pos[0] / dim[0], spacing[1] * pos[1] / dim[1], t);
		//vtkm::Float32 norm = outVel[0] * outVel[0] + outVel[1] * outVel[1];
		//norm = sqrt(norm);
		//outVel[0] /= norm;
		//outVel[1] /= norm;
    return true;
  }


private:
  VTKM_EXEC
  FieldType a(FieldType t) const
  {
   return epsilon * vtkm::Sin(omega * t);
  }
  VTKM_EXEC
  FieldType b(FieldType t) const
  {
   return 1 - 2 * epsilon * vtkm::Sin(omega * t);
  }
  VTKM_EXEC
  FieldType f(FieldType x, FieldType t) const
  {
    return a(t) * x*x + b(t) * x;
  }
  VTKM_EXEC
  FieldType calcU(FieldType x, FieldType y, FieldType t) const
  {
       return -vtkm::Pi() * A * vtkm::Sin(vtkm::Pi()*f(x,t)) * vtkm::Cos(vtkm::Pi()*y);
  }

  VTKM_EXEC
  FieldType calcV(FieldType x, FieldType y, FieldType t) const
  {
    return vtkm::Pi() * A * vtkm::Cos(vtkm::Pi()*f(x,t)) * vtkm::Sin(vtkm::Pi()*y) * (2 * a(t) * x + b(t));
  }

	vtkm::Id2 dim;
  vtkm::Vec<FieldType,Size> spacing;
  Bounds bounds;
  FieldType omega, A, epsilon;
  FieldType t;

};

#endif
