#ifndef Evaluator_h
#define Evaluator_h
#include <math.h>
#include <helper_math.h>
template <typename VecType>
class VectorField
{
public:
  VectorField()
  {
  }

  
  VectorField(const float t,
              const float2& bb_min,
              const float2 &bb_max,

              const float2 s)
    : spacing(s),
      t(0.0)
  {
    dim.x = bb_max.x - bb_min.x;
    dim.y = bb_max.y - bb_min.y;
  }

  void incrT(VecType dt){
    t += dt;
  }
  template<typename VelFieldType>
  __host__ __device__
  bool Evaluate(const float2& pos,
                const VelFieldType* vecData,
                float2 & outVel) const
  {
    //if (!bounds.Contains(pos)) Contains is inclusive
    if (pos.x < 0 || pos.x >= dim.x || pos.y < 0 || pos.y >= dim.y || pos.x != pos.x || pos.y != pos.y)
      return false;

    outVel = vecData[static_cast<int>(floor(pos.y)) * dim.x + static_cast<int>(floor(pos.x))];
    return dot(outVel, outVel) > 0;
  }


private:

   uint2 dim;
  float2 spacing;
  VecType omega, A, epsilon;
  VecType t;

};
template <typename FieldType>
class DoubleGyreField
{
public:
  
  DoubleGyreField()
    : omega(2 * CUDART_PI_F / 10.0),
      A(0.1),
      epsilon(1e-6),
      t(0.0)
  {
  }

  
  DoubleGyreField(const float t,
                  const float2& bb_min,
				  const float2 &bb_max,
                  const float2 s)
    : spacing(s),
      omega(2 * CUDART_PI_F / 10.0),
      A(0.1),
      epsilon(1e-6),
      t(0.0)
  {
		dim.x = bb_max.x - bb_min.x;
		dim.y = bb_max.y - bb_min.y;
  }

  
  void incrT(FieldType dt){
    t += dt;
  }

	template<typename VelFieldType>
  __host__ __device__
  bool Evaluate(const float2& pos,
                const VelFieldType* vecData,
                float2& outVel) const
  {
    //if (!bounds.Contains(pos))
    //  return false;
    outVel.x = calcU(spacing.x * pos.x / dim.x, spacing.y * pos.y / dim.y, t);
    outVel.y = calcV(spacing.x * pos.x / dim.x, spacing.y * pos.y / dim.y, t);
		//vtkm::Float32 norm = outVel[0] * outVel[0] + outVel[1] * outVel[1];
		//norm = sqrt(norm);
		//outVel[0] /= norm;
		//outVel[1] /= norm;
    return true;
  }


private:
	__host__ __device__
  FieldType a(FieldType t) const
  {
   return epsilon * sin(omega * t);
  }
  __host__ __device__
  FieldType b(FieldType t) const
  {
   return 1 - 2 * epsilon * sin(omega * t);
  }
  
  __host__ __device__
  FieldType f(FieldType x, FieldType t) const
  {
    return a(t) * x*x + b(t) * x;
  }
  
  __host__ __device__
  FieldType calcU(FieldType x, FieldType y, FieldType t) const
  {
       return -CUDART_PI_F * A * sin(CUDART_PI_F*f(x,t)) * cos(CUDART_PI_F*y);
  }

  __host__ __device__
  FieldType calcV(FieldType x, FieldType y, FieldType t) const
  {
    return CUDART_PI_F * A * cos(CUDART_PI_F*f(x,t)) * sin(CUDART_PI_F*y) * (2 * a(t) * x + b(t));
  }

	uint2 dim;
  float2 spacing;
  FieldType omega, A, epsilon;
  FieldType t;

};

#endif
