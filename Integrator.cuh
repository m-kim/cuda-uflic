//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 Sandia Corporation.
//  Copyright 2014 UT-Battelle, LLC.
//  Copyright 2014 Los Alamos National Security.
//
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================

#ifndef Integrator_h
#define Integrator_h

#include <vtkm/Types.h>


template <typename FieldEvaluateType, typename FieldType, vtkm::IdComponent Size>
class RK4Integrator
{
public:
  VTKM_EXEC_CONT
  RK4Integrator()
    : h(0)
    , h_2(0)
  {
  }

  VTKM_EXEC_CONT
  RK4Integrator(const FieldEvaluateType& _eval, FieldType _h)
    : eval(_eval)
    , h(_h)
    , h_2(_h / 2.f)
  {
  }

  template <typename VelFieldType>
  VTKM_EXEC bool Step(const vtkm::Vec<FieldType, Size>& pos,
                      const VelFieldType& pf,
                      vtkm::Vec<FieldType, Size>& out) const
  {
		vtkm::Vec<FieldType, Size> k1(0,0), k2(0,0), k3(0,0), k4(0,0);

#if 1
    if (!eval.Evaluate(pos, pf, k1)) {
      out = pos;
      return false;
		}
		vtkm::Normalize(k1);

		k1 = k1 * h * 0.5;
		k1 = k1 + pos;
		
    if (!eval.Evaluate(k1, pf, k2)) {
			out = pos;
			return false;
		}
		vtkm::Normalize(k2);
		k1 = k1 - pos;

		//v = (getVector(k1));
		//if (v.length() > 1e-6)
		//	v.normalize();
		//k1.sub(pt);

		//k2 = new Particle(v);
		//k2.mult(h*0.5);
		//k2.add(pt);
		//v = (getVector(k2));
		//if (v.length() > 1e-6)
		//	v.normalize();
		////v.Print();
		//k2.sub(pt);

		k2 = k2 * h * 0.5;
		k2 = k2 + pos;
    if (!eval.Evaluate(k2, pf, k3)) {
			out = pos;
			return false;
		}
		
		vtkm::Normalize(k3);
		k2 = k2 - pos;

		//k3 = new Particle(v);
		//k3.mult(h);
		//k3.add(pt);
		//v = (getVector(k3));
		//if (v.length() > 1e-6)
		//	v.normalize();
		//k3.sub(pt);
		k3 = k3 * h;
		k3 = k3 + pos;
    if (!eval.Evaluate(k3, pf, k4)) {
			out = pos;
			return false;
		}

		vtkm::Normalize(k4);
		k3 = k3 - pos;
		////v.Print();

		//k4 = new Particle(v);
		//k4.mult(h);
		k4 = k4 * h;

		//k1.mult(1.0 / 6.0);
		//k2.mult(2.0 / 3.0);
		//k3.mult(1.0 / 3.0);
		//k4.mult(4.0 / 6.0);
		k1 = k1 / 6.0;
		k2 = 2.0 * k2 / 3.0;
		k3 = k3 / 3.0;
		k4 = 4.0 * k4 / 6.0;

		out = pos + k1 + k2 + k3 + k4;

		return true;
		//pt.add(k1);
		//pt.add(k2);
		//pt.add(k3);
		//pt.add(k4);
#else		
    if (eval.Evaluate(pos, pf, k1) && eval.Evaluate(pos + h_2 * k1, pf, k2) &&
        eval.Evaluate(pos + h_2 * k2, pf, k3) && eval.Evaluate(pos + h * k3, pf, k4))
    {
      out = pos + h / 6.0f * (k1 + 2 * k2 + 2 * k3 + k4);
      return true;
    }
    return false;
#endif
  }

  FieldEvaluateType eval;
  FieldType h, h_2;
};

template <typename FieldEvaluateType, typename FieldType, vtkm::IdComponent Size>
class EulerIntegrator
{
public:
  VTKM_EXEC_CONT
  EulerIntegrator()
    : h(0)
  {
  }
  VTKM_EXEC_CONT
  EulerIntegrator(const FieldEvaluateType& _eval, FieldType _h)
    : eval(_eval)
    , h(_h)
  {
  }

  template <typename PortalType>
  VTKM_EXEC bool Step(const vtkm::Vec<FieldType, Size>& pos,
                      const PortalType& field,
                      vtkm::Vec<FieldType, Size>& out) const
  {
    vtkm::Vec<FieldType, Size> vCur;
//    eval.incrT(h);
    if (eval.Evaluate(pos, field, vCur))
    {
      out = pos + vCur;
      return true;
    }
    return false;
  }

  FieldEvaluateType eval;
  FieldType h;
};

#endif // vtk_m_worklet_particleadvection_Integrators_h
