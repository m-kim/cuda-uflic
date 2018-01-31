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

#ifndef vtk_m_worklet_particleadvection_ParticleAdvectionWorklets_h
#define vtk_m_worklet_particleadvection_ParticleAdvectionWorklets_h

template<class IntegratorType, class VecType, class VelFieldType>
__global__
void advect(const VecType* p1,const VelFieldType *field,
	IntegratorType integrator,
	VecType *p2)

{
	uint tid = blockDim.x * blockIdx.x + threadIdx.x;
	integrator.Step(p1[tid], field, p2[tid]);
}




#endif // vtk_m_worklet_particleadvection_ParticleAdvectionWorklets_h
