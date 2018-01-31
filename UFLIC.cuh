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

#include <vtkm/Types.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/CellSetExplicit.h>
#include <vtkm/cont/CellSetStructured.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/Field.h>
#include <vtkm/exec/ExecutionObjectBase.h>

#include <vtkm/worklet/DispatcherMapField.h>



template <typename IntegratorType, typename FieldType, vtkm::IdComponent Size, typename DeviceAdapterTag>
class ParticleAdvectWorklet : public vtkm::worklet::WorkletMapField
{
public:
  typedef vtkm::Vec<vtkm::Float32, Size> VecType;

  typedef void ControlSignature(FieldIn<>, FieldOut<>, WholeArrayInOut<>);
  typedef void ExecutionSignature(_1, _2, _3);
  typedef _1 InputDomain;

  template<typename VelFieldType>
  VTKM_EXEC void operator()(const VecType& p1, VecType& p2, const VelFieldType &field) const
  {

    integrator.Step(p1, field, p2);
  }

  ParticleAdvectWorklet(const IntegratorType& it)
    : integrator(it)
  {
  }

  IntegratorType integrator;
};


template <typename IntegratorType, typename FieldType, vtkm::IdComponent Size, typename DeviceAdapterTag>
class ParticleAdvectionWorklet
{
public:

  typedef ParticleAdvectWorklet<IntegratorType,
                                FieldType,
                                Size,
                                DeviceAdapterTag>
    ParticleAdvectWorkletType;

  ParticleAdvectionWorklet(const IntegratorType& it)
    : integrator(it)
  {

  }

  template <typename PointStorage, typename FieldStorage>
  vtkm::cont::ArrayHandle<vtkm::Vec<FieldType, Size>, PointStorage> Run(
    const vtkm::cont::ArrayHandle<vtkm::Vec<FieldType, Size>, PointStorage>& _pl,
      const vtkm::cont::ArrayHandle<vtkm::Vec<FieldType, Size>, PointStorage>& _pr,
    const vtkm::cont::ArrayHandle<vtkm::Vec<FieldType, Size>, FieldStorage> fieldArray)
  {
    pl = _pl;
    pr = _pr;
    field = fieldArray;
    return run();
  }

  ~ParticleAdvectionWorklet() {}

private:
  vtkm::cont::ArrayHandle<vtkm::Vec<FieldType, 2>> run(bool dumpOutput = false)
  {
    typedef typename vtkm::worklet::DispatcherMapField<ParticleAdvectWorkletType>
      ParticleWorkletDispatchType;


    ParticleAdvectWorkletType particleWorklet(integrator);
    ParticleWorkletDispatchType particleWorkletDispatch(particleWorklet);
    particleWorkletDispatch.Invoke(pl, pr, field);


    return pr;
  }

  IntegratorType integrator;
  vtkm::cont::ArrayHandle<vtkm::Vec<FieldType, Size>> pl, pr;
  vtkm::cont::DataSet ds;
  vtkm::Id maxSteps;
  vtkm::Id ParticlesPerRound;
  vtkm::cont::ArrayHandle<vtkm::Vec<FieldType, Size>> field;

};




#endif // vtk_m_worklet_particleadvection_ParticleAdvectionWorklets_h
