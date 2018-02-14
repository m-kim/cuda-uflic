##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2015 Sandia Corporation.
##  Copyright 2015 UT-Battelle, LLC.
##  Copyright 2015 Los Alamos National Security.
##
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
##  Laboratory (LANL), the U.S. Government retains certain rights in
##  this software.
##
##=============================================================================
cmake_minimum_required(VERSION 3.3)
project(cuda-uflic LANGUAGES CXX CUDA)

string(REGEX REPLACE "include$" "samples" CUDA_SDK_ROOT_DIR ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

message("${CUDA_SDK_ROOT_DIR}")
set(CMAKE_CUDA_STANDARD 11)
set (CMAKE_CXX_STANDARD 11)
set(UFLIC_SRC
  main.cu
)
set(UFLIC_HDR
	Evaluator.cuh
	Integrator.cuh
	Jitter.cuh
  Reader.h
	Sharpen.cuh
)
link_directories(/home/mkim/mark-vtk-m/build/release/lib)
include_directories("${CUDA_SDK_ROOT_DIR}/common/inc" "/home/mkim/mark-vtk-m/build/debug/include" "/home/mkim/mark-vtk-m/")
add_executable(UFLIC_rendering ${UFLIC_SRC} ${UFLIC_HDR} )

target_link_libraries(UFLIC_rendering
vtkm_cont-1.1
 )


