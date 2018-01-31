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

find_package(CUDA QUIET REQUIRED)

set(UFLIC_SRC
  main.cu
)
set(UFLIC_HDR
	Evaluator.cuh
	Integrator.cuh
	Jitter.cuh
	Sharpen.cuh
)

include_directories("${CUDA_SDK_ROOT_DIR}/common/inc")
add_executable(UFLIC_rendering ${UFLIC_SRC} ${UFLIC_HDR} )

target_link_libraries(UFLIC_rendering )


