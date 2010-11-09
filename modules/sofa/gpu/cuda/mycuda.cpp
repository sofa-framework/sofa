/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/system/config.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/BackTrace.h>
#include <iostream>
#include <fstream>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "mycuda.h"

namespace sofa
{
namespace gpu
{
namespace cuda
{

SOFA_LINK_CLASS(CudaBoxROI)
SOFA_LINK_CLASS(CudaFixedConstraint)
SOFA_LINK_CLASS(CudaMechanicalObject)
//SOFA_LINK_CLASS(CudaSpringForceField)
SOFA_LINK_CLASS(CudaUniformMass)
SOFA_LINK_CLASS(CudaPlaneForceField)
SOFA_LINK_CLASS(CudaSphereForceField)
SOFA_LINK_CLASS(CudaEllipsoidForceField)
SOFA_LINK_CLASS(CudaIdentityMapping)
SOFA_LINK_CLASS(CudaBarycentricMapping)
SOFA_LINK_CLASS(CudaRigidMapping)
SOFA_LINK_CLASS(CudaSubsetMapping)
SOFA_LINK_CLASS(CudaDistanceGridCollisionModel)
SOFA_LINK_CLASS(CudaTetrahedronFEMForceField)
SOFA_LINK_CLASS(CudaCollision)
SOFA_LINK_CLASS(CudaCollisionDetection)
SOFA_LINK_CLASS(CudaPointModel)
SOFA_LINK_CLASS(CudaSphereModel)
SOFA_LINK_CLASS(CudaSetTopology)
SOFA_LINK_CLASS(CudaVisualModel)
SOFA_LINK_CLASS(CudaOglTetrahedralModel)

#ifdef SOFA_DEV

SOFA_LINK_CLASS(CudaTetrahedronTLEDForceField)
SOFA_LINK_CLASS(CudaHexahedronTLEDForceField)
SOFA_LINK_CLASS(CudaTetrahedronSuperTLEDForceField)
//SOFA_LINK_CLASS(CudaMasterContactSolver)
SOFA_LINK_CLASS(CudaSpatialGridContainer)

#endif // SOFA_DEV

extern "C"
{
//MycudaVerboseLevel mycudaVerboseLevel = LOG_ERR;
    MycudaVerboseLevel mycudaVerboseLevel = LOG_INFO;
//MycudaVerboseLevel mycudaVerboseLevel = LOG_TRACE;

    int mycudaMultiOpMax = 0;
}

static void timerSyncCB(void*)
{
    mycudaThreadSynchronize();
}

void mycudaPrivateInit(int /*device*/)
{
    const char* sync = getenv("CUDA_TIMER_SYNC");
    if (!sync || !*sync || atoi(sync))
        sofa::helper::AdvancedTimer::setSyncCallBack(timerSyncCB, NULL);
    const char* verbose = getenv("CUDA_VERBOSE");
    if (verbose && *verbose)
        mycudaVerboseLevel = (MycudaVerboseLevel) atoi(verbose);
    const char* multiop = getenv("CUDA_MULTIOP");
    if (multiop && *multiop)
        mycudaMultiOpMax = atoi(multiop);
}

void mycudaLogError(const char* err, const char* src)
{
    std::cerr << "CUDA error: "<< err <<" returned from "<< src <<".\n";
    sofa::helper::BackTrace::dump();
    exit(1);
}

int myprintf(const char* fmt, ...)
{
    va_list args;
    va_start( args, fmt );
    int r = vfprintf( stderr, fmt, args );
    va_end( args );
    return r;
}

const char* mygetenv(const char* name)
{
    return getenv(name);
}

} // namespace cuda
} // namespace gpu
} // namespace sofa
