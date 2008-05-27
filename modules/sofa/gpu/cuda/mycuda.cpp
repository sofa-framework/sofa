#include <sofa/helper/system/config.h>
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

SOFA_LINK_CLASS(CudaFixedConstraint)
SOFA_LINK_CLASS(CudaMechanicalObject)
SOFA_LINK_CLASS(CudaSpringForceField)
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
SOFA_LINK_CLASS(CudaTestForceField)
SOFA_LINK_CLASS(CudaSetTopology)

#ifdef SOFA_DEV

SOFA_LINK_CLASS(CudaMasterContactSolver)

#endif // SOFA_DEV

//MycudaVerboseLevel mycudaVerboseLevel = LOG_ERR;
MycudaVerboseLevel mycudaVerboseLevel = LOG_INFO;
//MycudaVerboseLevel mycudaVerboseLevel = LOG_TRACE;

void mycudaLogError(int err, const char* src)
{
    std::cerr << "CUDA: Error "<<err<<" returned from "<<src<<".\n";
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
