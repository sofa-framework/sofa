#include "HexaRemover.inl"
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS ( HexaRemover );

int HexaRemoverClass = core::RegisterObject ( "Hexahedra removing using volumetric collision detection." )
#ifdef SOFA_FLOAT
        .add< HexaRemover<defaulttype::Vec3fTypes> >(true)
#else
        .add< HexaRemover<defaulttype::Vec3dTypes> >(true)
#ifndef SOFA_DOUBLE
        .add< HexaRemover<defaulttype::Vec3fTypes> >()
#endif
#endif
//                                                                  .add< HexaRemover<gpu::cuda::CudaVec3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
//                                                                  .add< HexaRemover<gpu::cuda::CudaVec3dTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class HexaRemover<defaulttype::Vec3dTypes> ;
#endif
#ifndef SOFA_DOUBLE
template class HexaRemover<defaulttype::Vec3fTypes> ;
#endif
//                                                                  template class HexaRemover<gpu::cuda::CudaVec3fTypes> ;
#ifdef SOFA_GPU_CUDA_DOUBLE
//                                                                  template class HexaRemover<gpu::cuda::CudaVec3dTypes> ;
#endif

}

}

}
