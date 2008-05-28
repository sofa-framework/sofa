#include "CudaTetrahedralVisualModel.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

SOFA_DECL_CLASS(CudaOglTetrahedralModel)

int CudaOglTetrahedralModelClass = sofa::core::RegisterObject("Tetrahedral model for OpenGL display")
        .add< OglTetrahedralModel<sofa::gpu::cuda::CudaVec3fTypes> >()
        ;

template class OglTetrahedralModel<sofa::gpu::cuda::CudaVec3fTypes>;

}
}
}
