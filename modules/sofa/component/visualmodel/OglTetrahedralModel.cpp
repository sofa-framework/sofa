#include "OglTetrahedralModel.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

SOFA_DECL_CLASS(OglTetrahedralModel)

int OglTetrahedralModelClass = sofa::core::RegisterObject("Tetrahedral model for OpenGL display")
#ifndef SOFA_FLOAT
        .add< OglTetrahedralModel<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< OglTetrahedralModel<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class OglTetrahedralModel<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class OglTetrahedralModel<Vec3fTypes>;
#endif

}
}
}
