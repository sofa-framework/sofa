#define OglFluidModel_CPP_

#include "OglFluidModel.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{



SOFA_DECL_CLASS(OglFluidModel)

int OglFluidModelClass = sofa::core::RegisterObject("Particle model for OpenGL display - NG")
#ifndef SOFA_FLOAT
        .add< OglFluidModel<sofa::defaulttype::Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< OglFluidModel<sofa::defaulttype::Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class OglFluidModel<sofa::defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class OglFluidModel<sofa::defaulttype::Vec3fTypes>;
#endif

}
}
}
