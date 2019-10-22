#include <SofaSphFluid/OglFluidModel.inl>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

SOFA_DECL_CLASS(OglFluidModel)

int OglFluidModelClass = sofa::core::RegisterObject("Particle model for OpenGL display, using glsl")
.add< OglFluidModel<sofa::defaulttype::Vec3Types> >();

template class SOFA_SPH_FLUID_API OglFluidModel<sofa::defaulttype::Vec3Types>;

}
}
}
