#include <sofa/component/forcefield/SPHFluidForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace core::componentmodel::behavior;

// Each instance of our class must be compiled
template class SPHFluidForceField<Vec3fTypes>;
template class SPHFluidForceField<Vec3dTypes>;

SOFA_DECL_CLASS(SPHFluidForceField)

// And registered in the Factory
int SPHFluidForceFieldClass = core::RegisterObject("Smooth Particle Hydrodynamics")
        .add< SPHFluidForceField<Vec3dTypes> >()
        .add< SPHFluidForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

