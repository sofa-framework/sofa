#include <sofa/component/forcefield/PenalityContactForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class PenalityContactForceField<Vec3dTypes>;
template class PenalityContactForceField<Vec3fTypes>;


SOFA_DECL_CLASS(PenalityContactForceField)

// Register in the Factory
int PenalityContactForceFieldClass = core::RegisterObject("Contact using repulsive springs")
        .add< PenalityContactForceField<Vec3dTypes> >()
        .add< PenalityContactForceField<Vec3fTypes> >()
        ;
} // namespace forcefield

} // namespace component

} // namespace sofa

