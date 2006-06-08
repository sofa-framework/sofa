#include "PenalityContactForceField.inl"
#include "Common/Vec3Types.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(PenalityContactForceField)

using namespace Common;

template class PenalityContactForceField<Vec3dTypes>;
template class PenalityContactForceField<Vec3fTypes>;

template<class DataTypes>
void create(PenalityContactForceField<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< PenalityContactForceField<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        XML::createWith2Objects< PenalityContactForceField<DataTypes>, Core::MechanicalModel<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
}

Creator< ObjectFactory, PenalityContactForceField<Vec3dTypes> > PenalityContactForceFieldVec3dClass("PenalityContactForceField", true);
Creator< ObjectFactory, PenalityContactForceField<Vec3fTypes> > PenalityContactForceFieldVec3fClass("PenalityContactForceField", true);

} // namespace Components

} // namespace Sofa
