#include "LennardJonesForceField.inl"
#include "Common/Vec3Types.h"
#include "Common/ObjectFactory.h"


namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Core;

namespace Common
{
template<class DataTypes>
void create(LennardJonesForceField<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< LennardJonesForceField<DataTypes>, MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("alpha"))  obj->setAlpha((typename DataTypes::Coord::value_type)atof(arg->getAttribute("alpha")));
        if (arg->getAttribute("beta"))  obj->setBeta((typename DataTypes::Coord::value_type)atof(arg->getAttribute("beta")));
        if (arg->getAttribute("fmax"))  obj->setFMax((typename DataTypes::Coord::value_type)atof(arg->getAttribute("fmax")));
        if (arg->getAttribute("dmax"))  obj->setDMax((typename DataTypes::Coord::value_type)atof(arg->getAttribute("dmax")));
        else if (arg->getAttribute("d0"))  obj->setDMax(2*(typename DataTypes::Coord::value_type)atof(arg->getAttribute("d0")));
        if (arg->getAttribute("d0"))  obj->setD0((typename DataTypes::Coord::value_type)atof(arg->getAttribute("d0")));
        if (arg->getAttribute("p0"))  obj->setP0((typename DataTypes::Coord::value_type)atof(arg->getAttribute("p0")));
        if (arg->getAttribute("damping"))  obj->setP0((typename DataTypes::Coord::value_type)atof(arg->getAttribute("damping")));
    }
}
}

SOFA_DECL_CLASS(LennardJonesForceField)

// Each instance of our class must be compiled
template class LennardJonesForceField<Vec3fTypes>;
template class LennardJonesForceField<Vec3dTypes>;

// And registered in the Factory
Creator< ObjectFactory, LennardJonesForceField<Vec3fTypes> > LennardJonesForceField3fClass("LennardJonesForceField", true);
Creator< ObjectFactory, LennardJonesForceField<Vec3dTypes> > LennardJonesForceField3dClass("LennardJonesForceField", true);

} // namespace Sofa

} // namespace Components
