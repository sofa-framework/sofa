#include "LennardJonesForceField.inl"
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/XML/ForceFieldNode.h"

using namespace Sofa::Components;
using namespace Sofa::Components::Common;
using namespace Sofa::Core;

namespace Sofa
{
namespace Components
{
namespace Common
{

template<class DataTypes>
void create(LennardJonesForceField<DataTypes>*& obj, XML::Node<Sofa::Core::ForceField>* arg)
{
    XML::createWithParent< LennardJonesForceField<DataTypes>, MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("alpha"))  obj->setAlpha(atof(arg->getAttribute("alpha")));
        if (arg->getAttribute("beta"))  obj->setBeta(atof(arg->getAttribute("beta")));
        if (arg->getAttribute("fmax"))  obj->setFMax(atof(arg->getAttribute("fmax")));
        if (arg->getAttribute("dmax"))  obj->setDMax(atof(arg->getAttribute("dmax")));
        else if (arg->getAttribute("d0"))  obj->setDMax(2*atof(arg->getAttribute("d0")));
        if (arg->getAttribute("d0"))  obj->setD0(atof(arg->getAttribute("d0")));
        if (arg->getAttribute("p0"))  obj->setP0(atof(arg->getAttribute("p0")));
    }
}
}
}
}

// Each instance of our class must be compiled
template class LennardJonesForceField<Vec3fTypes>;
template class LennardJonesForceField<Vec3dTypes>;

// And registered in the Factory
Creator< XML::ForceFieldNode::Factory, LennardJonesForceField<Vec3fTypes> > LennardJonesForceField3fClass("LennardJones", true);
Creator< XML::ForceFieldNode::Factory, LennardJonesForceField<Vec3dTypes> > LennardJonesForceField3dClass("LennardJones", true);
