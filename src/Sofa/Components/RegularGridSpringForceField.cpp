#include "RegularGridSpringForceField.inl"
#include "Common/Vec3Types.h"
#include "XML/DynamicNode.h"
#include "Sofa/Core/MechanicalObject.h"
#include "XML/ForceFieldNode.h"
#include "XML/InteractionForceFieldNode.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(RegularGridSpringForceField)

using namespace Common;

template class RegularGridSpringForceField<Vec3dTypes>;
template class RegularGridSpringForceField<Vec3fTypes>;

template<class DataTypes>
void create(RegularGridSpringForceField<DataTypes>*& obj, XML::Node<Core::ForceField>* arg)
{
    XML::createWithParent< RegularGridSpringForceField<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("stiffness")) obj->setStiffness(atof(arg->getAttribute("stiffness")));
        if (arg->getAttribute("linesStiffness")) obj->setLinesStiffness(atof(arg->getAttribute("linesStiffness")));
        if (arg->getAttribute("quadsStiffness")) obj->setLinesStiffness(atof(arg->getAttribute("quadsStiffness")));
        if (arg->getAttribute("cubesStiffness")) obj->setLinesStiffness(atof(arg->getAttribute("cubesStiffness")));
        if (arg->getAttribute("damping")) obj->setDamping(atof(arg->getAttribute("damping")));
        if (arg->getAttribute("linesDamping")) obj->setLinesDamping(atof(arg->getAttribute("linesDamping")));
        if (arg->getAttribute("quadsDamping")) obj->setLinesDamping(atof(arg->getAttribute("quadsDamping")));
        if (arg->getAttribute("cubesDamping")) obj->setLinesDamping(atof(arg->getAttribute("cubesDamping")));
    }
}

Creator< XML::ForceFieldNode::Factory, RegularGridSpringForceField<Vec3dTypes> > RegularGridSpringForceFieldVec3dClass("RegularGridSpringForceField", true);
Creator< XML::ForceFieldNode::Factory, RegularGridSpringForceField<Vec3fTypes> > RegularGridSpringForceFieldVec3fClass("RegularGridSpringForceField", true);
/*
template<class DataTypes>
void create(RegularGridSpringForceField<DataTypes>*& obj, XML::Node<Core::InteractionForceField>* arg)
{
	XML::createWith2Objects< RegularGridSpringForceField<DataTypes>, Core::MechanicalObject<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
}

Creator< XML::InteractionForceFieldNode::Factory, RegularGridSpringForceField<Vec3dTypes> > RegularGridSpringInteractionForceFieldVec3dClass("RegularGridSpringForceField", true);
Creator< XML::InteractionForceFieldNode::Factory, RegularGridSpringForceField<Vec3fTypes> > RegularGridSpringInteractionForceFieldVec3fClass("RegularGridSpringForceField", true);
*/
} // namespace Components

} // namespace Sofa
