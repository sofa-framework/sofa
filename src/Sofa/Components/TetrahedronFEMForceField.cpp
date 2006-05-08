#include "TetrahedronFEMForceField.inl"
#include "Common/Vec3Types.h"
#include "XML/DynamicNode.h"
#include "Sofa/Core/MechanicalObject.h"
#include "XML/ForceFieldNode.h"

//#include <typeinfo>

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(TetrahedronFEMForceField)

using namespace Common;

template class TetrahedronFEMForceField<Vec3dTypes>;
template class TetrahedronFEMForceField<Vec3fTypes>;

template<class DataTypes>
void create(TetrahedronFEMForceField<DataTypes>*& obj, XML::Node<Core::ForceField>* arg)
{
    XML::createWithParent< TetrahedronFEMForceField<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        obj->setPoissonRatio((typename TetrahedronFEMForceField<DataTypes>::Real)atof(arg->getAttribute("poissonRatio","0.49")));
        obj->setYoungModulus((typename TetrahedronFEMForceField<DataTypes>::Real)atof(arg->getAttribute("youngModulus","100000")));
        std::string method = arg->getAttribute("method","");
        if (method == "small")
            obj->setMethod(TetrahedronFEMForceField<DataTypes>::SMALL);
        else if (method == "large")
            obj->setMethod(TetrahedronFEMForceField<DataTypes>::LARGE);
        else if (method == "polar")
            obj->setMethod(TetrahedronFEMForceField<DataTypes>::POLAR);
        obj->setUpdateStiffnessMatrix(std::string(arg->getAttribute("updateStiffnessMatrix","false"))=="true");
        obj->setComputeGlobalMatrix(std::string(arg->getAttribute("computeGlobalMatrix","false"))=="true");
    }
}

Creator< XML::ForceFieldNode::Factory, TetrahedronFEMForceField<Vec3dTypes> > TetrahedronFEMForceFieldVec3dClass("TetrahedronFEMForceField", true);
Creator< XML::ForceFieldNode::Factory, TetrahedronFEMForceField<Vec3fTypes> > TetrahedronFEMForceFieldVec3fClass("TetrahedronFEMForceField", true);

} // namespace Components

} // namespace Sofa
