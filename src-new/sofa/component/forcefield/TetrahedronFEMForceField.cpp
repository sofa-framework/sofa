#include <sofa/component/forcefield/TetrahedronFEMForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
//#include <typeinfo>


namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(TetrahedronFEMForceField)

using namespace sofa::defaulttype;

template class TetrahedronFEMForceField<Vec3dTypes>;
template class TetrahedronFEMForceField<Vec3fTypes>;

template<class DataTypes>
void create(TetrahedronFEMForceField<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< TetrahedronFEMForceField<DataTypes>, component::MechanicalObject<DataTypes> >(obj, arg);
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

Creator<simulation::tree::xml::ObjectFactory, TetrahedronFEMForceField<Vec3dTypes> > TetrahedronFEMForceFieldVec3dClass("TetrahedronFEMForceField", true);
Creator<simulation::tree::xml::ObjectFactory, TetrahedronFEMForceField<Vec3fTypes> > TetrahedronFEMForceFieldVec3fClass("TetrahedronFEMForceField", true);

} // namespace forcefield

} // namespace component

} // namespace sofa

