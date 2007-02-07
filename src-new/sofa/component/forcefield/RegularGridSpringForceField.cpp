#include <sofa/component/forcefield/RegularGridSpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(RegularGridSpringForceField)

using namespace sofa::defaulttype;

template class RegularGridSpringForceField<Vec3dTypes>;
template class RegularGridSpringForceField<Vec3fTypes>;

template<class DataTypes>
void create(RegularGridSpringForceField<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    XML::createWithParent< RegularGridSpringForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("stiffness")) obj->setStiffness((typename DataTypes::Coord::value_type)atof(arg->getAttribute("stiffness")));
        if (arg->getAttribute("linesStiffness")) obj->setLinesStiffness((typename DataTypes::Coord::value_type)atof(arg->getAttribute("linesStiffness")));
        if (arg->getAttribute("quadsStiffness")) obj->setQuadsStiffness((typename DataTypes::Coord::value_type)atof(arg->getAttribute("quadsStiffness")));
        if (arg->getAttribute("cubesStiffness")) obj->setCubesStiffness((typename DataTypes::Coord::value_type)atof(arg->getAttribute("cubesStiffness")));
        if (arg->getAttribute("damping")) obj->setDamping((typename DataTypes::Coord::value_type)atof(arg->getAttribute("damping")));
        if (arg->getAttribute("linesDamping")) obj->setLinesDamping((typename DataTypes::Coord::value_type)atof(arg->getAttribute("linesDamping")));
        if (arg->getAttribute("quadsDamping")) obj->setQuadsDamping((typename DataTypes::Coord::value_type)atof(arg->getAttribute("quadsDamping")));
        if (arg->getAttribute("cubesDamping")) obj->setCubesDamping((typename DataTypes::Coord::value_type)atof(arg->getAttribute("cubesDamping")));
    }
}

Creator<simulation::tree::xml::ObjectFactory, RegularGridSpringForceField<Vec3dTypes> > RegularGridSpringForceFieldVec3dClass("RegularGridSpringForceField", true);
Creator<simulation::tree::xml::ObjectFactory, RegularGridSpringForceField<Vec3fTypes> > RegularGridSpringForceFieldVec3fClass("RegularGridSpringForceField", true);

} // namespace forcefield

} // namespace component

} // namespace sofa

