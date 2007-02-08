#include <sofa/component/forcefield/SparseGridSpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(SparseGridSpringForceField)

using namespace sofa::defaulttype;

template class SparseGridSpringForceField<Vec3dTypes>;
template class SparseGridSpringForceField<Vec3fTypes>;

template<class DataTypes>
void create(SparseGridSpringForceField<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< SparseGridSpringForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("stiffness")) obj->setStiffness((typename DataTypes::Coord::value_type)atof(arg->getAttribute("stiffness")));
        if (arg->getAttribute("linesStiffness")) obj->setLinesStiffness((typename DataTypes::Coord::value_type)atof(arg->getAttribute("linesStiffness")));
        if (arg->getAttribute("quadsStiffness")) obj->setLinesStiffness((typename DataTypes::Coord::value_type)atof(arg->getAttribute("quadsStiffness")));
        if (arg->getAttribute("cubesStiffness")) obj->setLinesStiffness((typename DataTypes::Coord::value_type)atof(arg->getAttribute("cubesStiffness")));
        if (arg->getAttribute("damping")) obj->setDamping((typename DataTypes::Coord::value_type)atof(arg->getAttribute("damping")));
        if (arg->getAttribute("linesDamping")) obj->setLinesDamping((typename DataTypes::Coord::value_type)atof(arg->getAttribute("linesDamping")));
        if (arg->getAttribute("quadsDamping")) obj->setLinesDamping((typename DataTypes::Coord::value_type)atof(arg->getAttribute("quadsDamping")));
        if (arg->getAttribute("cubesDamping")) obj->setLinesDamping((typename DataTypes::Coord::value_type)atof(arg->getAttribute("cubesDamping")));


    }
}

Creator<simulation::tree::xml::ObjectFactory, SparseGridSpringForceField<Vec3dTypes> > SparseGridSpringForceFieldVec3dClass("SparseGridSpringForceField", true);
Creator<simulation::tree::xml::ObjectFactory, SparseGridSpringForceField<Vec3fTypes> > SparseGridSpringForceFieldVec3fClass("SparseGridSpringForceField", true);

} // namespace forcefield

} // namespace component

} // namespace sofa

