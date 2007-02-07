#include <sofa/component/forcefield/SPHFluidForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>


namespace sofa
{

namespace helper
{
using namespace component::forcefield;
template<class DataTypes>
void create(SPHFluidForceField<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< SPHFluidForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("radius"))  obj->setParticleRadius((typename DataTypes::Coord::value_type)atof(arg->getAttribute("radius")));
        if (arg->getAttribute("mass"))  obj->setParticleMass((typename DataTypes::Coord::value_type)atof(arg->getAttribute("mass")));
        if (arg->getAttribute("pressure"))  obj->setPressureStiffness((typename DataTypes::Coord::value_type)atof(arg->getAttribute("pressure")));
        if (arg->getAttribute("density"))  obj->setDensity0((typename DataTypes::Coord::value_type)atof(arg->getAttribute("density")));
        if (arg->getAttribute("viscosity"))  obj->setViscosity((typename DataTypes::Coord::value_type)atof(arg->getAttribute("viscosity")));
        if (arg->getAttribute("surfaceTension"))  obj->setSurfaceTension((typename DataTypes::Coord::value_type)atof(arg->getAttribute("surfaceTension")));
    }
}
} // namespace helper

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

using helper::Creator;

// And registered in the Factory
Creator<simulation::tree::xml::ObjectFactory, SPHFluidForceField<Vec3fTypes> > SPHFluidForceField3fClass("SPHFluidForceField", true);
Creator<simulation::tree::xml::ObjectFactory, SPHFluidForceField<Vec3dTypes> > SPHFluidForceField3dClass("SPHFluidForceField", true);

} // namespace forcefield

} // namespace component

} // namespace sofa

