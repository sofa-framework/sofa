#include "SPHFluidForceField.inl"
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
void create(SPHFluidForceField<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< SPHFluidForceField<DataTypes>, MechanicalModel<DataTypes> >(obj, arg);
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
}

SOFA_DECL_CLASS(SPHFluidForceField)

// Each instance of our class must be compiled
template class SPHFluidForceField<Vec3fTypes>;
template class SPHFluidForceField<Vec3dTypes>;

// And registered in the Factory
Creator< ObjectFactory, SPHFluidForceField<Vec3fTypes> > SPHFluidForceField3fClass("SPHFluidForceField", true);
Creator< ObjectFactory, SPHFluidForceField<Vec3dTypes> > SPHFluidForceField3dClass("SPHFluidForceField", true);

} // namespace Sofa

} // namespace Components
