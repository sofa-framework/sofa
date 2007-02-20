#include "CudaTypes.h"
#include "Sofa-old/Components/Common/ObjectFactory.h"
#include "CudaUniformMass.inl"

namespace Sofa
{

namespace Components
{

// \todo This code is duplicated Sofa/Components/UniformMass.cpp

namespace Common   // \todo Why this must be inside Common namespace
{

template<class DataTypes, class MassType>
void create(UniformMass<DataTypes, MassType>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< UniformMass<DataTypes, MassType>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("mass"))
        {
            obj->setMass((MassType)atof(arg->getAttribute("mass")));
        }
        if (arg->getAttribute("totalmass"))
        {
            obj->setTotalMass(atof(arg->getAttribute("totalmass")));
        }
    }
}
}
}

namespace Contrib
{

namespace CUDA
{
using namespace Components::Common;
using namespace Components;

SOFA_DECL_CLASS(UniformMassCuda)

Creator< ObjectFactory, UniformMass<CudaVec3fTypes,float > > UniformMassCuda3fClass("UniformMass",true);

} // namespace CUDA

} // namespace Contrib

} // namespace Sofa
