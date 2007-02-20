#include "CudaTypes.h"
#include "Sofa-old/Components/Common/ObjectFactory.h"
#include "CudaFixedConstraint.inl"

namespace Sofa
{

namespace Components
{

// \todo This code is duplicated Sofa/Components/FixedConstraint.cpp

namespace Common   // \todo Why this must be inside Common namespace
{
template<class DataTypes>
void create(FixedConstraint<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< FixedConstraint<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        obj->parseFields( arg->getAttributeMap() );
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

SOFA_DECL_CLASS(FixedConstraintCuda)

Creator< ObjectFactory, FixedConstraint<CudaVec3fTypes> > FixedConstraintCuda3fClass("FixedConstraint",true);

} // namespace CUDA

} // namespace Contrib

} // namespace Sofa
