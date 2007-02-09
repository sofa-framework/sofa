#include <sofa/component/constraint/FixedPlaneConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(FixedPlaneConstraint)

template class FixedPlaneConstraint<Vec3dTypes>;
template class FixedPlaneConstraint<Vec3fTypes>;


int FixedPlaneConstraintClass = core::RegisterObject("TODO-FixedPlaneConstraintClass")
        .add< FixedPlaneConstraint<Vec3dTypes> >()
        .add< FixedPlaneConstraint<Vec3fTypes> >()
        ;

} // namespace constraint

} // namespace component

} // namespace sofa

