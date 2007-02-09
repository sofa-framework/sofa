#include <sofa/component/constraint/FixedConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(FixedConstraint)

int FixedConstraintClass = core::RegisterObject("TODO")
        .add< FixedConstraint<Vec3dTypes> >()
        .add< FixedConstraint<Vec3fTypes> >()
        .add< FixedConstraint<RigidTypes> >()
        ;

template <>
void FixedConstraint<RigidTypes>::draw()
{
    const SetIndexArray & indices = f_indices.getValue().getArray();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mstate->getX();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        gl::glVertexT(x[0].getCenter());
    }
    glEnd();
}

template <>
void FixedConstraint<RigidTypes>::projectResponse(VecDeriv& res)
{
    res[0] = Deriv();
}

template class FixedConstraint<Vec3dTypes>;
template class FixedConstraint<Vec3fTypes>;

} // namespace constraint

} // namespace component

} // namespace sofa

