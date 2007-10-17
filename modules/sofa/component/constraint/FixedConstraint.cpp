#include <sofa/component/constraint/FixedConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/mass/UniformMass.h>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(FixedConstraint)

int FixedConstraintClass = core::RegisterObject("Attach given particles to their initial positions")
        .add< FixedConstraint<Vec3dTypes> >()
        .add< FixedConstraint<Vec3fTypes> >()
        .add< FixedConstraint<Vec2dTypes> >()
        .add< FixedConstraint<Vec2fTypes> >()
        .add< FixedConstraint<Vec1dTypes> >()
        .add< FixedConstraint<Vec1fTypes> >()
        .add< FixedConstraint<Vec6dTypes> >()
        .add< FixedConstraint<Vec6fTypes> >()
        .add< FixedConstraint<Rigid3dTypes> >()
        .add< FixedConstraint<Rigid3fTypes> >()
        .add< FixedConstraint<Rigid2dTypes> >()
        .add< FixedConstraint<Rigid2fTypes> >()
        ;

template <>
void FixedConstraint<Rigid3dTypes>::draw()
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
void FixedConstraint<Rigid3fTypes>::draw()
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
void FixedConstraint<Rigid2dTypes>::draw()
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
void FixedConstraint<Rigid2fTypes>::draw()
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

template class FixedConstraint<Vec3dTypes>;
template class FixedConstraint<Vec3fTypes>;
template class FixedConstraint<Vec2dTypes>;
template class FixedConstraint<Vec2fTypes>;
template class FixedConstraint<Vec1dTypes>;
template class FixedConstraint<Vec1fTypes>;
template class FixedConstraint<Vec6dTypes>;
template class FixedConstraint<Vec6fTypes>;
template class FixedConstraint<Rigid3dTypes>;
template class FixedConstraint<Rigid3fTypes>;
template class FixedConstraint<Rigid2dTypes>;
template class FixedConstraint<Rigid2fTypes>;

} // namespace constraint

} // namespace component

} // namespace sofa

