#include <sofa/component/constraint/LinearMovementConstraint.inl>
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

SOFA_DECL_CLASS(LinearMovementConstraint)

int LinearMovementConstraintClass = core::RegisterObject("translate given particles")
        .add< LinearMovementConstraint<Vec3dTypes> >()
        .add< LinearMovementConstraint<Vec3fTypes> >()
        .add< LinearMovementConstraint<Vec2dTypes> >()
        .add< LinearMovementConstraint<Vec2fTypes> >()
        .add< LinearMovementConstraint<Vec1dTypes> >()
        .add< LinearMovementConstraint<Vec1fTypes> >()
        .add< LinearMovementConstraint<Vec6dTypes> >()
        .add< LinearMovementConstraint<Vec6fTypes> >()
        .add< LinearMovementConstraint<Rigid3dTypes> >()
        .add< LinearMovementConstraint<Rigid3fTypes> >()
        ;

template <>
void LinearMovementConstraint<Rigid3dTypes>::draw()
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
void LinearMovementConstraint<Rigid3fTypes>::draw()
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


template class LinearMovementConstraint<Vec3dTypes>;
template class LinearMovementConstraint<Vec3fTypes>;
template class LinearMovementConstraint<Vec2dTypes>;
template class LinearMovementConstraint<Vec2fTypes>;
template class LinearMovementConstraint<Vec1dTypes>;
template class LinearMovementConstraint<Vec1fTypes>;
template class LinearMovementConstraint<Vec6dTypes>;
template class LinearMovementConstraint<Vec6fTypes>;
template class LinearMovementConstraint<Rigid3dTypes>;
template class LinearMovementConstraint<Rigid3fTypes>;

} // namespace constraint

} // namespace component

} // namespace sofa

