#ifndef SOFA_COMPONENT_CONSTRAINT_FIXEDPLANECONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_FIXEDPLANECONSTRAINT_INL

#include <sofa/core/componentmodel/behavior/Constraint.inl>
#include <sofa/component/constraint/FixedPlaneConstraint.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;

template <class DataTypes>
FixedPlaneConstraint<DataTypes>::FixedPlaneConstraint()
    : core::componentmodel::behavior::Constraint<DataTypes>(NULL)
{
    selectVerticesFromPlanes=false;
}


template <class DataTypes>
FixedPlaneConstraint<DataTypes>::FixedPlaneConstraint(core::componentmodel::behavior::MechanicalState<DataTypes>* mstate)
    : core::componentmodel::behavior::Constraint<DataTypes>(mstate),direction(0.0,0.0,1.0)
{
    selectVerticesFromPlanes=false;
}

template <class DataTypes>
FixedPlaneConstraint<DataTypes>::~FixedPlaneConstraint()
{
}

template <class DataTypes>
FixedPlaneConstraint<DataTypes>*  FixedPlaneConstraint<DataTypes>::addConstraint(int index)
{
    this->indices.insert(index);
    return this;
}

template <class DataTypes>
FixedPlaneConstraint<DataTypes>*  FixedPlaneConstraint<DataTypes>::removeConstraint(int index)
{
    this->indices.erase(index);
    return this;
}

// -- Mass interface
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectResponse(VecDeriv& res)
{

    for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
    {
        /// only constraint one projection of the displacement to be zero
        res[*it]-= direction*dot(res[*it],direction);
    }

}
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::setDirection(Coord dir)
{
    if (dir.norm2()>0)
    {
        direction=dir;
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::selectVerticesAlongPlane()
{
    VecCoord& x = *this->mstate->getX();
    unsigned int i;
    for(i=0; i<x.size(); ++i)
    {
        if (isPointInPlane(x[i]))
            addConstraint(i);
    }

}
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::init()
{
    if (selectVerticesFromPlanes)
        selectVerticesAlongPlane();

}
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::draw()
{
    if (!getContext()->getShowBehaviorModels()) return;
    const VecCoord& x = *this->mstate->getX();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,1.0,0.5,1);
    glBegin (GL_POINTS);
    for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
    {
        GL::glVertexT(x[*it]);
    }
    glEnd();
}


} // namespace constraint

} // namespace component

} // namespace sofa

#endif
