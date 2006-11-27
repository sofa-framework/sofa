#ifndef SOFA_COMPONENTS_FIXEDPLANECONSTRAINT_INL
#define SOFA_COMPONENTS_FIXEDPLANECONSTRAINT_INL

#include "Sofa/Core/Constraint.inl"
#include "FixedPlaneConstraint.h"
#include "GL/template.h"
#include "Common/RigidTypes.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class DataTypes>
FixedPlaneConstraint<DataTypes>::FixedPlaneConstraint()
    : Core::Constraint<DataTypes>(NULL)
{
    selectVerticesFromPlanes=false;
}


template <class DataTypes>
FixedPlaneConstraint<DataTypes>::FixedPlaneConstraint(Core::MechanicalModel<DataTypes>* mmodel)
    : Core::Constraint<DataTypes>(mmodel),direction(0.0,0.0,1.0)
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
    VecCoord& x = *this->mmodel->getX();
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
    const VecCoord& x = *this->mmodel->getX();
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


} // namespace Components

} // namespace Sofa

#endif
