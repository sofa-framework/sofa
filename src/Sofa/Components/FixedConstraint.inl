#ifndef SOFA_COMPONENTS_FIXEDCONSTRAINT_INL
#define SOFA_COMPONENTS_FIXEDCONSTRAINT_INL

#include "Sofa/Core/Constraint.inl"
#include "FixedConstraint.h"
#include "GL/template.h"
#include "Common/RigidTypes.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class DataTypes>
FixedConstraint<DataTypes>::FixedConstraint()
    : Core::Constraint<DataTypes>(NULL)
{
}


template <class DataTypes>
FixedConstraint<DataTypes>::FixedConstraint(Core::MechanicalModel<DataTypes>* mmodel)
    : Core::Constraint<DataTypes>(mmodel)
{
}

template <class DataTypes>
FixedConstraint<DataTypes>::~FixedConstraint()
{
}

template <class DataTypes>
FixedConstraint<DataTypes>*  FixedConstraint<DataTypes>::addConstraint(int index)
{
    this->indices.insert(index);
    return this;
}

template <class DataTypes>
FixedConstraint<DataTypes>*  FixedConstraint<DataTypes>::removeConstraint(int index)
{
    this->indices.erase(index);
    return this;
}

// -- Mass interface
template <class DataTypes>
void FixedConstraint<DataTypes>::projectResponse(VecDeriv& res)
{
    for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
    {
        res[*it] = Deriv();
    }
}

// Matrix Integration interface
template <class DataTypes>
void FixedConstraint<DataTypes>::applyConstraint(Components::Common::SofaBaseMatrix *mat, unsigned int &offset)
{
    std::cout << "applyConstraint in Matrix with offset = " << offset << std::endl;

    for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
    {
        mat->element(3 * (*it) + offset, 3 * (*it) + offset) = 1.0;
        mat->element(3 * (*it) + offset, 3 * (*it) + offset + 1) = 0.0;
        mat->element(3 * (*it) + offset, 3 * (*it) + offset + 2) = 0.0;

        mat->element(3 * (*it) + offset + 1, 3 * (*it) + offset) = 0.0;
        mat->element(3 * (*it) + offset + 1, 3 * (*it) + offset + 1) = 1.0;
        mat->element(3 * (*it) + offset + 1, 3 * (*it) + offset + 2) = 0.0;

        mat->element(3 * (*it) + offset + 2, 3 * (*it) + offset) = 0.0;
        mat->element(3 * (*it) + offset + 2, 3 * (*it) + offset + 1) = 0.0;
        mat->element(3 * (*it) + offset + 2, 3 * (*it) + offset + 2) = 1.0;
    }
}

template <class DataTypes>
void FixedConstraint<DataTypes>::applyConstraint(Components::Common::SofaBaseVector *vect, unsigned int &offset)
{
    std::cout << "applyConstraint in Vector with offset = " << offset << std::endl;

    for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
    {
        vect->element(3 * (*it) + offset) = 0.0;
        vect->element(3 * (*it) + offset + 1) = 0.0;
        vect->element(3 * (*it) + offset + 2) = 0.0;
    }
}

template <class DataTypes>
void FixedConstraint<DataTypes>::draw()
{
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *this->mmodel->getX();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
    {
        GL::glVertexT(x[*it]);
    }
    glEnd();
}

// Specialization for rigids
template <>
void FixedConstraint<RigidTypes >::draw();
template <>
void FixedConstraint<RigidTypes >::projectResponse(VecDeriv& dx);

} // namespace Components

} // namespace Sofa

#endif
