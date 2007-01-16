#ifndef SOFA_CORE_CONSTRAINT_INL
#define SOFA_CORE_CONSTRAINT_INL

#include "Constraint.h"

namespace Sofa
{

namespace Core
{

template<class DataTypes>
Constraint<DataTypes>::Constraint(MechanicalModel<DataTypes> *mm)
    : mmodel(mm)
{
}

template<class DataTypes>
Constraint<DataTypes>::~Constraint()
{
}

template<class DataTypes>
void Constraint<DataTypes>::init()
{
    BasicConstraint::init();
    mmodel = dynamic_cast< MechanicalModel<DataTypes>* >(getContext()->getMechanicalModel());
}

template<class DataTypes>
void Constraint<DataTypes>::projectResponse()
{
    if (mmodel)
        projectResponse(*mmodel->getDx());
}
template<class DataTypes>
void Constraint<DataTypes>::projectVelocity()
{
    if (mmodel)
        projectVelocity(*mmodel->getV());
}
template<class DataTypes>
void Constraint<DataTypes>::projectPosition()
{
    if (mmodel)
        projectPosition(*mmodel->getX());
}

template<class DataTypes>
void Constraint<DataTypes>::applyConstraint()
{
    if (mmodel)
        applyConstraint(*mmodel->getC());
}

} // namespace Core

} // namespace Sofa

#endif
