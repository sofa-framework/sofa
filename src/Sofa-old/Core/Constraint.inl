#ifndef SOFA_CORE_CONSTRAINT_INL
#define SOFA_CORE_CONSTRAINT_INL

#include "Constraint.h"

namespace Sofa
{

namespace Core
{

template<class DataTypes>
Constraint<DataTypes>::Constraint(MechanicalModel<DataTypes> *mm)
    :  endTime( dataField(&endTime,(Real)-1,"endTime","The constraint stops acting after the given value. Une a negative value for infinite constraints") )
    , mmodel(mm)
{
}

template<class DataTypes>
Constraint<DataTypes>::~Constraint()
{
}

template <class DataTypes>
bool   Constraint<DataTypes>::isActive() const
{
    if( endTime.getValue()<0 ) return true;
    return endTime.getValue()>getContext()->getTime();
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
    if( !isActive() ) return;
    if (mmodel)
        projectResponse(*mmodel->getDx());
}
template<class DataTypes>
void Constraint<DataTypes>::projectVelocity()
{
    if( !isActive() ) return;
    if (mmodel)
        projectVelocity(*mmodel->getV());
}
template<class DataTypes>
void Constraint<DataTypes>::projectPosition()
{
    if( !isActive() ) return;
    if (mmodel)
        projectPosition(*mmodel->getX());
}

template<class DataTypes>
void Constraint<DataTypes>::applyConstraint()
{
    if( !isActive() ) return;
    if (mmodel)
        applyConstraint(*mmodel->getC());
}

} // namespace Core

} // namespace Sofa

#endif
