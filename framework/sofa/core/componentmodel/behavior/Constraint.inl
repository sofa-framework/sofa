#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_CONSTRAINT_INL
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_CONSTRAINT_INL

#include <sofa/core/componentmodel/behavior/Constraint.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class DataTypes>
Constraint<DataTypes>::Constraint(MechanicalState<DataTypes> *mm)
    :  endTime( dataField(&endTime,(Real)-1,"endTime","The constraint stops acting after the given value. Une a negative value for infinite constraints") )
    , mstate(mm)
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
    BaseConstraint::init();
    mstate = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
}

template<class DataTypes>
void Constraint<DataTypes>::projectResponse()
{
    if( !isActive() ) return;
    if (mstate)
        projectResponse(*mstate->getDx());
}
template<class DataTypes>
void Constraint<DataTypes>::projectVelocity()
{
    if( !isActive() ) return;
    if (mstate)
        projectVelocity(*mstate->getV());
}
template<class DataTypes>
void Constraint<DataTypes>::projectPosition()
{
    if( !isActive() ) return;
    if (mstate)
        projectPosition(*mstate->getX());
}

template<class DataTypes>
void Constraint<DataTypes>::applyConstraint()
{
    if( !isActive() ) return;
    if (mstate)
        applyConstraint(*mstate->getC());
}

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
