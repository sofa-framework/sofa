#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_FORCEFIELD_INL
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_FORCEFIELD_INL

#include <sofa/core/objectmodel/Field.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class DataTypes>
ForceField<DataTypes>::ForceField(MechanicalState<DataTypes> *mm)
    : mstate(mm)
{
}

template<class DataTypes>
ForceField<DataTypes>::~ForceField()
{
}

template<class DataTypes>
void ForceField<DataTypes>::init()
{
    BaseForceField::init();
    mstate = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
}

template<class DataTypes>
void ForceField<DataTypes>::addForce()
{
    if (mstate)
        addForce(*mstate->getF(), *mstate->getX(), *mstate->getV());
}

template<class DataTypes>
void ForceField<DataTypes>::addDForce()
{
    if (mstate)
        addDForce(*mstate->getF(), *mstate->getDx());
}


template<class DataTypes>
double ForceField<DataTypes>::getPotentialEnergy()
{
    if (mstate)
        return getPotentialEnergy(*mstate->getX());
    else return 0;
}

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
