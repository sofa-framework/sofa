#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MASS_INL
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MASS_INL

#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/core/componentmodel/behavior/ForceField.inl>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class DataTypes>
Mass<DataTypes>::Mass(MechanicalState<DataTypes> *mm)
    : ForceField<DataTypes>(mm)
{
}

template<class DataTypes>
Mass<DataTypes>::~Mass()
{
}

template<class DataTypes>
void Mass<DataTypes>::addMDx()
{
    if (this->mstate)
        addMDx(*this->mstate->getF(), *this->mstate->getDx());
}

template<class DataTypes>
void Mass<DataTypes>::accFromF()
{
    if (this->mstate)
        accFromF(*this->mstate->getDx(), *this->mstate->getF());
}

template<class DataTypes>
double Mass<DataTypes>::getKineticEnergy()
{
    if (this->mstate)
        return getKineticEnergy(*this->mstate->getV());
    return 0.;
}

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
