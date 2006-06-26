#ifndef SOFA_CORE_MASS_INL
#define SOFA_CORE_MASS_INL

#include "Mass.h"
#include "ForceField.inl"

namespace Sofa
{

namespace Core
{

template<class DataTypes>
Mass<DataTypes>::Mass(MechanicalModel<DataTypes> *mm)
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
    if (this->mmodel)
        addMDx(*this->mmodel->getF(), *this->mmodel->getDx());
}

template<class DataTypes>
void Mass<DataTypes>::accFromF()
{
    if (mmodel)
        accFromF(*this->mmodel->getDx(), *this->mmodel->getF());
}

} // namespace Core

} // namespace Sofa

#endif
