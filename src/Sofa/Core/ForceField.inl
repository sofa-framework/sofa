#ifndef SOFA_CORE_FORCEFIELD_INL
#define SOFA_CORE_FORCEFIELD_INL

#include "ForceField.h"

namespace Sofa
{

namespace Core
{

template<class DataTypes>
ForceField<DataTypes>::ForceField(MechanicalModel<DataTypes> *mm)
    : mmodel(mm)
{
}

template<class DataTypes>
ForceField<DataTypes>::~ForceField()
{
}

template<class DataTypes>
void ForceField<DataTypes>::init()
{
    BasicForceField::init();
    mmodel = dynamic_cast< MechanicalModel<DataTypes>* >(getContext()->getMechanicalModel());
}

template<class DataTypes>
void ForceField<DataTypes>::addForce()
{
    if (mmodel)
        addForce(*mmodel->getF(), *mmodel->getX(), *mmodel->getV());
}

template<class DataTypes>
void ForceField<DataTypes>::addDForce()
{
    if (mmodel)
        addDForce(*mmodel->getF(), *mmodel->getDx());
}

template<class DataTypes>
double ForceField<DataTypes>::getPotentialEnergy()
{
    if (mmodel)
        return getPotentialEnergy(*mmodel->getX());
    else return 0;
}

} // namespace Core

} // namespace Sofa

#endif
