#ifndef SOFA_CORE_MASS_INL
#define SOFA_CORE_MASS_INL

#include "Mass.h"

namespace Sofa
{

namespace Core
{

template<class DataTypes>
Mass<DataTypes>::Mass(MechanicalModel<DataTypes> *mm)
    : mmodel(mm)
{
}

template<class DataTypes>
Mass<DataTypes>::~Mass()
{
}

template<class DataTypes>
void Mass<DataTypes>::init()
{
    BasicMass::init();
    mmodel = dynamic_cast< MechanicalModel<DataTypes>* >(getContext()->getMechanicalModel());
    assert(mmodel);
}

template<class DataTypes>
void Mass<DataTypes>::addMDx()
{
    if (mmodel)
        addMDx(*mmodel->getF(), *mmodel->getDx());
}

template<class DataTypes>
void Mass<DataTypes>::accFromF()
{
    if (mmodel)
        accFromF(*mmodel->getDx(), *mmodel->getF());
}

template<class DataTypes>
void Mass<DataTypes>::computeForce()
{
    if (mmodel)
        computeForce(*mmodel->getF(), *mmodel->getX(), *mmodel->getV());
}

template<class DataTypes>
void Mass<DataTypes>::computeDf()
{
    if (mmodel)
        computeDf(*mmodel->getF(), *mmodel->getX(), *mmodel->getV(), *mmodel->getDx());
}

} // namespace Core

} // namespace Sofa

#endif
