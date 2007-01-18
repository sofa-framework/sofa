#ifndef SOFA_COMPONENTS_WashingMachineForceField_inl
#define SOFA_COMPONENTS_WashingMachineForceField_inl

#include "Sofa/Core/ForceField.inl"
#include "WashingMachineForceField.h"
#include "Common/config.h"
#include <assert.h>
#include <GL/gl.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Sofa
{

namespace Components
{

template<class DataTypes>
void WashingMachineForceField<DataTypes>::addForce(VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1)
{
    for(int i=0; i<6; ++i)
    {
        _planes[i]->addForce(f1,p1,v1);
        _planes[i]->rotate(Deriv(1,0,0),.01);
    }
}

template<class DataTypes>
void WashingMachineForceField<DataTypes>::addDForce(VecDeriv& f1, const VecDeriv& dx1)
{
    for(int i=0; i<6; ++i)
        _planes[i]->addDForce(f1,dx1);
}



template <class DataTypes>
double WashingMachineForceField<DataTypes>::getPotentialEnergy(const VecCoord&)
{
    cerr<<"WashingMachineForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}



template<class DataTypes>
void WashingMachineForceField<DataTypes>::draw()
{
    for(int i=0; i<6; ++i)
// 				_planes[i]->draw2(_size.getValue()[0]);
        _planes[i]->draw();
}


} // namespace Components

} // namespace Sofa

#endif
