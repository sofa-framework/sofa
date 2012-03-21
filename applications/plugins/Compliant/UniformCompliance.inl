#include "UniformCompliance.h"
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{
namespace component
{
namespace compliance
{

template<class DataTypes>
UniformCompliance<DataTypes>::UniformCompliance( core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , compliance( initData(&compliance, Block(), "compliance", "Compliance value uniformly applied to all the DOF."))
{
}

template<class DataTypes>
void UniformCompliance<DataTypes>::init()
{
    Inherit::init();
    if( this->getMState()==NULL ) serr<<"UniformCompliance<DataTypes>::init(), no mstate !" << sendl;
    reinit();
}

template<class DataTypes>
void UniformCompliance<DataTypes>::reinit()
{
    core::behavior::BaseMechanicalState* state = this->getContext()->getMechanicalState();
    assert(state);
    matC.resize(state->getMatrixSize(),state->getMatrixSize());
//    cerr<<"UniformCompliance<DataTypes>::reinit, compliance.getValue()[0][0] = " << compliance.getValue()[0][0] << endl;
    for(unsigned i=0; i<state->getMatrixSize(); i++)
    {
        matC.set(i,i,compliance.getValue()[0][0]);
    }
    matC.endEdit();
}

//// Compute the displacement in response to the given force
//template<class DataTypes>
//void UniformCompliance<DataTypes>::computeDisplacement(const core::MechanicalParams* /*mparams*/, DataVecDeriv& displacement, const DataVecDeriv& force )
//{
//    helper::WriteAccessor< DataVecDeriv > d(displacement);
//    helper::ReadAccessor< DataVecDeriv > f(force);
//    for(unsigned i=0; i<d.size(); i++)
//        d[i] = compliance.getValue() * f[i];
//}

template<class DataTypes>
void UniformCompliance<DataTypes>::setConstraint(const core::ComplianceParams* params, core::MultiVecDerivId fId )
{
//    const DataVecCoord *xd = params->readX(this->mstate);
    helper::ReadAccessor< DataVecCoord > x = params->readX(this->mstate);
    helper::ReadAccessor< DataVecDeriv > v = params->readV(this->mstate);
    helper::WriteAccessor< DataVecDeriv > f = *fId[this->mstate.get(params)].write();
    Real alpha = params->implicitVelocity();
    Real beta  = params->implicitPosition();
    Real h     = params->dt();
    Real d     = this->dampingRatio.getValue();

    for(unsigned i=0; i<f.size(); i++)
        f[i] = -( x[i] + v[i] * (d + alpha*h) ) / (alpha * (h*beta +d));
}

/// return a pointer to the compliance matrix
template<class DataTypes>
const sofa::defaulttype::BaseMatrix* UniformCompliance<DataTypes>::getMatrix(const core::MechanicalParams*)
{
    return &matC;
}

}
}
}
