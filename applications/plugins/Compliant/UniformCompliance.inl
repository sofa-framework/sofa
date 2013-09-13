#include "UniformCompliance.h"
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{
namespace component
{
namespace forcefield
{

template<class DataTypes>
UniformCompliance<DataTypes>::UniformCompliance( core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , compliance( initData(&compliance, (Real)0, "compliance", "Compliance value uniformly applied to all the DOF."))
    , dampingRatio( initData(&dampingRatio, (Real)0.1, "dampingRatio", "weight of the velocity in the constraint violation"))
{
    this->isCompliance.setValue(true);
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

    Real c = this->isCompliance.getValue() ?  compliance.getValue() : -1/compliance.getValue();  // the stiffness df/dx is the opposite of the inverse compliance

    matC.resize(state->getMatrixSize(),state->getMatrixSize());
    for(unsigned i=0; i<state->getMatrixSize(); i++)
    {
//        matC.beginRow(i);
//        matC.insertBack(i,i,c);
        matC.add(i,i,c);
    }
    matC.compress();
}

//template<class DataTypes>
//void UniformCompliance<DataTypes>::setCompliance( Real c )
//{
//    if(isCompliance.getValue() )
//        compliance.setValue(c);
//    else {
//        assert( c!= (Real)0 );
//        for(unsigned i=0; i<C.size(); i++ )
//            C[i][i] = 1/c;
//    }
//    compliance.setValue(C);
//}


template<class DataTypes>
void UniformCompliance<DataTypes>::writeConstraintValue(const core::MechanicalParams* params, core::MultiVecDerivId fId )
{
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

template<class DataTypes>
const sofa::defaulttype::BaseMatrix* UniformCompliance<DataTypes>::getComplianceMatrix(const core::MechanicalParams*)
{
    return &matC;
}

template<class DataTypes>
const sofa::defaulttype::BaseMatrix* UniformCompliance<DataTypes>::getStiffnessMatrix(const core::MechanicalParams*)
{
    return &matC;
}

template<class DataTypes>
void UniformCompliance<DataTypes>::addForce(const core::MechanicalParams *, DataVecDeriv& _f, const DataVecCoord& _x, const DataVecDeriv& _v)
{
    helper::ReadAccessor< DataVecCoord >  x(_x);
    helper::ReadAccessor< DataVecDeriv >  v(_v);
    helper::WriteAccessor< DataVecDeriv > f(_f);

    Real stiffness = -1/compliance.getValue();

//    cerr<<"UniformCompliance<DataTypes>::addForce, f before = " << f << endl;
    for(unsigned i=0; i<f.size(); i++)
        f[i] += ( x[i] + v[i] * dampingRatio.getValue() ) * stiffness;
//    cerr<<"UniformCompliance<DataTypes>::addForce, f after = " << f << endl;

}

template<class DataTypes>
void UniformCompliance<DataTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv& _df,  const DataVecDeriv& _dx)
{
    Real kfactor = mparams->kFactor();

    helper::ReadAccessor< DataVecDeriv >  dx(_dx);
    helper::WriteAccessor< DataVecDeriv > df(_df);

    Real stiffness = -kfactor/compliance.getValue();

    for(unsigned i=0; i<df.size(); i++)
        df[i] += dx[i] * stiffness;

}


}
}
}
