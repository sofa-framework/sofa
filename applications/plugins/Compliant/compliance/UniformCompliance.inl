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
    , compliance( initData(&compliance, (Real)0, "compliance", "Compliance value uniformly applied to all the DOF.")),
	  damping( initData(&damping, Real(0), "damping", "uniform viscous damping."))
	  
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

	// the stiffness df/dx is the opposite of the inverse compliance
    Real c = this->isCompliance.getValue() ?  
		compliance.getValue() : 
		( helper::rabs(compliance.getValue()) >std::numeric_limits<Real>::epsilon() ? 
		  -1 / compliance.getValue() : 
		  -std::numeric_limits<Real>::max() ); 

    matC.resize(state->getMatrixSize(), state->getMatrixSize());
	
    for(unsigned i=0, n = state->getMatrixSize(); i < n; i++) {
		matC.compressedMatrix.startVec(i);
		matC.compressedMatrix.insertBack(i, i) = c;
    }
    
	matC.compressedMatrix.finalize();

	if( damping.getValue() > 0 ) {
		SReal d = damping.getValue();
		
		matB.resize(state->getMatrixSize(), state->getMatrixSize());
		
		for(unsigned i=0, n = state->getMatrixSize(); i < n; i++) {
			matB.compressedMatrix.startVec(i);
			matB.compressedMatrix.insertBack(i, i) = -d;
		}

		matB.compressedMatrix.finalize();
	}
	
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
const sofa::defaulttype::BaseMatrix* UniformCompliance<DataTypes>::getComplianceMatrix(const core::MechanicalParams*)
{
    return &matC;
}


template<class DataTypes>
void UniformCompliance<DataTypes>::addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, double kFact, unsigned int &offset )
{
	if( this->isCompliance.getValue() ) {
		// max: this might happen if you try to use rayleighStiffness
		// while in compliance mode. you shouldn't, use damping instead.
		std::cerr << "UniformCompliance warning: adding compliance where stiffness is expected !" 
				  << std::endl;
	} else {
		matC.addToBaseMatrix( matrix, kFact, offset );
	}
}

template<class DataTypes>
void UniformCompliance<DataTypes>::addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, double bFact, unsigned int &offset )
{
	if( damping.getValue() > 0 ) {
		matB.addToBaseMatrix( matrix, bFact, offset );
	}
}

template<class DataTypes>
void UniformCompliance<DataTypes>::addForce(const core::MechanicalParams *, DataVecDeriv& _f, const DataVecCoord& _x, const DataVecDeriv& /*_v*/)
{
    helper::ReadAccessor< DataVecCoord >  x(_x);
//    helper::ReadAccessor< DataVecDeriv >  v(_v);
    helper::WriteAccessor< DataVecDeriv > f(_f);

    Real stiffness = helper::rabs(compliance.getValue())>std::numeric_limits<Real>::epsilon() ? -1/compliance.getValue() : -std::numeric_limits<Real>::max();


//    cerr<<"UniformCompliance<DataTypes>::addForce, f before = " << f << endl;
    for(unsigned i=0; i<f.size(); i++)
        f[i] += x[i] * stiffness;
//    cerr<<"UniformCompliance<DataTypes>::addForce, f after = " << f << endl;

}

template<class DataTypes>
void UniformCompliance<DataTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv& _df,  const DataVecDeriv& _dx)
{
    Real kfactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    helper::ReadAccessor< DataVecDeriv >  dx(_dx);
    helper::WriteAccessor< DataVecDeriv > df(_df);

    Real stiffness = helper::rabs(compliance.getValue())>std::numeric_limits<Real>::epsilon() ? -kfactor/compliance.getValue() : -kfactor*std::numeric_limits<Real>::max();

    for(unsigned i=0; i<df.size(); i++)
        df[i] += dx[i] * stiffness;

}


}
}
}
