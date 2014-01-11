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

    if( this->isCompliance.getValue() )
    {
        matC.resize(state->getMatrixSize(), state->getMatrixSize());

        for(unsigned i=0, n = state->getMatrixSize(); i < n; i++) {
            matC.compressedMatrix.startVec(i);
            matC.compressedMatrix.insertBack(i, i) = compliance.getValue();
        }

        matC.compressedMatrix.finalize();
    }

    if( !this->isCompliance.getValue() || this->rayleighStiffness.getValue() )
    {
        // the stiffness df/dx is the opposite of the inverse compliance
        Real k = helper::rabs(compliance.getValue()) >std::numeric_limits<Real>::epsilon() ?
                -1 / compliance.getValue() :
                -std::numeric_limits<Real>::max();

        matK.resize(state->getMatrixSize(), state->getMatrixSize());

        for(unsigned i=0, n = state->getMatrixSize(); i < n; i++) {
            matK.compressedMatrix.startVec(i);
            matK.compressedMatrix.insertBack(i, i) = k;
        }

        matK.compressedMatrix.finalize();
    }


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
    matK.addToBaseMatrix( matrix, kFact, offset );
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
    matK.addMult( _f, _x  );

//    cerr<<"UniformCompliance<DataTypes>::addForce, f after = " << f << endl;
}

template<class DataTypes>
void UniformCompliance<DataTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv& _df,  const DataVecDeriv& _dx)
{
    Real kfactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    matK.addMult( _df, _dx, kfactor );
}


}
}
}
