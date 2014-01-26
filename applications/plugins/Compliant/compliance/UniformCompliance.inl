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
    , damping( initData(&damping, Real(0), "damping", "uniform viscous damping."))
	  
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

        if( helper::rabs(compliance.getValue()) <= std::numeric_limits<Real>::epsilon() && this->rayleighStiffness.getValue() )
        {
            serr<<"Warning: a null compliance can not generate rayleighDamping, forced to 0"<<sendl;
            this->rayleighStiffness.setValue(0);
        }
    }
    else matC.compressedMatrix.resize(0,0);

    // matK must be computed since it is used by MechanicalComputeComplianceForceVisitor to compute the compliance forces
//    if( !this->isCompliance.getValue() || this->rayleighStiffness.getValue() )
//    {
        // the stiffness df/dx is the opposite of the inverse compliance
        Real k = helper::rabs(compliance.getValue()) > std::numeric_limits<Real>::epsilon() ?
                -1 / compliance.getValue() :
                -std::numeric_limits<Real>::max();

        matK.resize(state->getMatrixSize(), state->getMatrixSize());

        for(unsigned i=0, n = state->getMatrixSize(); i < n; i++) {
            matK.compressedMatrix.startVec(i);
            matK.compressedMatrix.insertBack(i, i) = k;
        }

        matK.compressedMatrix.finalize();
//    }
//    else matK.compressedMatrix.resize(0,0);


    // TODO if(this->isCompliance.getValue() && this->rayleighStiffness.getValue()) mettre rayleigh dans B mais attention à kfactor avec/sans rayleigh factor


	if( damping.getValue() > 0 ) {
		SReal d = damping.getValue();
		
		matB.resize(state->getMatrixSize(), state->getMatrixSize());
		
		for(unsigned i=0, n = state->getMatrixSize(); i < n; i++) {
			matB.compressedMatrix.startVec(i);
			matB.compressedMatrix.insertBack(i, i) = -d;
		}

		matB.compressedMatrix.finalize();
	}
    else matB.compressedMatrix.resize(0,0);
	
}


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
//	if( damping.getValue() > 0 ) // B is empty in that case
    {
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

    if( kfactor )
    {
        matK.addMult( _df, _dx, kfactor );
    }

    if( damping.getValue() > 0 )
    {
        Real bfactor = (Real)mparams->bFactor();
        matB.addMult( _df, _dx, bfactor );
    }
}


}
}
}
