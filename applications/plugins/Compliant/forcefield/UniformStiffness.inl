#include "UniformStiffness.h"
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
UniformStiffness<DataTypes>::UniformStiffness( core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , stiffness( initData(&stiffness, (Real)0, "stiffness", "stiffness value uniformly applied to all the DOF."))
    , damping( initData(&damping, Real(0), "damping", "uniform viscous damping."))
    , resizable( initData(&resizable, false, "resizable", "can the associated dofs can be resized? (in which case the matrices must be updated)")) 
{
}

template<class DataTypes>
void UniformStiffness<DataTypes>::init()
{
    Inherit::init();
    if( this->getMState()==NULL ) serr<<"UniformStiffness<DataTypes>::init(), no mstate !" << sendl;
    reinit();
}

template<class DataTypes>
void UniformStiffness<DataTypes>::reinit()
{
    core::behavior::BaseMechanicalState* state = this->getContext()->getMechanicalState();
    assert(state);

    const Real& k = stiffness.getValue();


    if( this->isCompliance.getValue() )
    {
        matC.resize(state->getMatrixSize(), state->getMatrixSize());

        SReal compliance = k ? 1.0/k : 1e100;

        for(unsigned i=0, n = state->getMatrixSize(); i < n; i++) {
            matC.beginRow(i);
            matC.insertBack(i, i, compliance);
        }
        matC.compressedMatrix.finalize();
    }
    else matC.compressedMatrix.resize(0,0);

    // matK must be computed since it is used by MechanicalComputeStiffnessForceVisitor to compute the Stiffness forces
//    if( !this->isCompliance.getValue() || this->rayleighStiffness.getValue() )
//    {
        matK.resize(state->getMatrixSize(), state->getMatrixSize());

        if( k )
        {
            for(unsigned i=0, n = state->getMatrixSize(); i < n; i++) {
                matK.beginRow(i);
                matK.insertBack(i, i, -k);
            }

            matK.compressedMatrix.finalize();
        }

//    }
//    else matK.compressedMatrix.resize(0,0);


    // TODO if(this->isStiffness.getValue() && this->rayleighStiffness.getValue()) mettre rayleigh dans B mais attention à kfactor avec/sans rayleigh factor


	if( damping.getValue() > 0 ) {
        const SReal& d = damping.getValue();
		
		matB.resize(state->getMatrixSize(), state->getMatrixSize());
		
		for(unsigned i=0, n = state->getMatrixSize(); i < n; i++) {
            matB.beginRow(i);
            matB.insertBack(i, i, -d);
		}

		matB.compressedMatrix.finalize();
	}
    else matB.compressedMatrix.resize(0,0);
	
//    std::cerr<<SOFA_CLASS_METHOD<<matC<<" "<<matK<<std::endl;
}

template<class DataTypes>
SReal UniformStiffness<DataTypes>::getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x ) const
{
    const VecCoord& _x = x.getValue();
    unsigned int m = this->mstate->getMatrixBlockSize();

    const Real& k = stiffness.getValue();

    SReal e = 0;
    for( unsigned int i=0 ; i<_x.size() ; ++i )
    {
        for( unsigned int j=0 ; j<m ; ++j )
        {
            e += .5 * k * _x[i][j]*_x[i][j];
        }
    }
    return e;
}


template<class DataTypes>
const sofa::defaulttype::BaseMatrix* UniformStiffness<DataTypes>::getStiffnessMatrix(const core::MechanicalParams*)
{
    if( resizable.getValue() && (defaulttype::BaseMatrix::Index)this->mstate->getSize() != matC.rows() ) reinit();

    return &matC;
}


template<class DataTypes>
void UniformStiffness<DataTypes>::addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset )
{
    matK.addToBaseMatrix( matrix, kFact, offset );
}

template<class DataTypes>
void UniformStiffness<DataTypes>::addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal bFact, unsigned int &offset )
{
//	if( damping.getValue() > 0 ) // B is empty in that case
    {
		matB.addToBaseMatrix( matrix, bFact, offset );
	}
}

template<class DataTypes>
void UniformStiffness<DataTypes>::addForce(const core::MechanicalParams *, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /*v*/)
{
    if( resizable.getValue() &&  (defaulttype::BaseMatrix::Index)x.getValue().size() != matK.compressedMatrix.rows() ) reinit();

//    if( matK.compressedMatrix.nonZeros() )
        matK.addMult( f, x  );

//    cerr<<SOFA_CLASS_METHOD<< f << endl;
}

template<class DataTypes>
void UniformStiffness<DataTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    Real kfactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    if( kfactor )
    {
        matK.addMult( df, dx, kfactor );
    }

    if( damping.getValue() > 0 )
    {
        Real bfactor = (Real)mparams->bFactor();
        matB.addMult( df, dx, bfactor );
    }
}

template<class DataTypes>
void UniformStiffness<DataTypes>::addClambda(const core::MechanicalParams *, DataVecDeriv &res, const DataVecDeriv &lambda, SReal cfactor)
{
    matC.addMult( res, lambda, cfactor );
}


}
}
}
