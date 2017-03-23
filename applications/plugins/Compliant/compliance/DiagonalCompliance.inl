#include "DiagonalCompliance.h"
#include <iostream>

#include "../utils/edit.h"

using std::cerr;
using std::endl;

namespace sofa
{
namespace component
{
namespace forcefield
{

template<class DataTypes>
const typename DiagonalCompliance<DataTypes>::Real DiagonalCompliance<DataTypes>::s_complianceEpsilon = std::numeric_limits<typename DiagonalCompliance<DataTypes>::Real>::epsilon();

template<class DataTypes>
DiagonalCompliance<DataTypes>::DiagonalCompliance( core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , diagonal( initData(&diagonal, 
                         "compliance", 
                         "Compliance value diagonally applied to all the DOF."))
    , damping( initData(&damping, "damping", "viscous damping."))
{
	this->isCompliance.setValue(true);
    editOnly(damping)->push_back(0);
}

template<class DataTypes>
void DiagonalCompliance<DataTypes>::init()
{
    Inherit::init();
    if( this->getMState()==NULL ) serr<<"init(), no mstate !" << sendl;
    reinit();
}

template<class DataTypes>
void DiagonalCompliance<DataTypes>::reinit()
{
    core::behavior::BaseMechanicalState* state = this->getContext()->getMechanicalState();
    assert(state);

//    cerr<<SOFA_CLASS_METHOD<<std::endl;

    unsigned int m = state->getMatrixBlockSize(), n = state->getSize();
    VecDeriv const& diag = diagonal.getValue();

    if( this->isCompliance.getValue() )
    {
        matC.resize(state->getMatrixSize(), state->getMatrixSize());

        unsigned int row = 0;
        for(unsigned i = 0; i < n; ++i)
        {
            for(unsigned int j = 0; j < m; ++j)
            {
                const SReal& c = diag[i][j];
                matC.beginRow(row);
                if(c) matC.insertBack(row, row, c);

                ++row;
            }
        }
        matC.compress();
    }
    else matC.compressedMatrix.resize(0,0);

//    if( !this->isCompliance.getValue() || this->rayleighStiffness.getValue() )
//    {
        matK.resize(state->getMatrixSize(), state->getMatrixSize());

        unsigned int row = 0;
        for(unsigned i = 0; i < n; ++i)
        {
            for(unsigned int j = 0; j < m; ++j)
            {
                const SReal& c = diag[i][j];
                // the stiffness df/dx is the opposite of the inverse compliance
                Real k = c > std::numeric_limits<Real>::epsilon() ?
                        (c < 1 / std::numeric_limits<Real>::epsilon() ? -1 / c : 0 ) : // if the compliance is really large, let's consider the stiffness is null
                        -1 / std::numeric_limits<Real>::epsilon(); // if the compliance is too small, we have to take a huge stiffness in the numerical limits

                matK.beginRow(row);
                matK.insertBack(row, row, k);

                ++row;
            }
        }
        matK.compress();
//    }
//    else matK.compressedMatrix.resize(0,0);

		if( damping.getValue().size() > 1 || damping.getValue()[0] > 0 ) {
		
        matB.resize(state->getMatrixSize(), state->getMatrixSize());

        for(unsigned i=0, n = state->getMatrixSize(); i < n; i++) {
			const unsigned index = std::min<unsigned>(i, damping.getValue().size() - 1);
			
            const SReal& d = damping.getValue()[index];

            matB.beginRow(i);
            matB.insertBack(i, i, -d);
        }

        matB.compressedMatrix.finalize();
    }
    else matB.compressedMatrix.resize(0,0);

}

template<class DataTypes>
SReal DiagonalCompliance<DataTypes>::getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x ) const
{
    const VecCoord& _x = x.getValue();
    unsigned int m = this->mstate->getMatrixBlockSize();
    VecDeriv const& diag = diagonal.getValue();

    SReal e = 0;
    for( unsigned int i=0 ; i<_x.size() ; ++i )
    {
        for( unsigned int j=0 ; j<m ; ++j )
        {
            const Real& compliance = diag[i][j];
            Real k = compliance > s_complianceEpsilon ?
                    1. / compliance :
                    1. / s_complianceEpsilon;

            e += .5 * k * _x[i][j]*_x[i][j];
        }
    }
    return e;
}

//template<class DataTypes>
//void DiagonalCompliance<DataTypes>::setCompliance( Real c )
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
const sofa::defaulttype::BaseMatrix* DiagonalCompliance<DataTypes>::getComplianceMatrix(const core::MechanicalParams*)
{
    return &matC;
}

template<class DataTypes>
void DiagonalCompliance<DataTypes>::addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset )
{
//    cerr<<SOFA_CLASS_METHOD<<std::endl;
    matK.addToBaseMatrix( matrix, kFact, offset );
}

template<class DataTypes>
void DiagonalCompliance<DataTypes>::addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal bFact, unsigned int &offset )
{
    matB.addToBaseMatrix( matrix, bFact, offset );
}


template<class DataTypes>
void DiagonalCompliance<DataTypes>::addForce(const core::MechanicalParams *, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /*v*/)
{
//    if( matK.compressedMatrix.nonZeros() )
        matK.addMult( f, x );

//        cerr<<SOFA_CLASS_METHOD<<"f after = " << f << std::endl << x << std::endl << matK << endl;
}

template<class DataTypes>
void DiagonalCompliance<DataTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv& df,  const DataVecDeriv& dx)
{
    Real kfactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    if( kfactor )
    {
        matK.addMult( df, dx, kfactor );
    }

    if( damping.getValue().size() > 1 || damping.getValue()[0] > 0 ) {
        Real bfactor = (Real)mparams->bFactor();
        matB.addMult( df, dx, bfactor );
    }
}

template<class DataTypes>
void DiagonalCompliance<DataTypes>::addClambda(const core::MechanicalParams *, DataVecDeriv &res, const DataVecDeriv &lambda, SReal cfactor)
{
    matC.addMult( res, lambda, cfactor );
}

}
}
}
