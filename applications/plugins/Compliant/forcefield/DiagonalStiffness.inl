#include "DiagonalStiffness.h"
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
DiagonalStiffness<DataTypes>::DiagonalStiffness( core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , diagonal( initData(&diagonal, 
                         "stiffness",
                         "stiffness value diagonally applied to all the DOF."))
    , damping( initData(&damping, "damping", "viscous damping."))
{
    editOnly(damping)->push_back(0);
}

template<class DataTypes>
void DiagonalStiffness<DataTypes>::init()
{
    Inherit::init();
    if( this->getMState()==NULL ) serr<<"init(), no mstate !" << sendl;
    reinit();
}

template<class DataTypes>
void DiagonalStiffness<DataTypes>::reinit()
{
    core::behavior::BaseMechanicalState* state = this->getContext()->getMechanicalState();
    assert(state);


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
                const SReal& k = diag[i][j];
                SReal c = k ? 1.0/k : 1e100;
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
                const SReal& k = diag[i][j];
                matK.beginRow(row);
                if(k) matK.insertBack(row, row, -k);

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
            if(d) matB.insertBack(i, i, -d);
        }

        matB.compressedMatrix.finalize();
    }
    else matB.compressedMatrix.resize(0,0);

}

template<class DataTypes>
SReal DiagonalStiffness<DataTypes>::getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x ) const
{
    const VecCoord& _x = x.getValue();
    unsigned int m = this->mstate->getMatrixBlockSize();
    VecDeriv const& diag = diagonal.getValue();

    SReal e = 0;
    for( unsigned int i=0 ; i<_x.size() ; ++i )
    {
        for( unsigned int j=0 ; j<m ; ++j )
        {
            e += .5 * diag[i][j] * _x[i][j]*_x[i][j];
        }
    }
    return e;
}

template<class DataTypes>
const sofa::defaulttype::BaseMatrix* DiagonalStiffness<DataTypes>::getStiffnessMatrix(const core::MechanicalParams*)
{
    return &matC;
}

template<class DataTypes>
void DiagonalStiffness<DataTypes>::addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset )
{
//    cerr<<SOFA_CLASS_METHOD<<std::endl;
    matK.addToBaseMatrix( matrix, kFact, offset );
}

template<class DataTypes>
void DiagonalStiffness<DataTypes>::addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal bFact, unsigned int &offset )
{
    matB.addToBaseMatrix( matrix, bFact, offset );
}


template<class DataTypes>
void DiagonalStiffness<DataTypes>::addForce(const core::MechanicalParams *, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /*v*/)
{
//    if( matK.compressedMatrix.nonZeros() )
        matK.addMult( f, x );

//        cerr<<SOFA_CLASS_METHOD<<"f after = " << f << std::endl << x << std::endl << matK << endl;
}

template<class DataTypes>
void DiagonalStiffness<DataTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv& df,  const DataVecDeriv& dx)
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
void DiagonalStiffness<DataTypes>::addClambda(const core::MechanicalParams *, DataVecDeriv &res, const DataVecDeriv &lambda, SReal cfactor)
{
    matC.addMult( res, lambda, cfactor );
}

}
}
}
