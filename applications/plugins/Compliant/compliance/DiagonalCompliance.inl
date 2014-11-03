#include "DiagonalCompliance.h"
#include <iostream>

#include <utils/edit.h>

using std::cerr;
using std::endl;

namespace sofa
{
namespace component
{
namespace forcefield
{

template<class DataTypes>
DiagonalCompliance<DataTypes>::DiagonalCompliance( core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , diagonal( initData(&diagonal, 
                         "compliance", 
                         "Compliance value diagonally applied to all the DOF."))
    , damping( initData(&damping, "damping", "uniform viscous damping."))
{
	this->isCompliance.setValue(true);
	edit(damping)->push_back(0);
}

template<class DataTypes>
void DiagonalCompliance<DataTypes>::init()
{
    Inherit::init();
    if( this->getMState()==NULL ) serr<<"DiagonalCompliance<DataTypes>::init(), no mstate !" << sendl;
    reinit();
}

template<class DataTypes>
void DiagonalCompliance<DataTypes>::reinit()
{
    core::behavior::BaseMechanicalState* state = this->getContext()->getMechanicalState();
    assert(state);

//    cerr<<SOFA_CLASS_METHOD<<std::endl;

    unsigned int m = state->getMatrixBlockSize(), n = state->getSize();

    if( this->isCompliance.getValue() )
    {
        matC.resize(state->getMatrixSize(), state->getMatrixSize());

        unsigned int row = 0;
        for(unsigned i = 0; i < n; ++i)
        {
            for(unsigned int j = 0; j < m; ++j)
            {
    //            matC.beginRow(row);
                if( diagonal.getValue()[i][j] ) matC.insertBack(row, row, diagonal.getValue()[i][j]);
    //            matC.add(row, row, diagonal.getValue()[i][j]);
                ++row;
            }
        }
        matC.compress();
    }
    else matC.compressedMatrix.resize(0,0);

    // matK must be computed since it is used by MechanicalComputeComplianceForceVisitor to compute the compliance forces
//    if( !this->isCompliance.getValue() || this->rayleighStiffness.getValue() )
//    {
        matK.resize(state->getMatrixSize(), state->getMatrixSize());

        unsigned int row = 0;
        for(unsigned i = 0; i < n; ++i)
        {
            for(unsigned int j = 0; j < m; ++j)
            {
                // the stiffness df/dx is the opposite of the inverse compliance
                Real k = diagonal.getValue()[i][j] > std::numeric_limits<Real>::epsilon() ?
                        -1 / diagonal.getValue()[i][j] :
                        -1 / std::numeric_limits<Real>::epsilon();

                if( k ) matK.insertBack(row, row, k);
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
			
			const SReal d = damping.getValue()[index];
			
            matB.compressedMatrix.startVec(i);
            if( d ) matB.compressedMatrix.insertBack(i, i) = -d;
        }

        matB.compressedMatrix.finalize();
    }
    else matB.compressedMatrix.resize(0,0);
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
void DiagonalCompliance<DataTypes>::addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, double kFact, unsigned int &offset )
{
//    cerr<<SOFA_CLASS_METHOD<<std::endl;
    matK.addToBaseMatrix( matrix, kFact, offset );
}

template<class DataTypes>
void DiagonalCompliance<DataTypes>::addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, double bFact, unsigned int &offset )
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


}
}
}
