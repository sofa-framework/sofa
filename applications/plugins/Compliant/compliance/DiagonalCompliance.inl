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
const typename DiagonalCompliance<DataTypes>::Real DiagonalCompliance<DataTypes>::s_complianceEpsilon = std::numeric_limits<typename DiagonalCompliance<DataTypes>::Real>::epsilon();

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
//    cerr<<SOFA_CLASS_METHOD<<std::endl;

    unsigned int m = this->mstate->getMatrixBlockSize(), n = this->mstate->getSize(), matrixsize = this->mstate->getMatrixSize();

    if( this->isCompliance.getValue() )
    {
        matC.resize(matrixsize,matrixsize);

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
    else matC.resize(0,0);

    // the stiffness matrix needs to be computed for NewtonSolver, but I need to check why
//    if( !this->isCompliance.getValue() || this->rayleighStiffness.getValue() )
    {
        matK.resize(matrixsize,matrixsize);

        unsigned int row = 0;
        for(unsigned i = 0; i < n; ++i)
        {
            for(unsigned int j = 0; j < m; ++j)
            {
                // the stiffness df/dx is the opposite of the inverse compliance
                Real k = diagonal.getValue()[i][j] > s_complianceEpsilon ?
                        -1. / diagonal.getValue()[i][j] :
                        -1. / s_complianceEpsilon;

                if( k ) matK.insertBack(row, row, k);
                ++row;
            }
        }
        matK.compress();
    }
//    else matK.resize(0,0);

		if( damping.getValue().size() > 1 || damping.getValue()[0] > 0 ) {
		
        matB.resize(matrixsize,matrixsize);

        for(unsigned i=0; i < matrixsize; i++) {
			const unsigned index = std::min<unsigned>(i, damping.getValue().size() - 1);
			
			const SReal d = damping.getValue()[index];
			
            matB.compressedMatrix.startVec(i);
            if( d ) matB.compressedMatrix.insertBack(i, i) = -d;
        }

        matB.compressedMatrix.finalize();
    }
    else matB.resize(0,0);
}


template<class DataTypes>
double DiagonalCompliance<DataTypes>::getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x ) const
{
    const VecCoord& _x = x.getValue();
    unsigned int m = this->mstate->getMatrixBlockSize();

    double e = 0;
    for( unsigned int i=0 ; i<_x.size() ; ++i )
    {
        for( unsigned int j=0 ; j<m ; ++j )
        {
            Real compliance = diagonal.getValue()[i][j];
            Real k = compliance > s_complianceEpsilon ?
                    1. / compliance :
                    1. / s_complianceEpsilon;

            e += .5 * k * _x[i][j]*_x[i][j];
        }
    }
    return e;
}


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
