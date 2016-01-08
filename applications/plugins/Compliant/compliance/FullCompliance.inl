#include "FullCompliance.h"
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
const typename FullCompliance<DataTypes>::Real FullCompliance<DataTypes>::s_complianceEpsilon = std::numeric_limits<typename FullCompliance<DataTypes>::Real>::epsilon();

template<class DataTypes>
FullCompliance<DataTypes>::FullCompliance( core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , matC( initData(&matC, "C", "Compliance Matrix (PSD)") )
    , matK( initData(&matK, "K", "Stiffness Matrix (PSD)") )
    , damping( initData(&damping, "damping", "uniform viscous damping."))
{
    this->isCompliance.setValue(true);
    editOnly(damping)->push_back(0);
}

template<class DataTypes>
void FullCompliance<DataTypes>::init()
{
    Inherit::init();
    if( this->getMState()==NULL ) serr<<"init(), no mstate !" << sendl;
    reinit();
}

template<class DataTypes>
void FullCompliance<DataTypes>::reinit()
{
    core::behavior::BaseMechanicalState* state = this->getContext()->getMechanicalState();

    // check sizes
    bool CisOk = ( matC.getValue().rows() != matC.getValue().cols() || (unsigned)matC.getValue().rows()!=state->getMatrixSize() )?false:true;
    bool KisOk = ( matK.getValue().rows() != matK.getValue().cols() || (unsigned)matK.getValue().rows()!=state->getMatrixSize() )?false:true;

    if( this->isCompliance.getValue() )
    {
        if( !CisOk ) // need to compute C ?
        {
            block_matrix_type& C = *matC.beginWriteOnly();
            C.resize(state->getMatrixSize(), state->getMatrixSize()); // set to 0
            if( KisOk )
            {
                // compute from provided K
                Eigen::SparseMatrix<Real> Kinv;
                const Eigen::SparseMatrix<Real> K = matK.getValue().compressedMatrix;
                if (invertMatrix(Kinv,K)) C.compressedMatrix=Kinv;
            }
            matC.endEdit();
        }
    }

    if( !KisOk ) // need to compute K ?
    {
        block_matrix_type& K = *matK.beginWriteOnly();
        K.resize(state->getMatrixSize(), state->getMatrixSize());
        if( !CisOk )
        {
            // set to Id
            K.compressedMatrix.setIdentity();
        }
        else
        {
            // compute from provided C
            Eigen::SparseMatrix<Real> Cinv;
            const Eigen::SparseMatrix<Real> C = matC.getValue().compressedMatrix;
            if(invertMatrix(Cinv,C)) K.compressedMatrix=Cinv;
            else  K.compressedMatrix.setIdentity();
        }
        matK.endEdit();
    }

//    std::cout<<matK.getValue()<<std::endl;
//    std::cout<<matC.getValue()<<std::endl;

    if( damping.getValue().size() > 1 || damping.getValue()[0] > 0 )
    {
        matB.resize(state->getMatrixSize(), state->getMatrixSize());
        for(unsigned i=0, n = state->getMatrixSize(); i < n; i++)
        {
            const unsigned index = std::min<unsigned>(i, damping.getValue().size() - 1);
            const SReal d = damping.getValue()[index];
            matB.beginRow(i);
            matB.insertBack(i, i, -d);
        }
        matB.compressedMatrix.finalize();
    }
    else matB.compressedMatrix.resize(0,0);
}

template<class DataTypes>
bool FullCompliance<DataTypes>::invertMatrix(Eigen::SparseMatrix<Real>& Minv, const Eigen::SparseMatrix<Real>& M )
{
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<Real> > chol;
    chol.compute(M);
    if (chol.info() != Eigen::Success)  return false;
    Eigen::SparseMatrix<Real> I(M.cols(),M.rows());
    I.setIdentity();
    Minv = chol.solve(I);
    if (chol.info() != Eigen::Success)  return false;
    return true;
}

template<class DataTypes>
SReal FullCompliance<DataTypes>::getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x ) const
{
    const VecCoord& _x = x.getValue();

    SReal e = 0;
    VecCoord Kx;
    matK.getValue().mult( Kx, _x );
    for( unsigned int i=0 ; i<_x.size() ; i++ ) e += _x[i] * Kx[i];
    return e/2;
}


template<class DataTypes>
const sofa::defaulttype::BaseMatrix* FullCompliance<DataTypes>::getComplianceMatrix(const core::MechanicalParams*)
{
    return &matC.getValue();
}

template<class DataTypes>
void FullCompliance<DataTypes>::addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset )
{
    //    cerr<<SOFA_CLASS_METHOD<<std::endl;
    matK.getValue().addToBaseMatrix( matrix, -kFact, offset );
}

template<class DataTypes>
void FullCompliance<DataTypes>::addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal bFact, unsigned int &offset )
{
    matB.addToBaseMatrix( matrix, bFact, offset );
}


template<class DataTypes>
void FullCompliance<DataTypes>::addForce(const core::MechanicalParams *, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /*v*/)
{
    matK.getValue().addMult( f, x, -1.0 );
}

template<class DataTypes>
void FullCompliance<DataTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv& df,  const DataVecDeriv& dx)
{
    Real kfactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    if( kfactor )
    {
        matK.getValue().addMult( df, dx, -kfactor );
    }

    if( damping.getValue().size() > 1 || damping.getValue()[0] > 0 ) {
        Real bfactor = (Real)mparams->bFactor();
        matB.addMult( df, dx, bfactor );
    }
}


}
}
}
