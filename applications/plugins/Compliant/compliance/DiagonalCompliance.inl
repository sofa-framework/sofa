#include "DiagonalCompliance.h"
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
DiagonalCompliance<DataTypes>::DiagonalCompliance( core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , diagonal( initData(&diagonal, 
                         "compliance", 
                         "Compliance value diagonally applied to all the DOF."))
{
	this->isCompliance.setValue(true);
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
                matC.insertBack(row, row, diagonal.getValue()[i][j]);
    //            matC.add(row, row, diagonal.getValue()[i][j]);
                ++row;
            }
        }
        matC.compress();
    }
    else matC.compressedMatrix.resize(0,0);

    if( !this->isCompliance.getValue() || this->rayleighStiffness.getValue() )
    {
        matK.resize(state->getMatrixSize(), state->getMatrixSize());

        unsigned int row = 0;
        for(unsigned i = 0; i < n; ++i)
        {
            for(unsigned int j = 0; j < m; ++j)
            {
    //            matC.beginRow(row);
                matK.insertBack(row, row, -1.0/diagonal.getValue()[i][j]);
    //            matC.add(row, row, diagonal.getValue()[i][j]);
                ++row;
            }
        }
        matK.compress();
    }
    else matK.compressedMatrix.resize(0,0);
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
    matK.addToBaseMatrix( matrix, kFact, offset );
}


template<class DataTypes>
void DiagonalCompliance<DataTypes>::addForce(const core::MechanicalParams *, DataVecDeriv& _f, const DataVecCoord& _x, const DataVecDeriv& /*_v*/)
{
    matK.addMult( _f, _x  );

//    cerr<<"UniformCompliance<DataTypes>::addForce, f after = " << f << endl;
}

template<class DataTypes>
void DiagonalCompliance<DataTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv& _df,  const DataVecDeriv& _dx)
{
    Real kfactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    matK.addMult( _df, _dx, kfactor );
}


}
}
}
