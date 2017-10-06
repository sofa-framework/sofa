#ifndef ISPHYSICS_MECHANICS_GEOMETRICSTIFFNESSFORCEFIELD_INL
#define ISPHYSICS_MECHANICS_GEOMETRICSTIFFNESSFORCEFIELD_INL

#include "GeometricStiffnessForceField.h"
#include <sofa/core/behavior/ForceField.inl>
#include <SofaBaseLinearSolver/BlocMatrixWriter.h>

namespace isphysics
{
namespace mechanics
{

template< class DataTypes> 
GeometricStiffnessForceField<DataTypes>::GeometricStiffnessForceField()
:l_mapping(initLink("mapping", "Path to the mapping instance whose geometric stiffness is assembled"))
{

}

template< class DataTypes>
GeometricStiffnessForceField<DataTypes>::~GeometricStiffnessForceField()
{

}

template< class DataTypes>
void GeometricStiffnessForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& /*f*/, 
    const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/)
{

}

template< class DataTypes>
void GeometricStiffnessForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams* mparams,
    DataVecDeriv& /*df*/, const DataVecDeriv& /*dx*/)
{
    mparams->kFactor();
}

template< class DataTypes>
void GeometricStiffnessForceField<DataTypes>::addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SReal kFact = (Real)mparams->kFactor();
    if (kFact == 0)
    {
        return;
    }

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;

    const sofa::defaulttype::BaseMatrix* mappingK = l_mapping->getK();

    for (int i = 0; i < mappingK->rowSize(); ++i)
    {
        for (int j = 0; j < mappingK->colSize(); ++j)
        {
            mat->add(offset + i, offset + j, mappingK->element(i, j)*kFact);
        }
    }

    //typedef typename DataTypes::Deriv TBloc;
    //sofa::component::linearsolver::BlocMatrixWriter< TBloc > writer;
    //writer.addKToMatrix(this, mparams, matrix->getMatrix(this->getMState()));
}

//template<class MatrixWriter>
//template<class DataTypes>
//void GeometricStiffnessForceField<DataTypes>::addKToMatrixT(const sofa::core::MechanicalParams* mparams, MatrixWriter m)
//{
//
//}

}

}


#endif // ISPHYSICS_MECHANICS_GEOMETRICSTIFFNESSFORCEFIELD_INL
