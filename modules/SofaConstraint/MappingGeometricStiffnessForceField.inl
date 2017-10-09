#ifndef SOFA_CONSTRAINT_MAPPINGGEOMTRICSTIFFNESSFORCEFIELD_INL
#define SOFA_CONSTRAINT_MAPPINGGEOMTRICSTIFFNESSFORCEFIELD_INL

#include "MappingGeometricStiffnessForceField.h"
#include <sofa/core/behavior/ForceField.inl>
#include <SofaBaseLinearSolver/BlocMatrixWriter.h>

namespace sofa
{
namespace constraint
{

template< class DataTypes> 
MappingGeometricStiffnessForceField<DataTypes>::MappingGeometricStiffnessForceField()
:l_mapping(initLink("mapping", "Path to the mapping instance whose geometric stiffness is assembled"))
{

}

template< class DataTypes>
MappingGeometricStiffnessForceField<DataTypes>::~MappingGeometricStiffnessForceField()
{

}

template< class DataTypes>
void MappingGeometricStiffnessForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& /*f*/, 
    const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/)
{

}

template< class DataTypes>
void MappingGeometricStiffnessForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams* mparams,
    DataVecDeriv& /*df*/, const DataVecDeriv& /*dx*/)
{
    mparams->kFactor();
}

template< class DataTypes>
void MappingGeometricStiffnessForceField<DataTypes>::addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
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
//void MappingGeometricStiffnessForceField<DataTypes>::addKToMatrixT(const sofa::core::MechanicalParams* mparams, MatrixWriter m)
//{
//
//}

}

}


#endif // SOFA_CONSTRAINT_MAPPINGGEOMTRICSTIFFNESSFORCEFIELD_INL
