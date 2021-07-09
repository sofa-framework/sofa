/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <SofaConstraint/MappingGeometricStiffnessForceField.h>

#include <sofa/core/behavior/ForceField.inl>
#include <SofaBaseLinearSolver/BlocMatrixWriter.h>

namespace sofa::constraint
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
    SReal kFact = (SReal)mparams->kFactor();
    if (kFact == 0)
    {
        return;
    }

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;

    const sofa::defaulttype::BaseMatrix* mappingK = l_mapping->getK();

    for (sofa::defaulttype::BaseMatrix::Index i = 0; i < mappingK->rowSize(); ++i)
    {
        for (sofa::defaulttype::BaseMatrix::Index j = 0; j < mappingK->colSize(); ++j)
        {
            mat->add(offset + i, offset + j, mappingK->element(i, j)*kFact);
        }
    }
}

} // namespace sofa::constraint
