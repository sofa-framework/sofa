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
#include <sofa/component/mapping/mappedmatrix/MappingGeometricStiffnessForceField.h>
#ifndef SOFA_BUILD_SOFA_COMPONENT_MAPPING_MAPPEDMATRIX
SOFA_DEPRECATED_HEADER_NOT_REPLACED("v23.06", "v23.12")
#endif

#include <sofa/core/behavior/ForceField.inl>

namespace sofa::component::mapping::mappedmatrix
{

template< class DataTypes> 
MappingGeometricStiffnessForceField<DataTypes>::MappingGeometricStiffnessForceField()
: d_yesIKnowMatrixMappingIsSupportedAutomatically(initData(&d_yesIKnowMatrixMappingIsSupportedAutomatically, false, "yesIKnowMatrixMappingIsSupportedAutomatically", "If true the component is activated, otherwise it is deactivated.\nThis Data is used to explicitly state that the component must be used even though matrix mapping is now supported automatically, without MappingGeometricStiffnessForceField."))
, l_mapping(initLink("mapping", "Path to the mapping instance whose geometric stiffness is assembled"))
{

}

template< class DataTypes>
MappingGeometricStiffnessForceField<DataTypes>::~MappingGeometricStiffnessForceField()
{

}

template <class DataTypes>
void MappingGeometricStiffnessForceField<DataTypes>::init()
{
    Inherit::init();

    if (!d_yesIKnowMatrixMappingIsSupportedAutomatically.getValue())
    {
        msg_error() << "This component is deprecated and deactivated because geometric stiffness is now supported automatically";
        this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
    }
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
    if(this->d_componentState.getValue() !=core::objectmodel::ComponentState::Valid)
    {
        return ;
    }

    SReal kFact = (SReal)mparams->kFactor();
    if (kFact == 0)
    {
        return;
    }

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::linearalgebra::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;

    const sofa::linearalgebra::BaseMatrix* mappingK = l_mapping->getK();

    for (sofa::linearalgebra::BaseMatrix::Index i = 0; i < mappingK->rowSize(); ++i)
    {
        for (sofa::linearalgebra::BaseMatrix::Index j = 0; j < mappingK->colSize(); ++j)
        {
            mat->add(offset + i, offset + j, mappingK->element(i, j)*kFact);
        }
    }
}

template <class DataTypes>
void MappingGeometricStiffnessForceField<DataTypes>::buildStiffnessMatrix(
    core::behavior::StiffnessMatrix* matrix)
{
    if(this->d_componentState.getValue() !=core::objectmodel::ComponentState::Valid)
    {
        return ;
    }

    const sofa::linearalgebra::BaseMatrix* mappingK = l_mapping->getK();

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    for (sofa::linearalgebra::BaseMatrix::Index i = 0; i < mappingK->rowSize(); ++i)
    {
        for (sofa::linearalgebra::BaseMatrix::Index j = 0; j < mappingK->colSize(); ++j)
        {
            dfdx(i, j) += mappingK->element(i, j);
        }
    }
}
} // namespace sofa::component::mapping::mappedmatrix
