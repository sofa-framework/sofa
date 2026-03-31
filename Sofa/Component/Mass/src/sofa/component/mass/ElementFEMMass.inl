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
#include <sofa/component/mass/ElementFEMMass.h>
#include <sofa/core/behavior/BaseLocalMassMatrix.h>
#include <sofa/helper/IotaView.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::mass
{

template <class TDataTypes, class TElementType>
ElementFEMMass<TDataTypes, TElementType>::ElementFEMMass()
    : l_nodalMassDensity(initLink("nodalMassDensity", "Link to nodal mass density"))
{
}


template <class TDataTypes, class TElementType>
void ElementFEMMass<TDataTypes, TElementType>::init()
{
    TopologyAccessor::init();

    if (!this->isComponentStateInvalid())
    {
        core::behavior::Mass<TDataTypes>::init();
    }

    if (!this->isComponentStateInvalid())
    {
        validateNodalMassDensity();
    }

    if (!this->isComponentStateInvalid())
    {
        elementFEMMass_init();
    }
}

template <class TDataTypes, class TElementType>
void ElementFEMMass<TDataTypes, TElementType>::validateNodalMassDensity()
{
    if (l_nodalMassDensity.empty())
    {
        msg_info() << "Link to a nodal mass density should be set to ensure right behavior. First "
                      "component found in current context will be used.";
        l_nodalMassDensity.set(this->getContext()->get<NodalMassDensity>());
    }

    if (l_nodalMassDensity == nullptr)
    {
        msg_error() << "No nodal mass density component found at path: " << this->l_nodalMassDensity.getLinkedPath()
                    << ", nor in current context: " << this->getContext()->name << ".";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}


template <class TDataTypes, class TElementType>
void ElementFEMMass<TDataTypes, TElementType>::elementFEMMass_init()
{
    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    const auto nbElements = elements.size();

    sofa::type::vector<ElementMassMatrix> elementMassMatrices;

    //1. compute element mass matrix
    calculateElementMassMatrix(elements, elementMassMatrices);

    // 2. convert element matrices to dof matrices and store them for later use
    initializeGlobalMassMatrix(elements, elementMassMatrices);
}

template <class TDataTypes, class TElementType>
void ElementFEMMass<TDataTypes, TElementType>::calculateElementMassMatrix(
    const auto& elements, sofa::type::vector<ElementMassMatrix> &elementMassMatrices)
{
    const auto nbElements = elements.size();
    elementMassMatrices.resize(nbElements);

    sofa::helper::ReadAccessor nodalMassDensityAccessor { l_nodalMassDensity->d_property };
    auto restPositionAccessor = this->mstate->readRestPositions();

    SCOPED_TIMER("elementMassMatrix");
    helper::IotaView indices{static_cast<decltype(nbElements)>(0ul), nbElements};
    std::for_each(
        indices.begin(), indices.end(),
        [&](const auto elementId)
        {
            const auto& element = elements[elementId];
            auto& elementMassMatrix = elementMassMatrices[elementId];

            std::array<Real_t<DataTypes>, NumberOfNodesInElement> nodeDensityInElement;
            for (std::size_t i = 0; i < NumberOfNodesInElement; ++i)
            {
                nodeDensityInElement[i] =
                    l_nodalMassDensity->getNodeProperty(element[i], nodalMassDensityAccessor);
            }

            std::array<Coord_t<DataTypes>, NumberOfNodesInElement> nodeCoordinatesInElement;
            for (std::size_t i = 0; i < NumberOfNodesInElement; ++i)
            {
                nodeCoordinatesInElement[i] = restPositionAccessor[element[i]];
            }

            for (const auto& [quadraturePoint, weight] : FiniteElement::quadraturePoints())
            {
                // gradient of shape functions in the reference element evaluated at the quadrature
                // point
                const sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real_t<DataTypes>>
                    dN_dq_ref = FiniteElement::gradientShapeFunctions(quadraturePoint);

                // jacobian of the mapping from the reference space to the physical space, evaluated
                // at the quadrature point
                sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real_t<DataTypes>>
                    jacobian = FiniteElement::Helper::jacobianFromReferenceToPhysical(
                        nodeCoordinatesInElement, dN_dq_ref);

                const auto detJ = sofa::type::absGeneralizedDeterminant(jacobian);

                // shape functions in the reference element evaluated at the quadrature point
                const auto N = FiniteElement::shapeFunctions(quadraturePoint);

                const auto density =
                    FiniteElement::Helper::evaluateValueInElement(nodeDensityInElement, N);

                const auto NT_N = sofa::type::dyad(N, N);

                elementMassMatrix += (weight * density * detJ) * NT_N;
            }
        });
}

template <class TDataTypes, class TElementType>
void ElementFEMMass<TDataTypes, TElementType>::initializeGlobalMassMatrix(
    const auto& elements, const sofa::type::vector<ElementMassMatrix>& elementMassMatrices)
{
    SCOPED_TIMER("elementMassMatrix");

    const auto nbElements = elements.size();

    m_globalMassMatrix.clear();

    const auto matrixSize = this->mstate->getSize();
    m_globalMassMatrix.resize(matrixSize, matrixSize);

    helper::IotaView indices{static_cast<decltype(nbElements)>(0ul), nbElements};
    std::for_each(indices.begin(), indices.end(),
                  [&](const auto elementId)
                  {
                      const auto& element = elements[elementId];
                      auto& elementMassMatrix = elementMassMatrices[elementId];

                      for (std::size_t i = 0; i < NumberOfNodesInElement; ++i)
                      {
                          const auto node_i = element[i];
                          for (std::size_t j = 0; j < NumberOfNodesInElement; ++j)
                          {
                              const auto node_j = element[j];
                              m_globalMassMatrix.add(node_i, node_j, elementMassMatrix(i, j));
                          }
                      }
                  });

    m_globalMassMatrix.compress();
}

template <class TDataTypes, class TElementType>
void ElementFEMMass<TDataTypes, TElementType>::addForce(const core::MechanicalParams* mparams,
                                                        sofa::DataVecDeriv_t<DataTypes>& f,
                                                        const sofa::DataVecCoord_t<DataTypes>& x,
                                                        const sofa::DataVecDeriv_t<DataTypes>& v)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(x);
    SOFA_UNUSED(v);

    auto forceAccessor = sofa::helper::getWriteAccessor(f);

    const auto g = getContext()->getGravity();
    Deriv_t<DataTypes> theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2] );

    for (Index xi = 0; xi < (Index)m_globalMassMatrix.rowIndex.size(); ++xi)
    {
        const auto rowId = m_globalMassMatrix.rowIndex[xi];
        typename GlobalMassMatrixType::Range rowRange(m_globalMassMatrix.rowBegin[xi], m_globalMassMatrix.rowBegin[xi + 1]);
        for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
        {
            const auto columnId = m_globalMassMatrix.colsIndex[xj];
            const auto& value = m_globalMassMatrix.colsValue[xj];

            const auto force = value * theGravity;
            forceAccessor[rowId] += force;
        }
    }
}

template <class TDataTypes, class TElementType>
void ElementFEMMass<TDataTypes, TElementType>::buildMassMatrix(
    sofa::core::behavior::MassMatrixAccumulator* matrices)
{
    for (Index xi = 0; xi < (Index)m_globalMassMatrix.rowIndex.size(); ++xi)
    {
        const auto rowId = m_globalMassMatrix.rowIndex[xi];
        typename GlobalMassMatrixType::Range rowRange(m_globalMassMatrix.rowBegin[xi], m_globalMassMatrix.rowBegin[xi + 1]);
        for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
        {
            const auto columnId = m_globalMassMatrix.colsIndex[xj];
            const auto& value = m_globalMassMatrix.colsValue[xj];

            for (std::size_t d = 0; d < spatial_dimensions; ++d)
            {
                matrices->add(rowId * spatial_dimensions + d, columnId * spatial_dimensions +d, value);
            }
        }
    }
}

template <class TDataTypes, class TElementType>
void ElementFEMMass<TDataTypes, TElementType>::addMDx(const core::MechanicalParams*,
                                                      DataVecDeriv_t<DataTypes>& f,
                                                      const DataVecDeriv_t<DataTypes>& dx,
                                                      SReal factor)
{
    auto result = sofa::helper::getWriteAccessor(f);
    const auto dxAccessor = sofa::helper::getReadAccessor(dx);

    for (Index xi = 0; xi < (Index)m_globalMassMatrix.rowIndex.size(); ++xi)
    {
        const auto rowId = m_globalMassMatrix.rowIndex[xi];
        typename GlobalMassMatrixType::Range rowRange(m_globalMassMatrix.rowBegin[xi], m_globalMassMatrix.rowBegin[xi + 1]);
        for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
        {
            const auto columnId = m_globalMassMatrix.colsIndex[xj];
            const auto& value = m_globalMassMatrix.colsValue[xj];

            result[rowId] += (factor * value) * dxAccessor[columnId];
        }
    }
}

}  // namespace sofa::component::mass
