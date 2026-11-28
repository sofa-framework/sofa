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
#include <sofa/component/mass/BaseFEMMassMatrix.h>
#include <sofa/core/behavior/BaseLocalMassMatrix.h>
#include <sofa/helper/IotaView.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::mass
{

template <class TDataTypes, class TSpatialTypes, class TElementType>
BaseFEMMassMatrix<TDataTypes, TSpatialTypes, TElementType>::BaseFEMMassMatrix()
    : d_lumping(initData(&d_lumping, "lumping","If true, the matrix is lumped, meaning the matrix is approximated to a diagonal matrix (summing all values of a line on the diagonal)"))
    , l_spatialState(initLink("spatialState", "Link to the state holding the spatial positions of the mesh"))
{
    this->addUpdateCallback("recomputeMatrix", {&this->d_lumping},
    [this](const sofa::core::DataTracker& )
    {
        assembleGlobalMatrix();
        return this->getComponentState();
    }, {});
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
void BaseFEMMassMatrix<TDataTypes, TSpatialTypes, TElementType>::init()
{
    TopologyAccessor::init();

    if (!this->isComponentStateInvalid())
    {
        core::behavior::Mass<TDataTypes>::init();
    }

    if (!this->isComponentStateInvalid())
    {
        validateNodalCoefficient();
    }

    if (!this->isComponentStateInvalid())
    {
        validateSpatialState();
    }

    if (!this->isComponentStateInvalid())
    {
        assembleGlobalMatrix();
    }
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
void BaseFEMMassMatrix<TDataTypes, TSpatialTypes, TElementType>::validateSpatialState()
{
    if (l_spatialState.empty())
    {
        if constexpr (std::is_same_v<DataTypes, SpatialTypes>)
        {
            l_spatialState.set(this->mstate.get());
        }
        else
        {
            msg_info() << "Link to a spatial state should be set to ensure right behavior. First "
                          "compatible component found in current context will be used.";
            l_spatialState.set(this->getContext()->template get<sofa::core::behavior::MechanicalState<SpatialTypes>>());
        }
    }

    if (l_spatialState == nullptr)
    {
        msg_error() << "No compatible spatial state (MechanicalState<" << SpatialTypes::Name()
                    << ">) found at path: " << this->l_spatialState.getLinkedPath()
                    << ", nor in current context: " << this->getContext()->name
                    << ". Please set the link `spatialState` explicitly.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
void BaseFEMMassMatrix<TDataTypes, TSpatialTypes, TElementType>::assembleGlobalMatrix()
{
    if (!this->l_topology || !this->l_spatialState || !this->mstate || !getNodalCoefficient())
        return;

    // the rest positions may not be available yet if the spatial state has not been initialized
    // (e.g. if it is located in a child node)
    const auto nbSpatialRestPositions = this->l_spatialState->readRestPositions().size();
    if (nbSpatialRestPositions != this->mstate->getSize())
    {
        msg_error() << "The rest positions of the spatial state (" << nbSpatialRestPositions
                    << " nodes) and the state of this component (" << this->mstate->getSize()
                    << " nodes) must have the same number of nodes. Make sure that the spatial "
                       "state is initialized before this component.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    sofa::type::vector<ElementMatrix> elementMatrices;

    // 1. compute element matrices
    computeElementMatrices(elements, elementMatrices);

    // 2. convert element matrices to dof matrices and store them for later use
    initializeGlobalMatrix(elements, elementMatrices);
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
void BaseFEMMassMatrix<TDataTypes, TSpatialTypes, TElementType>::computeElementMatrices(
    const auto& elements, sofa::type::vector<ElementMatrix>& elementMatrices)
{
    const auto nbElements = elements.size();
    elementMatrices.resize(nbElements);

    const NodalCoefficient* nodalCoefficient = getNodalCoefficient();
    sofa::helper::ReadAccessor nodalCoefficientAccessor { nodalCoefficient->d_property };
    auto restPositionAccessor = this->l_spatialState->readRestPositions();

    const bool lumping = d_lumping.getValue();

    SCOPED_TIMER("elementMatrices");
    helper::IotaView indices{static_cast<decltype(nbElements)>(0ul), nbElements};
    std::for_each(
        indices.begin(), indices.end(),
        [&](const auto elementId)
        {
            const auto& element = elements[elementId];
            auto& elementMatrix = elementMatrices[elementId];

            std::array<Real, NumberOfNodesInElement> nodeCoefficientInElement;
            for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
            {
                nodeCoefficientInElement[i] =
                    nodalCoefficient->getNodeProperty(element[i], nodalCoefficientAccessor);
            }

            std::array<GeometryCoord, NumberOfNodesInElement> nodeCoordinatesInElement;
            for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
            {
                nodeCoordinatesInElement[i] = restPositionAccessor[element[i]];
            }

            for (const auto& [quadraturePoint, weight] : FiniteElement::quadraturePoints())
            {
                // gradient of shape functions in the reference element evaluated at the quadrature
                // point
                const sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, GeometryReal>
                    dN_dq_ref = FiniteElement::gradientShapeFunctions(quadraturePoint);

                // jacobian of the mapping from the reference space to the physical space, evaluated
                // at the quadrature point
                sofa::type::Mat<spatial_dimensions, TopologicalDimension, GeometryReal>
                    jacobian = FiniteElement::Helper::jacobianFromReferenceToPhysical(
                        nodeCoordinatesInElement, dN_dq_ref);

                const auto detJ = static_cast<Real>(sofa::type::absGeneralizedDeterminant(jacobian));

                // shape functions in the reference element evaluated at the quadrature point
                const auto N = FiniteElement::shapeFunctions(quadraturePoint);

                const auto coefficient =
                    FiniteElement::Helper::evaluateValueInElement(nodeCoefficientInElement, N);

                const sofa::type::Vec<NumberOfNodesInElement, Real> N_field(N);
                const auto NT_N = sofa::type::dyad(N_field, N_field);

                elementMatrix += (static_cast<Real>(weight) * coefficient * detJ) * NT_N;
            }

            if (lumping)
            {
                ElementMatrix lumped{};
                for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
                {
                    Real rowSum = 0;
                    for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
                        rowSum += elementMatrix(i, j);
                    lumped(i, i) = rowSum;
                }
                elementMatrix = lumped;
            }
        });
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
void BaseFEMMassMatrix<TDataTypes, TSpatialTypes, TElementType>::initializeGlobalMatrix(
    const auto& elements, const sofa::type::vector<ElementMatrix>& elementMatrices)
{
    SCOPED_TIMER("globalMatrix");

    const auto nbElements = elements.size();

    m_globalMatrix.clear();

    const auto matrixSize = this->mstate->getSize();
    m_globalMatrix.resize(matrixSize, matrixSize);

    helper::IotaView indices{static_cast<decltype(nbElements)>(0ul), nbElements};
    std::for_each(indices.begin(), indices.end(),
                  [&](const auto elementId)
                  {
                      const auto& element = elements[elementId];
                      const auto& elementMatrix = elementMatrices[elementId];

                      for (sofa::Size i = 0; i < NumberOfNodesInElement; ++i)
                      {
                          const auto node_i = element[i];
                          for (sofa::Size j = 0; j < NumberOfNodesInElement; ++j)
                          {
                              const auto node_j = element[j];

                              const auto& value = elementMatrix(i, j);
                              constexpr Real tolerance { 1e-12 };
                              if (std::abs(value) > tolerance)
                              {
                                  m_globalMatrix.add(node_i, node_j, value);
                              }
                          }
                      }
                  });

    m_globalMatrix.compress();
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
void BaseFEMMassMatrix<TDataTypes, TSpatialTypes, TElementType>::buildMassMatrix(
    sofa::core::behavior::MassMatrixAccumulator* matrices)
{
    if (this->isComponentStateInvalid())
        return;

    for (std::size_t xi = 0; xi < m_globalMatrix.rowIndex.size(); ++xi)
    {
        const auto rowId = m_globalMatrix.rowIndex[xi];
        typename GlobalMatrixType::Range rowRange(m_globalMatrix.rowBegin[xi], m_globalMatrix.rowBegin[xi + 1]);
        for (typename GlobalMatrixType::Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
        {
            const auto columnId = m_globalMatrix.colsIndex[xj];
            const auto& value = m_globalMatrix.colsValue[xj];

            for (typename GlobalMatrixType::Index d = 0; d < static_cast<typename GlobalMatrixType::Index>(field_dimensions); ++d)
            {
                matrices->add(rowId * field_dimensions + d, columnId * field_dimensions + d, value);
            }
        }
    }
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
void BaseFEMMassMatrix<TDataTypes, TSpatialTypes, TElementType>::addMDx(const core::MechanicalParams* mparams,
                                                      DataVecDeriv_t<DataTypes>& f,
                                                      const DataVecDeriv_t<DataTypes>& dx,
                                                      SReal factor)
{
    SOFA_UNUSED(mparams);

    if (this->isComponentStateInvalid())
        return;

    auto result = sofa::helper::getWriteAccessor(f);
    const auto dxAccessor = sofa::helper::getReadAccessor(dx);

    for (std::size_t xi = 0; xi < m_globalMatrix.rowIndex.size(); ++xi)
    {
        const auto rowId = m_globalMatrix.rowIndex[xi];
        typename GlobalMatrixType::Range rowRange(m_globalMatrix.rowBegin[xi], m_globalMatrix.rowBegin[xi + 1]);
        for (typename GlobalMatrixType::Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
        {
            const auto columnId = m_globalMatrix.colsIndex[xj];
            const auto& value = m_globalMatrix.colsValue[xj];

            result[rowId] += (factor * value) * dxAccessor[columnId];
        }
    }
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
void BaseFEMMassMatrix<TDataTypes, TSpatialTypes, TElementType>::doSolveLinearSystem(const core::MechanicalParams* mparams,
                                                        DataVecDeriv_t<DataTypes>& a,
                                                        const DataVecDeriv_t<DataTypes>& f)
{
    SOFA_UNUSED(mparams);

    if (this->isComponentStateInvalid())
        return;

    if (!d_lumping.getValue())
    {
        msg_error() << "the method 'solveLinearSystem' can't be used with this component as this "
                        "SPARSE matrix can't be inversed easily. Enable 'lumping' instead.";
        return;
    }

    auto accAccessor = sofa::helper::getWriteAccessor(a);
    const auto fAccessor = sofa::helper::getReadAccessor(f);

    for (std::size_t xi = 0; xi < m_globalMatrix.rowIndex.size(); ++xi)
    {
        const auto rowId = m_globalMatrix.rowIndex[xi];
        typename GlobalMatrixType::Range rowRange(
            m_globalMatrix.rowBegin[xi], m_globalMatrix.rowBegin[xi + 1]);

        for (typename GlobalMatrixType::Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
        {
            const auto columnId = m_globalMatrix.colsIndex[xj];
            if (columnId == rowId) // diagonal is guaranteed to exist when lumped
            {
                const auto& value = m_globalMatrix.colsValue[xj];
                accAccessor[rowId] = fAccessor[rowId] / value;
            }
        }
    }
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
SReal BaseFEMMassMatrix<TDataTypes, TSpatialTypes, TElementType>::getKineticEnergy(
    const core::MechanicalParams* mparams,
    const DataVecDeriv_t<DataTypes>& v) const
{
    SOFA_UNUSED(mparams);

    if (this->isComponentStateInvalid())
        return 0_sreal;

    SReal kineticEnergy = 0.0;
    auto vAccessor = sofa::helper::getReadAccessor(v);

    for (std::size_t xi = 0; xi < m_globalMatrix.rowIndex.size(); ++xi)
    {
        const auto rowId = m_globalMatrix.rowIndex[xi];
        typename GlobalMatrixType::Range rowRange(m_globalMatrix.rowBegin[xi], m_globalMatrix.rowBegin[xi + 1]);
        for (typename GlobalMatrixType::Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
        {
            const auto columnId = m_globalMatrix.colsIndex[xj];
            const auto& value = m_globalMatrix.colsValue[xj];

            kineticEnergy += sofa::type::dot(vAccessor[rowId], vAccessor[columnId] * value);
        }
    }

    return kineticEnergy / 2_sreal;
}

}  // namespace sofa::component::mass
