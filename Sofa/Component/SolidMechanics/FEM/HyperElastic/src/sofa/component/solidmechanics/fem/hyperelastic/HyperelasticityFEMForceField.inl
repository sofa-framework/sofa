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

#include <sofa/component/solidmechanics/fem/hyperelastic/HyperelasticityFEMForceField.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/impl/Strain.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/VectorTools.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.inl>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class TDataTypes, class TElementType>
HyperelasticityFEMForceField<TDataTypes, TElementType>::HyperelasticityFEMForceField()
    : l_material(initLink("material", "Link to the material containing the constitutive law"))
{
}

template <class TDataTypes, class TElementType>
void HyperelasticityFEMForceField<TDataTypes, TElementType>::init()
{
    sofa::component::solidmechanics::fem::elastic::FEMForceField<TDataTypes, TElementType>::init();

    if (!this->isComponentStateInvalid())
    {
        this->validateMaterial();
    }

    if (!this->isComponentStateInvalid())
    {
        precomputeData();
    }

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class TDataTypes, class TElementType>
void HyperelasticityFEMForceField<TDataTypes, TElementType>::buildStiffnessMatrix(
    sofa::core::behavior::StiffnessMatrix* matrix)
{
    if (this->isComponentStateInvalid())
        return;

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
        .withRespectToPositionsIn(this->mstate);

    if (!m_isHessianValid)
    {
        computeHessian(*m_coordinates);
    }

    sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real> localMatrix(sofa::type::NOINIT);

    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    auto elementStiffnessIt = m_elementStiffness.begin();
    for (const auto& element : elements)
    {
        const auto& stiffnessMatrix = *elementStiffnessIt++;

        for (sofa::Index n1 = 0; n1 < NumberOfNodesInElement; ++n1)
        {
            for (sofa::Index n2 = 0; n2 < NumberOfNodesInElement; ++n2)
            {
                stiffnessMatrix.getsub(spatial_dimensions * n1, spatial_dimensions * n2, localMatrix); //extract the submatrix corresponding to the coupling of nodes n1 and n2
                dfdx(element[n1] * spatial_dimensions, element[n2] * spatial_dimensions) += -localMatrix;
            }
        }
    }
}

template <class TDataTypes, class TElementType>
SReal HyperelasticityFEMForceField<TDataTypes, TElementType>::getPotentialEnergy(
    const sofa::core::MechanicalParams*, const DataVecCoord& x) const
{
    return 0;
}

template <class TDataTypes, class TElementType>
void HyperelasticityFEMForceField<TDataTypes, TElementType>::addKToMatrix(
    sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned& offset)
{
    if (this->isComponentStateInvalid())
        return;

    if (!m_isHessianValid)
    {
        computeHessian(*m_coordinates);
    }

    using LocalMatType = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;
    LocalMatType localMatrix{sofa::type::NOINIT};

    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    auto elementStiffnessIt = m_elementStiffness.begin();
    for (const auto& element : elements)
    {
        const auto& stiffnessMatrix = *elementStiffnessIt++;

        for (sofa::Index n1 = 0; n1 < NumberOfNodesInElement; ++n1)
        {
            for (sofa::Index n2 = 0; n2 < NumberOfNodesInElement; ++n2)
            {
                stiffnessMatrix.getsub(spatial_dimensions * n1, spatial_dimensions * n2, localMatrix); //extract the submatrix corresponding to the coupling of nodes n1 and n2
                const auto value = (-static_cast<Real>(kFact)) * static_cast<sofa::type::ScalarOrMatrix<LocalMatType>>(localMatrix);
                matrix->add(
                   offset + element[n1] * spatial_dimensions,
                   offset + element[n2] * spatial_dimensions, value);
            }
        }
    }
}

template <class TDataTypes, class TElementType>
void HyperelasticityFEMForceField<TDataTypes, TElementType>::validateMaterial()
{
    if (l_material.empty())
    {
        msg_info() << "Link to a valid material should be set to ensure right behavior. First "
                      "material found in current context will be used.";
        l_material.set(this->getContext()->template get<HyperelasticMaterial<TDataTypes>>());
    }

    if (l_material == nullptr)
    {
        msg_error() << "No material component found at path: '" << this->l_material.getLinkedPath()
                    << "', nor in current context: " << this->getContext()->name
                    << ". Object must have a material. "
                    << "The list of available material components is: "
                    << sofa::core::ObjectFactory::getInstance()
                           ->listClassesDerivedFrom<HyperelasticMaterial<TDataTypes>>();
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

template <class TDataTypes, class TElementType>
void HyperelasticityFEMForceField<TDataTypes, TElementType>::computeHessian(const VecCoord& coordinates)
{
    if (this->l_topology == nullptr) return;
    if (l_material == nullptr) return;

    SCOPED_TIMER("ComputeHessian");

    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);

    if (m_precomputedData.size() != elements.size())
    {
        precomputeData();
    }

    m_elementStiffness.clear();
    m_elementStiffness.resize(elements.size());

    auto elementStiffnessIt = m_elementStiffness.begin();

    std::size_t elementIndex = 0;
    for (const auto& element : elements)
    {
        SCOPED_TIMER_TR("Element");
        const std::array<Coord, NumberOfNodesInElement> elementNodesCoordinates = sofa::component::solidmechanics::fem::elastic::extractNodesVectorFromGlobalVector(element, this->mstate->readPositions().ref());

        ElementStiffness& K = *elementStiffnessIt++;
        K.clear();

        static constexpr auto quadraturePoints = FiniteElement::quadraturePoints();
        static constexpr auto gradients = sofa::fem::FiniteElementHelper<TElementType, TDataTypes>::gradientShapeFunctionAtQuadraturePoints();

        for (sofa::Size q = 0; q < NumberOfQuadraturePoints; ++q)
        {
            const auto& weight = quadraturePoints[q].second;
            const PrecomputedData& precomputedData = m_precomputedData[elementIndex][q];

            // gradient of shape functions in the reference element evaluated at the quadrature point
            const sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real>& dN_dq_ref = gradients[q];

            // jacobian of the mapping from the reference space to the physical space, evaluated at the
            // quadrature point
            const auto J_q = sofa::fem::FiniteElementHelper<TElementType, TDataTypes>::jacobianFromReferenceToPhysical(elementNodesCoordinates, dN_dq_ref);

            const auto detJ_Q = precomputedData.detJacobian;

            const sofa::type::Mat<TopologicalDimension, spatial_dimensions, Real>& J_Q_inv = precomputedData.jacobianInv;

            // gradient of the shape functions in the physical element evaluated at the quadrature point
            const sofa::type::Mat<NumberOfNodesInElement, spatial_dimensions, Real>& dN_dQ = precomputedData.dN_dQ;

            const DeformationGradient F = computeDeformationGradient(J_q, J_Q_inv);

            Strain<TDataTypes> strain(deformationGradient, F);

            // derivative of first Piola-Kirchhoff stress tensor with respect to deformation gradient
            const auto dPdF = [&]()
            {
                SCOPED_TIMER_VARNAME_TR(dPdFTimer, "dPdF");
                return l_material->materialTangentModulus(strain);
            }();

            const auto factor = detJ_Q * weight;

            SCOPED_TIMER_VARNAME_TR(tensorTimer, "Tensor");
            for (sofa::Size element_i = 0; element_i < NumberOfNodesInElement; ++element_i)
            {
                for (sofa::Size element_j = 0; element_j < NumberOfNodesInElement; ++element_j)
                {
                    for (sofa::Size dimension_i = 0; dimension_i < spatial_dimensions; ++dimension_i)
                    {
                        for (sofa::Size dimension_j = 0; dimension_j < spatial_dimensions; ++dimension_j)
                        {
                            auto& k = K(element_i * spatial_dimensions + dimension_i, element_j * spatial_dimensions + dimension_j);
                            for (sofa::Size i = 0; i < spatial_dimensions; ++i)
                            {
                                for (sofa::Size j = 0; j < spatial_dimensions; ++j)
                                {
                                     k += factor * dN_dQ[element_i][i] * dPdF(dimension_i, i, dimension_j, j) * dN_dQ[element_j][j];
                                }
                            }
                        }
                    }
                }
            }
        }
        ++elementIndex;
    }

    m_isHessianValid = true;
}

template <class TDataTypes, class TElementType>
auto HyperelasticityFEMForceField<TDataTypes, TElementType>::computeDeformationGradient(
    const sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real>& J_q,
    const sofa::type::Mat<TopologicalDimension, spatial_dimensions, Real>& J_Q_inv) -> DeformationGradient
{
    return J_q * J_Q_inv;
}

template <class TDataTypes, class TElementType>
void HyperelasticityFEMForceField<TDataTypes, TElementType>::precomputeData()
{
    if (this->l_topology == nullptr) return;

    auto restPositionAccessor = this->mstate->readRestPositions();
    const auto& restPosition = restPositionAccessor.ref();

    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    m_precomputedData.resize(elements.size());

    static constexpr auto gradients = sofa::fem::FiniteElementHelper<TElementType, TDataTypes>::gradientShapeFunctionAtQuadraturePoints();

    for (std::size_t i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];
        const std::array<Coord, NumberOfNodesInElement> elementNodesRestCoordinates = sofa::component::solidmechanics::fem::elastic::extractNodesVectorFromGlobalVector(element, restPosition);

        for (std::size_t j = 0; j < NumberOfQuadraturePoints; ++j)
        {
            // gradient of shape functions in the reference element evaluated at the quadrature point
            const sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real>& dN_dq_ref = gradients[j];

            PrecomputedData& data = m_precomputedData[i][j];
            data.jacobian = sofa::fem::FiniteElementHelper<TElementType, TDataTypes>::jacobianFromReferenceToPhysical(elementNodesRestCoordinates, dN_dq_ref);
            data.jacobianInv = sofa::type::inverse(data.jacobian);
            data.detJacobian = sofa::type::absGeneralizedDeterminant(data.jacobian);

            for (sofa::Size n = 0; n < NumberOfNodesInElement; ++n)
            {
                data.dN_dQ[n] = data.jacobianInv.multTranspose(dN_dq_ref[n]);
            }
        }
    }
}
template <class TDataTypes, class TElementType>
void HyperelasticityFEMForceField<TDataTypes, TElementType>::beforeElementForce(
    const sofa::core::MechanicalParams* mparams, sofa::type::vector<ElementGradient>& f,
    const sofa::VecCoord_t<TDataTypes>& x)
{
    //store coordinates to use it later when computing the Hessian
    m_coordinates = &x;

    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);
    if (m_precomputedData.size() != elements.size())
    {
        precomputeData();
    }

    m_isHessianValid = false;

    //reset force vector
    for (auto& elF : f)
    {
        elF.clear();
    }
}

template <class TDataTypes, class TElementType>
void HyperelasticityFEMForceField<TDataTypes, TElementType>::computeElementsForces(
    const sofa::simulation::Range<std::size_t>& range, const sofa::core::MechanicalParams* mparams,
    sofa::type::vector<ElementGradient>& f, const sofa::VecCoord_t<TDataTypes>& x)
{
    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);

    static constexpr auto quadraturePoints = FiniteElement::quadraturePoints();
    static constexpr auto gradients = sofa::fem::FiniteElementHelper<TElementType, TDataTypes>::gradientShapeFunctionAtQuadraturePoints();

    for (std::size_t elementId = range.start; elementId < range.end; ++elementId)
    {
        const auto element = elements[elementId];
        const std::array<Coord, NumberOfNodesInElement> elementNodesCoordinates = sofa::component::solidmechanics::fem::elastic::extractNodesVectorFromGlobalVector(element, x);

        //access to the force vector in the element. This is the value to compute in this iteration
        //force assembly at the DoF level is done in a later function.
        auto& elementForce = f[elementId];

        for (sofa::Size q = 0; q < NumberOfQuadraturePoints; ++q)
        {
            const auto& weight = quadraturePoints[q].second;
            const PrecomputedData& precomputedData = m_precomputedData[elementId][q];

            // gradient of shape functions in the reference element evaluated at the quadrature point
            const sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real>& dN_dq_ref = gradients[q];

            // jacobian of the mapping from the reference space to the physical space, evaluated at the
            // quadrature point
            const auto J_q = sofa::fem::FiniteElementHelper<TElementType, TDataTypes>::jacobianFromReferenceToPhysical(elementNodesCoordinates, dN_dq_ref);

            const auto detJ_Q = precomputedData.detJacobian;

            const sofa::type::Mat<TopologicalDimension, spatial_dimensions, Real>& J_Q_inv = precomputedData.jacobianInv;

            // gradient of the shape functions in the physical element evaluated at the quadrature point
            const sofa::type::Mat<NumberOfNodesInElement, spatial_dimensions, Real>& dN_dQ = precomputedData.dN_dQ;

            const DeformationGradient F = computeDeformationGradient(J_q, J_Q_inv);

            Strain<TDataTypes> strain(deformationGradient, F);
            const auto P = l_material->firstPiolaKirchhoffStress(strain);

            for (sofa::Size i = 0; i < trait::NumberOfNodesInElement; ++i)
            {
                const auto f_q = (detJ_Q * weight) * P * dN_dQ[i];
                for (sofa::Size j = 0; j < trait::spatial_dimensions; ++j)
                {
                    elementForce[i * trait::spatial_dimensions + j] += f_q[j];
                }
            }
        }
    }
}

template <class TDataTypes, class TElementType>
void HyperelasticityFEMForceField<TDataTypes, TElementType>::beforeElementForceDeriv(
    const sofa::core::MechanicalParams* mparams)
{
    if (!m_isHessianValid)
    {
        computeHessian(*m_coordinates);
    }
}

template <class TDataTypes, class TElementType>
void HyperelasticityFEMForceField<TDataTypes, TElementType>::computeElementsForcesDeriv(
    const sofa::simulation::Range<std::size_t>& range, const sofa::core::MechanicalParams* mparams,
    sofa::type::vector<ElementGradient>& df, const sofa::VecDeriv_t<TDataTypes>& dx)
{
    const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);

    for (std::size_t elementId = range.start; elementId < range.end; ++elementId)
    {
        const auto& element = elements[elementId];
        const auto& stiffnessMatrix = m_elementStiffness[elementId];

        const std::array<sofa::Coord_t<TDataTypes>, trait::NumberOfNodesInElement> elementNodesDx =
            sofa::component::solidmechanics::fem::elastic::extractNodesVectorFromGlobalVector(element, dx);

        sofa::type::Vec<trait::NumberOfDofsInElement, sofa::Real_t<TDataTypes>> element_dx(sofa::type::NOINIT);
        for (sofa::Size nodeId = 0; nodeId < trait::NumberOfNodesInElement; ++nodeId)
        {
            const auto& nodeDx = elementNodesDx[nodeId];
            for (sofa::Size dim = 0; dim < trait::spatial_dimensions; ++dim)
            {
                element_dx[nodeId * trait::spatial_dimensions + dim] = nodeDx[dim];
            }
        }

        df[elementId] = stiffnessMatrix * element_dx;
    }
}

}  // namespace elasticity
