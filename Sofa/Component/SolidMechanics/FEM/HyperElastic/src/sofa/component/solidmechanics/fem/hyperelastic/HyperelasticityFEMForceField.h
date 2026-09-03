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

#include <sofa/component/solidmechanics/fem/hyperelastic/HyperelasticMaterial.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/config.h>
#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_HYPERLASTICITY_FEM_FORCE_FIELD_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class TDataTypes, class TElementType>
class HyperelasticityFEMForceField :
    public sofa::component::solidmechanics::fem::elastic::FEMForceField<TDataTypes, TElementType>
{
public:
    SOFA_CLASS(
        SOFA_TEMPLATE2(HyperelasticityFEMForceField, TDataTypes, TElementType),
        SOFA_TEMPLATE2(sofa::component::solidmechanics::fem::elastic::FEMForceField, TDataTypes, TElementType));

    using DataTypes = TDataTypes;

private:
    using DataVecCoord = sofa::DataVecDeriv_t<TDataTypes>;
    using DataVecDeriv = sofa::DataVecDeriv_t<TDataTypes>;
    using VecCoord = sofa::VecCoord_t<TDataTypes>;
    using VecDeriv = sofa::VecDeriv_t<TDataTypes>;
    using Coord = sofa::Coord_t<TDataTypes>;
    using Deriv = sofa::Deriv_t<TDataTypes>;
    using Real = sofa::Real_t<TDataTypes>;

    using FiniteElement = sofa::fem::FiniteElement<TElementType, TDataTypes>;

    static constexpr sofa::Size spatial_dimensions = TDataTypes::spatial_dimensions;
    static constexpr sofa::Size NumberOfNodesInElement = TElementType::NumberOfNodes;
    static constexpr sofa::Size NumberOfDofsInElement = NumberOfNodesInElement * spatial_dimensions;
    static constexpr sofa::Size TopologicalDimension = FiniteElement::TopologicalDimension;
    static constexpr sofa::Size NumberOfQuadraturePoints = FiniteElement::quadraturePoints().size();

    using DeformationGradient = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;

    /// the type of the element stiffness matrix
    using ElementStiffness = sofa::type::Mat<
        TElementType::NumberOfNodes * DataTypes::spatial_dimensions,
        TElementType::NumberOfNodes * DataTypes::spatial_dimensions,
        sofa::Real_t<DataTypes>
    >;

public:
    void init() override;

    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;

    using Inherit1::getPotentialEnergy;
    SReal getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord& x) const override;

    using Inherit1::addKToMatrix;
    // almost deprecated, but here for compatibility with unit tests
    void addKToMatrix(sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned& offset) override;

    sofa::SingleLink<MyType, HyperelasticMaterial<TDataTypes>,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_material;

protected:

    HyperelasticityFEMForceField();

    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, TElementType>;
    using ElementGradient = typename trait::ElementGradient;

    void validateMaterial();

    bool m_isHessianValid {false};

    void computeHessian(const VecCoord& coordinates);

    /**
     * List of precomputed element stiffness matrices
     */
    sofa::type::vector<ElementStiffness> m_elementStiffness;

    const VecCoord* m_coordinates{ nullptr };

    DeformationGradient computeDeformationGradient(
        const sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real>& J_q,
        const sofa::type::Mat<TopologicalDimension, spatial_dimensions, Real>& J_Q_inv);

    struct PrecomputedData
    {
        // jacobian of the mapping from the reference space to the rest physical space, evaluated at the
        // quadrature point
        sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real> jacobian { sofa::type::NOINIT };

        // inverse of the jacobian of the mapping from the reference space to the rest physical space,
        // evaluated at the quadrature point
        sofa::type::Mat<TopologicalDimension, spatial_dimensions, Real> jacobianInv { sofa::type::NOINIT };

        Real detJacobian {};

        // gradient of the shape functions in the physical element evaluated at the quadrature point
        sofa::type::Mat<NumberOfNodesInElement, spatial_dimensions, Real> dN_dQ { sofa::type::NOINIT };
    };

    /**
     * Data can be precomputed from the rest configuration and used later in computations for the
     * current configuration. A piece of precomputed data is stored for each quadrature point in
     * each element.
     */
    sofa::type::vector<std::array<PrecomputedData, NumberOfQuadraturePoints>> m_precomputedData;

    void precomputeData();

    void beforeElementForce(const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementGradient>& f,
        const sofa::VecCoord_t<DataTypes>& x) override;

    void computeElementsForces(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementGradient>& f,
        const sofa::VecCoord_t<TDataTypes>& x) override;

    void beforeElementForceDeriv(const sofa::core::MechanicalParams* mparams) override;

    void computeElementsForcesDeriv(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementGradient>& df,
        const sofa::VecDeriv_t<TDataTypes>& dx) override;
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_HYPERLASTICITY_FEM_FORCE_FIELD_CPP)
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
// template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
// template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
// template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
// template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace elasticity
