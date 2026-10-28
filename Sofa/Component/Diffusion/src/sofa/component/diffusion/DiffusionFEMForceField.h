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

#include <sofa/component/diffusion/config.h>

#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/trait.h>

#if !defined(SOFA_COMPONENT_DIFFUSION_DIFFUSION_FEM_FORCE_FIELD_CPP)
#include <sofa/fem/FiniteElement[all].h>
#endif

namespace sofa::component::diffusion
{
template <class DiffusionFieldTypes, class SpatialTypes, class ElementType>
class DiffusionFEMForceField :
    public sofa::component::solidmechanics::fem::elastic::FEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>
{
public:
    SOFA_CLASS(
        SOFA_TEMPLATE3(DiffusionFEMForceField, DiffusionFieldTypes, SpatialTypes, ElementType),
        SOFA_TEMPLATE3(sofa::component::solidmechanics::fem::elastic::FEMForceField, DiffusionFieldTypes, SpatialTypes, ElementType));

private:
    using Inherited = sofa::component::solidmechanics::fem::elastic::FEMForceField<DiffusionFieldTypes, SpatialTypes, ElementType>;
    using trait = sofa::component::solidmechanics::fem::elastic::trait<DiffusionFieldTypes, SpatialTypes, ElementType>;

    using Real = sofa::Real_t<DiffusionFieldTypes>;          // type of the diffused quantity
    using GeometryReal = sofa::Real_t<SpatialTypes>;         // type of the geometry

    using ElementHessian = typename trait::ElementHessian;
    using ElementGradient = typename trait::ElementGradient;
    using ElementValue = typename trait::ElementDisplacement;

    using GeometryCoord = sofa::Coord_t<SpatialTypes>;

    /// Number of components of the diffused quantity at each node (1 for a scalar field)
    static constexpr sofa::Size FieldDimension = trait::field_dimensions;
    /// Dimension of the space in which the mesh lives
    static constexpr sofa::Size SpatialDimension = trait::spatial_dimensions;
    static constexpr sofa::Size NumberOfNodes = trait::NumberOfNodesInElement;

    using FieldBlock = sofa::type::Mat<FieldDimension, FieldDimension, Real>;

public:
    DiffusionFEMForceField();

    void init() override;

    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;

    using sofa::core::behavior::ForceField<DiffusionFieldTypes>::getPotentialEnergy;
    SReal getPotentialEnergy(
        const sofa::core::MechanicalParams*,
        const sofa::DataVecCoord_t<DiffusionFieldTypes>& fieldValues) const override;

    using sofa::core::behavior::ForceField<DiffusionFieldTypes>::addKToMatrix;
    void addKToMatrix(sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned& offset) override;

protected:
    void computeElementsForces(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementGradient>& elementForces,
        const sofa::VecCoord_t<DiffusionFieldTypes>& nodeValues) override;

    void computeElementsForcesDeriv(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementGradient>& elementForcesDeriv,
        const sofa::VecDeriv_t<DiffusionFieldTypes>& nodeDx) override;

    void validateSpatialState();

private:
    void precomputeElementStiffness();

    /// Gathers the nodal values of the diffused quantity (or of its increment) of one element
    template <class VecType>
    ElementValue computeElementValue(
        const typename trait::TopologyElement& element,
        const VecType& nodeValues) const;

    Real getElementDiffusionCoefficient(const sofa::helper::ReadAccessor<sofa::Data<sofa::type::vector<Real> > >& elementDiffusionCoefficient, sofa::Index elementId) const;

public:
    static constexpr Real defaultDiffusionCoefficient = 1;

    sofa::Data<sofa::type::vector<Real> > d_elementDiffusionCoefficient;
    sofa::Data<sofa::type::vector<ElementHessian> > d_elementHessian;

    sofa::SingleLink<DiffusionFEMForceField,
                     sofa::core::behavior::MechanicalState<SpatialTypes>,
                     sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> l_spatialState;
};

#if !defined(SOFA_COMPONENT_DIFFUSION_DIFFUSION_FEM_FORCE_FIELD_CPP)
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;

extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;

extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;

extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;

extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;

extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;

extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Prism>;

extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;
extern template class SOFA_COMPONENT_DIFFUSION_API DiffusionFEMForceField<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types, sofa::geometry::Pyramid>;
#endif

} // namespace sofa::component::diffusion
