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

#include <sofa/component/solidmechanics/fem/elastic/config.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/trait.h>
#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.h>

#if !defined(ELASTICITY_COMPONENT_BASE_ELEMENT_LINEAR_FEM_FORCEFIELD_CPP)
#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * A base class for all element-based linear elastic force fields.
 *
 * It stores precomputed stiffness matrices (one per element) that are derived from:
 *   - The initial configuration of the mechanical model
 *   - Material properties (Young's modulus, Poisson's ratio)
 */
template <class DataTypes, class ElementType>
class BaseElementLinearFEMForceField : public sofa::component::solidmechanics::fem::elastic::BaseLinearElasticityFEMForceField<DataTypes>
{
public:
    SOFA_ABSTRACT_CLASS(
        SOFA_TEMPLATE2(BaseElementLinearFEMForceField, DataTypes, ElementType),
        sofa::component::solidmechanics::fem::elastic::BaseLinearElasticityFEMForceField<DataTypes>);

    void init() override;

private:
    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;
    using ElementStiffness = typename trait::ElementStiffness;
    using ElasticityTensor = typename trait::ElasticityTensor;
    using StrainDisplacement = typename trait::StrainDisplacement;

protected:

    BaseElementLinearFEMForceField();

    /**
     * With linear small strain, the element stiffness matrix is constant, so it can be precomputed.
     */
    void precomputeElementStiffness();

public:

    /**
     * List of precomputed element stiffness matrices
     */
    sofa::Data<sofa::type::vector<ElementStiffness> > d_elementStiffness;
};

#if !defined(ELASTICITY_COMPONENT_BASE_ELEMENT_LINEAR_FEM_FORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
