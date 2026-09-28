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
#define SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_HYPERLASTICITY_FEM_FORCE_FIELD_CPP

#include <sofa/component/solidmechanics/fem/hyperelastic/HyperelasticityFEMForceField.inl>

#include <sofa/fem/FiniteElement[all].h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

void registerHyperelasticityFEMForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Hyperelasticity")
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge> >()
        // .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge> >()
        // .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle> >()
        // .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad> >()
        // .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron> >()
    );
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
// template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
// template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
// template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
// template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

}
