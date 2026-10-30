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
#define SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_FEM_FORCEFIELD_CPP
#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.inl>
#include <sofa/fem/FiniteElement[all].h>

namespace sofa::component::solidmechanics::fem::elastic
{

using sofa::defaulttype::Vec1Types;
using sofa::defaulttype::Vec2Types;
using sofa::defaulttype::Vec3Types;
using sofa::geometry::Edge;
using sofa::geometry::Triangle;
using sofa::geometry::Quad;
using sofa::geometry::Tetrahedron;
using sofa::geometry::Hexahedron;
using sofa::geometry::Prism;
using sofa::geometry::Pyramid;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec1Types, Vec1Types, Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec2Types, Vec1Types, Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec3Types, Vec1Types, Edge>;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec1Types, Vec2Types, Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec2Types, Vec2Types, Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec3Types, Vec2Types, Edge>;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec1Types, Vec3Types, Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec2Types, Vec3Types, Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec3Types, Vec3Types, Edge>;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec1Types, Vec2Types, Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec2Types, Vec2Types, Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec3Types, Vec2Types, Triangle>;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec1Types, Vec3Types, Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec2Types, Vec3Types, Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec3Types, Vec3Types, Triangle>;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec1Types, Vec2Types, Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec2Types, Vec2Types, Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec3Types, Vec2Types, Quad>;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec1Types, Vec3Types, Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec2Types, Vec3Types, Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec3Types, Vec3Types, Quad>;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec1Types, Vec3Types, Tetrahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec2Types, Vec3Types, Tetrahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec3Types, Vec3Types, Tetrahedron>;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec1Types, Vec3Types, Hexahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec2Types, Vec3Types, Hexahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec3Types, Vec3Types, Hexahedron>;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec1Types, Vec3Types, Prism>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec2Types, Vec3Types, Prism>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec3Types, Vec3Types, Prism>;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec1Types, Vec3Types, Pyramid>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec2Types, Vec3Types, Pyramid>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API FEMForceField<Vec3Types, Vec3Types, Pyramid>;

}
