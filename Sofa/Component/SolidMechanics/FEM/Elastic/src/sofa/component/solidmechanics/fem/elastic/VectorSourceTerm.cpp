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
#define SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_VECTOR_SOURCE_TERM_CPP

#include <sofa/component/solidmechanics/fem/elastic/VectorSourceTerm.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>

namespace sofa::component::solidmechanics::fem::elastic
{

void registerNodalSourceDensity(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Definition of a nodal source density (one vector per dof).")
        .add< NodalSourceDensity<sofa::defaulttype::Vec1Types> >()
        .add< NodalSourceDensity<sofa::defaulttype::Vec2Types> >()
        .add< NodalSourceDensity<sofa::defaulttype::Vec3Types> >()
    );
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalSourceDensity<sofa::defaulttype::Vec1Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalSourceDensity<sofa::defaulttype::Vec2Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalSourceDensity<sofa::defaulttype::Vec3Types>;

void registerVectorSourceTerm(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Source density given as a vector at each node, per unit measure of the element")
        .add< VectorSourceTerm<sofa::defaulttype::Vec1Types, sofa::geometry::Edge> >()
        .add< VectorSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge> >()
        .add< VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Edge> >()
        .add< VectorSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle> >()
        .add< VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle> >()
        .add< VectorSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Quad> >()
        .add< VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad> >()
        .add< VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron> >()
        .add< VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron> >()
    );
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API VectorSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

}  // namespace sofa::component::solidmechanics::fem::elastic
