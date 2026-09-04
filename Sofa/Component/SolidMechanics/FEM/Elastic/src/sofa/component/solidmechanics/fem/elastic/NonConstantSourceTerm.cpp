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
#define SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_NON_CONSTANT_SOURCE_TERM_CPP

#include <sofa/component/solidmechanics/fem/elastic/NonConstantSourceTerm.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>
#include <sofa/geometry/Edge.h>
#include <sofa/geometry/Hexahedron.h>
#include <sofa/geometry/Quad.h>
#include <sofa/geometry/Tetrahedron.h>
#include <sofa/geometry/Triangle.h>

namespace sofa::component::solidmechanics::fem::elastic
{

void registerNonConstantSourceTerm(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Source term (per unit volume) depending on the current displacement; zero unless subclassed")
        .add< NonConstantSourceTerm<sofa::defaulttype::Vec1Types, sofa::geometry::Edge> >()
        .add< NonConstantSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge> >()
        .add< NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Edge> >()
        .add< NonConstantSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle> >()
        .add< NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle> >()
        .add< NonConstantSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Quad> >()
        .add< NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad> >()
        .add< NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron> >()
        .add< NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron> >()
    );
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NonConstantSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

}  // namespace sofa::component::solidmechanics::fem::elastic
