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
#define SOFA_COMPONENT_MASS_FEMMASS_CPP
#include <sofa/component/mass/FEMMass.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>

namespace sofa::component::mass
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
using sofa::geometry::QuadraticEdge;
using sofa::geometry::QuadraticTriangle;
using sofa::geometry::QuadraticQuad;
using sofa::geometry::QuadraticTetrahedron;
using sofa::geometry::QuadraticHexahedron;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec1Types, Edge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec1Types, Edge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec1Types, Edge>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec2Types, Edge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec2Types, Edge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec2Types, Edge>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, Edge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, Edge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, Edge>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec2Types, Triangle>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec2Types, Triangle>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec2Types, Triangle>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, Triangle>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, Triangle>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, Triangle>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec2Types, Quad>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec2Types, Quad>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec2Types, Quad>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, Quad>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, Quad>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, Quad>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, Tetrahedron>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, Tetrahedron>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, Tetrahedron>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, Hexahedron>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, Hexahedron>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, Hexahedron>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, Prism>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, Prism>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, Prism>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, Pyramid>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, Pyramid>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, Pyramid>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec1Types, QuadraticEdge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec1Types, QuadraticEdge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec1Types, QuadraticEdge>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec2Types, QuadraticEdge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec2Types, QuadraticEdge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec2Types, QuadraticEdge>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, QuadraticEdge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, QuadraticEdge>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, QuadraticEdge>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec2Types, QuadraticTriangle>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec2Types, QuadraticTriangle>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec2Types, QuadraticTriangle>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, QuadraticTriangle>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, QuadraticTriangle>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, QuadraticTriangle>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec2Types, QuadraticQuad>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec2Types, QuadraticQuad>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec2Types, QuadraticQuad>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, QuadraticQuad>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, QuadraticQuad>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, QuadraticQuad>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, QuadraticTetrahedron>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, QuadraticTetrahedron>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, QuadraticTetrahedron>;

template class SOFA_COMPONENT_MASS_API FEMMass<Vec1Types, Vec3Types, QuadraticHexahedron>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec2Types, Vec3Types, QuadraticHexahedron>;
template class SOFA_COMPONENT_MASS_API FEMMass<Vec3Types, Vec3Types, QuadraticHexahedron>;

void registerFEMMass(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Finite-element mass (inertia and body force)")
        .add< FEMMass<Vec1Types, Vec1Types, Edge> >()
        .add< FEMMass<Vec2Types, Vec1Types, Edge> >()
        .add< FEMMass<Vec3Types, Vec1Types, Edge> >()

        .add< FEMMass<Vec1Types, Vec2Types, Edge> >()
        .add< FEMMass<Vec2Types, Vec2Types, Edge> >()
        .add< FEMMass<Vec3Types, Vec2Types, Edge> >()

        .add< FEMMass<Vec1Types, Vec3Types, Edge> >()
        .add< FEMMass<Vec2Types, Vec3Types, Edge> >()
        .add< FEMMass<Vec3Types, Vec3Types, Edge> >()

        .add< FEMMass<Vec1Types, Vec2Types, Triangle> >()
        .add< FEMMass<Vec2Types, Vec2Types, Triangle> >()
        .add< FEMMass<Vec3Types, Vec2Types, Triangle> >()

        .add< FEMMass<Vec1Types, Vec3Types, Triangle> >()
        .add< FEMMass<Vec2Types, Vec3Types, Triangle> >()
        .add< FEMMass<Vec3Types, Vec3Types, Triangle> >()

        .add< FEMMass<Vec1Types, Vec2Types, Quad> >()
        .add< FEMMass<Vec2Types, Vec2Types, Quad> >()
        .add< FEMMass<Vec3Types, Vec2Types, Quad> >()

        .add< FEMMass<Vec1Types, Vec3Types, Quad> >()
        .add< FEMMass<Vec2Types, Vec3Types, Quad> >()
        .add< FEMMass<Vec3Types, Vec3Types, Quad> >()

        .add< FEMMass<Vec1Types, Vec3Types, Tetrahedron> >()
        .add< FEMMass<Vec2Types, Vec3Types, Tetrahedron> >()
        .add< FEMMass<Vec3Types, Vec3Types, Tetrahedron> >()

        .add< FEMMass<Vec1Types, Vec3Types, Hexahedron> >()
        .add< FEMMass<Vec2Types, Vec3Types, Hexahedron> >()
        .add< FEMMass<Vec3Types, Vec3Types, Hexahedron> >()

        .add< FEMMass<Vec1Types, Vec3Types, Prism> >()
        .add< FEMMass<Vec2Types, Vec3Types, Prism> >()
        .add< FEMMass<Vec3Types, Vec3Types, Prism> >()

        .add< FEMMass<Vec1Types, Vec3Types, Pyramid> >()
        .add< FEMMass<Vec2Types, Vec3Types, Pyramid> >()
        .add< FEMMass<Vec3Types, Vec3Types, Pyramid> >()

        .add< FEMMass<Vec1Types, Vec1Types, QuadraticEdge> >()
        .add< FEMMass<Vec2Types, Vec1Types, QuadraticEdge> >()
        .add< FEMMass<Vec3Types, Vec1Types, QuadraticEdge> >()

        .add< FEMMass<Vec1Types, Vec2Types, QuadraticEdge> >()
        .add< FEMMass<Vec2Types, Vec2Types, QuadraticEdge> >()
        .add< FEMMass<Vec3Types, Vec2Types, QuadraticEdge> >()

        .add< FEMMass<Vec1Types, Vec3Types, QuadraticEdge> >()
        .add< FEMMass<Vec2Types, Vec3Types, QuadraticEdge> >()
        .add< FEMMass<Vec3Types, Vec3Types, QuadraticEdge> >()

        .add< FEMMass<Vec1Types, Vec2Types, QuadraticTriangle> >()
        .add< FEMMass<Vec2Types, Vec2Types, QuadraticTriangle> >()
        .add< FEMMass<Vec3Types, Vec2Types, QuadraticTriangle> >()

        .add< FEMMass<Vec1Types, Vec3Types, QuadraticTriangle> >()
        .add< FEMMass<Vec2Types, Vec3Types, QuadraticTriangle> >()
        .add< FEMMass<Vec3Types, Vec3Types, QuadraticTriangle> >()

        .add< FEMMass<Vec1Types, Vec2Types, QuadraticQuad> >()
        .add< FEMMass<Vec2Types, Vec2Types, QuadraticQuad> >()
        .add< FEMMass<Vec3Types, Vec2Types, QuadraticQuad> >()

        .add< FEMMass<Vec1Types, Vec3Types, QuadraticQuad> >()
        .add< FEMMass<Vec2Types, Vec3Types, QuadraticQuad> >()
        .add< FEMMass<Vec3Types, Vec3Types, QuadraticQuad> >()

        .add< FEMMass<Vec1Types, Vec3Types, QuadraticTetrahedron> >()
        .add< FEMMass<Vec2Types, Vec3Types, QuadraticTetrahedron> >()
        .add< FEMMass<Vec3Types, Vec3Types, QuadraticTetrahedron> >()

        .add< FEMMass<Vec1Types, Vec3Types, QuadraticHexahedron> >()
        .add< FEMMass<Vec2Types, Vec3Types, QuadraticHexahedron> >()
        .add< FEMMass<Vec3Types, Vec3Types, QuadraticHexahedron> >()
    );
}

}
