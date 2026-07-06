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

#include <sofa/geometry/config.h>
#include <sofa/geometry/ElementType.h>

namespace sofa::geometry
{

struct QuadraticEdge
{
    static constexpr sofa::Size NumberOfNodes = 3;
    static constexpr sofa::Size PolynomialOrder = 2;
    static constexpr ElementType Element_type = ElementType::QUADRATIC_EDGE;
};

struct QuadraticTriangle
{
    static constexpr sofa::Size NumberOfNodes = 6;
    static constexpr sofa::Size PolynomialOrder = 2;
    static constexpr ElementType Element_type = ElementType::QUADRATIC_TRIANGLE;
};

struct QuadraticQuad
{
    static constexpr sofa::Size NumberOfNodes = 9;
    static constexpr sofa::Size PolynomialOrder = 2;
    static constexpr ElementType Element_type = ElementType::QUADRATIC_QUAD;
};

struct QuadraticTetrahedron
{
    static constexpr sofa::Size NumberOfNodes = 10;
    static constexpr sofa::Size PolynomialOrder = 2;
    static constexpr ElementType Element_type = ElementType::QUADRATIC_TETRAHEDRON;
};

struct QuadraticHexahedron
{
    static constexpr sofa::Size NumberOfNodes = 27;
    static constexpr sofa::Size PolynomialOrder = 2;
    static constexpr ElementType Element_type = ElementType::QUADRATIC_HEXAHEDRON;
};

struct QuadraticPrism
{
    static constexpr sofa::Size NumberOfNodes = 18;
    static constexpr sofa::Size PolynomialOrder = 2;
    static constexpr ElementType Element_type = ElementType::QUADRATIC_PRISM;
};

struct QuadraticPyramid
{
    static constexpr sofa::Size NumberOfNodes = 14;
    static constexpr sofa::Size PolynomialOrder = 2;
    static constexpr ElementType Element_type = ElementType::QUADRATIC_PYRAMID;
};

}
