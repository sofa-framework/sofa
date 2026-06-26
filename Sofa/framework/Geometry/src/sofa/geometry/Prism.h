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

struct Prism
{
    static constexpr sofa::Size NumberOfNodes = 6;
    static constexpr ElementType Element_type = ElementType::PRISM;

    Prism() = delete;

    /* CONVENTION : indices ordering for the nodes of a prism :
     *
     *      5
     *    / |  \
     *   /  |   \
     *  3---+----4
     *  |   |    |
     *  |   2    |
     *  | /   \  |
     *  |/      \|
     *  0--------1
     */
};

using Pentahedron SOFA_ATTRIBUTE_DEPRECATED("v25.12", "v26.06", "Pentahedron is renamed to Prism") = Prism;

} // namespace sofa::geometry
