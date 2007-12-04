/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
//
// C++ Implementation: TriangularBendingSprings
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/component/forcefield/TriangularBendingSprings.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/MeshTopology.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>

// #define DEBUG_TRIANGLEBS

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

using std::cerr;
using std::cout;
using std::endl;

SOFA_DECL_CLASS(TriangularBendingSprings)

template class TriangularBendingSprings<Vec3fTypes>;
template class TriangularBendingSprings<Vec3dTypes>;
template class TriangularBendingSprings<Vec2fTypes>;
template class TriangularBendingSprings<Vec2dTypes>;


// Register in the Factory
int TriangularBendingSpringsClass = core::RegisterObject("Springs added to a triangular mesh to prevent bending")
        .add< TriangularBendingSprings<Vec3dTypes> >()
        .add< TriangularBendingSprings<Vec3fTypes> >()
        .add< TriangularBendingSprings<Vec2dTypes> >()
        .add< TriangularBendingSprings<Vec2fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

