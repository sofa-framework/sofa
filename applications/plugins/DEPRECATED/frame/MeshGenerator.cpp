/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_FRAME_MESHGENERATOR_CPP

#include "MeshGenerator.inl"

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec.h>


namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology;
using namespace sofa::core::topology;
using namespace sofa::core;
using namespace sofa::helper::gl;
using namespace sofa::simulation;

SOFA_DECL_CLASS ( MeshGenerator );

// Register in the Factory
int MeshGeneratorClass = core::RegisterObject ( "Special case of mapping where HexahedronSetTopology is converted to QuadSetTopology" )
#ifndef SOFA_FLOAT   
	.add< MeshGenerator<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
	.add< MeshGenerator<Vec3fTypes> >()
#endif
;

} // namespace engine

} // namespace component

} // namespace sofa

