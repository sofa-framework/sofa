/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaMiscMapping/VoidMapping.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace core;
using namespace core::behavior;


SOFA_DECL_CLASS(VoidMapping)

// Register in the Factory
int VoidMappingClass = core::RegisterObject("Special mapping that 'map' points for void ( no input DOF ). This is useful to be able to create animated objects mixed with real DOFs.")
        .add< VoidMapping >()
        ;

} // namespace mapping

} // namespace component

} // namespace sofa

