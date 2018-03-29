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
#ifndef SOFA_CORE_LOADER_MATERIAL_H_
#define SOFA_CORE_LOADER_MATERIAL_H_

#include <sofa/helper/types/Material.h>

namespace sofa
{

namespace core
{

namespace loader
{

///The Material object that was previously in this sofa::core::loader is now in sofa::helper:types::Material.
///The following lines is there to provide backward compatibility with existing code base.
///This is just there for a transitional period of time and will be removed after 2018-01-07
using sofa::helper::types::Material ;

//TODO(dmarchal 2017-06-13): Delete that around 2018-01-07


} // namespace loader

} // namespace core

} // namespace sofa

#endif /* SOFA_CORE_LOADER_MATERIAL_H_ */
