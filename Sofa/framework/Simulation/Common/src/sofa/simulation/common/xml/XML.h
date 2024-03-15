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
#include <sofa/simulation/common/config.h>
#include <sofa/simulation/common/xml/Element.h>

namespace tinyxml2
{

  class XMLDocument;

} // namespace tinyxml2

namespace sofa::simulation::xml
{

SOFA_SIMULATION_COMMON_API BaseElement* processXMLLoading(const char *filename, const tinyxml2::XMLDocument &doc, bool fromMem=false);

SOFA_SIMULATION_COMMON_API BaseElement* loadFromFile(const char *filename);

SOFA_SIMULATION_COMMON_API BaseElement* loadFromMemory(const char *filename, const char *data);


SOFA_SIMULATION_COMMON_API bool save(const char *filename, BaseElement* root);

extern int SOFA_SIMULATION_COMMON_API numDefault;

} // namespace sofa::simulation::xml
