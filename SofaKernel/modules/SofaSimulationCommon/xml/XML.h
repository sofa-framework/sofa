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
#ifndef SOFA_SIMULATION_COMMON_XML_XML_H
#define SOFA_SIMULATION_COMMON_XML_XML_H

#include <SofaSimulationCommon/common.h>
#include <SofaSimulationCommon/xml/Element.h>

#ifdef SOFA_XML_PARSER_TINYXML
#include <tinyxml.h>
#endif
#ifdef SOFA_XML_PARSER_LIBXML
#include <libxml/parser.h>
#include <libxml/tree.h>
#endif


namespace sofa
{

namespace simulation
{

namespace xml
{

#ifdef SOFA_XML_PARSER_TINYXML
SOFA_SIMULATION_COMMON_API BaseElement* processXMLLoading(const char *filename, const TiXmlDocument &doc, bool fromMem=false);
#endif
#ifdef SOFA_XML_PARSER_LIBXML
SOFA_SIMULATION_COMMON_API BaseElement* processXMLLoading(const char *filename, const xmlDocPtr &doc, bool fromMem=false);
#endif

SOFA_SIMULATION_COMMON_API BaseElement* loadFromFile(const char *filename);

SOFA_SIMULATION_COMMON_API BaseElement* loadFromMemory(const char *filename, const char *data, unsigned int size );


SOFA_SIMULATION_COMMON_API bool save(const char *filename, BaseElement* root);

extern int SOFA_SIMULATION_COMMON_API numDefault;

} // namespace xml

} // namespace simulation

} // namespace sofa

#endif
