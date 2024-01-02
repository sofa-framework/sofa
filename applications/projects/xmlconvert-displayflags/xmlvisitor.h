/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_XMLCONVERT_XMLVISITOR
#define SOFA_XMLCONVERT_XMLVISITOR

#include <tinyxml2.h>
#include <map>
#include <vector>
#include <sofa/core/visual/DisplayFlags.h>

namespace sofa
{
namespace xml
{

class DiscoverNodes : public tinyxml2::XMLVisitor
{
public:

    virtual bool VisitEnter( const tinyxml2::XMLElement&, const tinyxml2::XMLAttribute* );

    tinyxml2::XMLElement* rootNode()
    {
        if( nodes.empty() ) return NULL;
        else return nodes[0];
    }

    std::vector<tinyxml2::XMLElement* > leaves;
    std::vector<tinyxml2::XMLElement* > nodes;

};

class DiscoverDisplayFlagsVisitor : public tinyxml2::XMLVisitor
{
public:

    virtual bool VisitEnter( const tinyxml2::XMLElement&, const tinyxml2::XMLAttribute* );
    ~DiscoverDisplayFlagsVisitor();
    std::map<const tinyxml2::XMLElement*,sofa::core::visual::DisplayFlags*> map_displayFlags;

};

void createVisualStyleVisitor(tinyxml2::XMLElement* leaf, const std::map<const tinyxml2::XMLElement*,sofa::core::visual::DisplayFlags*>& map_displayFlags);

void removeShowAttributes(tinyxml2::XMLElement* node);

void convert_false_to_neutral(sofa::core::visual::DisplayFlags& flags);


}
}


#endif
