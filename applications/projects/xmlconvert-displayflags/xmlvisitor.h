/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <tinyxml.h>
#include <map>
#include <vector>
#include <sofa/core/visual/DisplayFlags.h>

namespace sofa
{
namespace xml
{

class DiscoverNodes : public TiXmlVisitor
{
public:

    virtual bool VisitEnter( const TiXmlElement&, const TiXmlAttribute* );

    TiXmlElement* rootNode()
    {
        if( nodes.empty() ) return NULL;
        else return nodes[0];
    }

    std::vector<TiXmlElement* > leaves;
    std::vector<TiXmlElement* > nodes;

};

class DiscoverDisplayFlagsVisitor : public TiXmlVisitor
{
public:

    virtual bool VisitEnter( const TiXmlElement&, const TiXmlAttribute* );
    ~DiscoverDisplayFlagsVisitor();
    std::map<const TiXmlElement*,sofa::core::visual::DisplayFlags*> map_displayFlags;

};

void createVisualStyleVisitor(TiXmlElement* leaf, const std::map<const TiXmlElement*,sofa::core::visual::DisplayFlags*>& map_displayFlags);

void removeShowAttributes(TiXmlElement* node);

void convert_false_to_neutral(sofa::core::visual::DisplayFlags& flags);


}
}


#endif
