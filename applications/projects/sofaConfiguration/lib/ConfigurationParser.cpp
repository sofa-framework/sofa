/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "ConfigurationParser.h"

#include <algorithm>
#include <fstream>



namespace sofa
{

namespace gui
{

namespace qt
{


using sofa::gui::qt::DEFINES;
using sofa::gui::qt::CONDITION;
using sofa::gui::qt::TYPE_CONDITION;
using sofa::gui::qt::OPTION;
using sofa::gui::qt::ARCHI;

void ConfigurationParser::removeInitialCharacter(std::string &s, char c)
{
    unsigned int i=0;
    for (; i<s.size(); ++i)
    {
        if (s[i] != c) break;
    }
    s=s.substr(i);
}

void ConfigurationParser::removeFinalCharacter(std::string &s, char c)
{
    int i=s.size()-1;
    for (; i>=0; --i)
    {
        if (s[i] != c) break;
    }
    s.resize(i+1);
}

void ConfigurationParser::removeComment(std::string &s)
{
    std::size_t found=s.find('#');
    if (found != std::string::npos) s.resize(found);
}

std::string currentCategory;

void ConfigurationParser::processDescription(std::string &description, std::size_t pos)
{
    description=description.substr(pos+10);
}

void ConfigurationParser::processOption(std::string &name, bool &activated, std::size_t pos)
{
    std::string line=name;
    removeInitialCharacter(line,' ');
    if (line[0] == '#') activated=false;
    else                activated=true;

    name = name.substr(pos+10);
    removeInitialCharacter(name,' ');
    removeComment(name);
    removeFinalCharacter(name,' ');
}

void ConfigurationParser::processTextOption(std::string &description, std::string &name, bool &activated, std::size_t pos)
{
    removeInitialCharacter(description,' ');
    if (description[0] == '#')
    {
        activated=false;
        description=description.substr(1);
    }
    else                activated=true;

    name=description;
    name.resize(pos+1);
    removeInitialCharacter(name,' ');
    removeFinalCharacter(name,' ');

    description = description.substr(pos+1);
    removeInitialCharacter(description,' ');
    removeFinalCharacter(description,' ');
}

void ConfigurationParser::processCondition(std::string &description, bool &presence, TYPE_CONDITION &type, std::size_t pos)
{
    std::size_t posContains=description.find("contains(");
    std::size_t boolNot=description.find('!');
    if (posContains!=std::string::npos)
    {
        type=OPTION;
        std::size_t separator=description.find(',');
        std::string type=description;  type.resize(separator); type=type.substr(posContains+9);
        std::string option=description.substr(separator+1);
        separator=option.find(')'); option.resize(separator);
        if (type=="DEFINES") description=option;
    }
    else
    {
        type=ARCHI;
        description.resize(pos);
    }

    removeInitialCharacter(description,' ');
    removeFinalCharacter(description,' ');

    if (boolNot == std::string::npos || boolNot > pos) presence=true;
    else presence=false;

}

void ConfigurationParser::processCategory(std::string &description)
{
    removeInitialCharacter(description,'#');
    removeInitialCharacter(description,' ');
    removeFinalCharacter(description,'#');
    removeFinalCharacter(description,' ');
}

void ConfigurationParser::Parse(std::ifstream &in, std::vector<DEFINES>  &listOptions)
{
    enum State {NONE, CATEGORY};
    int STATE=NONE;

    std::string description;
    std::vector< CONDITION > conditions;
    std::string text;
    while (std::getline(in, text))
    {
        removeInitialCharacter(text,' ');
        std::size_t found;
        //Three keywords: Uncomment, DEFINES +=, contains(
        switch (STATE)
        {
        case CATEGORY:

            found = text.find("#############################");
            if (found != std::string::npos)
            {
                STATE=NONE;
                continue;
            }
            else
            {
                processCategory(text);
                currentCategory=text;
            }

            break;
        case NONE:

            found = text.find("Uncomment");
            if (found != std::string::npos)
            {
                STATE=NONE;
                processDescription(text, found);
                description=text;
                continue;
            }
            found = text.find("DEFINES +=");
            if (found != std::string::npos)
            {
                STATE=NONE;
                bool activated=false;
                processOption(text, activated, found);
                DEFINES op(activated, text, description, currentCategory, true);
                std::vector< DEFINES >::iterator it = std::find(listOptions.begin(), listOptions.end(), op);
                if (it != listOptions.end())
                {
                    it->description=description;
                    it->category=currentCategory;
                    it->value=activated;
                    it->addConditions(conditions);
                }
                else
                {
                    listOptions.push_back(op);
                    listOptions.back().addConditions(conditions);
                }
                continue;
            }

            //FIND {
            found = text.find("{");
            if (found != std::string::npos)
            {

                TYPE_CONDITION type;
                bool presence;
                processCondition(text, presence,type,found);
                conditions.push_back(CONDITION(type,presence,text));
                STATE=NONE;
                continue;
            }
            found = text.find("#############################");
            if (found != std::string::npos)
            {
                STATE=CATEGORY;
                continue;
            }
            if (text[0]=='}')
            {
                conditions.pop_back();
                STATE=NONE;
                continue;
            }
            found = text.find('=');
            if (found != std::string::npos           &&
                text.find("<=") == std::string::npos &&
                text.find(">=") == std::string::npos    )
            {
                std::string name;
                bool presence;
                processTextOption(text, name, presence, found);
                DEFINES op(presence,name,text,currentCategory,false);
                std::vector< DEFINES >::iterator it = std::find(listOptions.begin(), listOptions.end(), op);
                if (it != listOptions.end())
                {
                    it->description=text;
                    it->category=currentCategory;
                    it->value=presence;
                    it->addConditions(conditions);
                }
                else
                {
                    listOptions.push_back(op);
                    listOptions.back().addConditions(conditions);
                }
            }
            else
            {
                removeInitialCharacter(text,'#');
                removeInitialCharacter(text,' ');
                description+="\n"+text;
            }
            continue;
        }

    }
}
}
}
}
