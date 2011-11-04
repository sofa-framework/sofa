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


#ifndef SOFA_CONFIGURATIONPARSER_H
#define SOFA_CONFIGURATIONPARSER_H

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

namespace sofa
{

namespace gui
{

namespace qt
{


enum TYPE_CONDITION {OPTION,ARCHI};

class CONDITION
{
public:
    CONDITION(TYPE_CONDITION t, bool p, std::string o):type(t),presence(p),option(o)
    {}
    bool operator== (const CONDITION& other)
    {
        return type==other.type &&
                presence == other.presence &&
                option   == other.option;
    }
    bool operator!= (const CONDITION& other)
    {
        return type!=other.type ||
                presence != other.presence ||
                option   != other.option;
    }
    TYPE_CONDITION type;
    bool presence;
    std::string option;
};


class DEFINES
{
public:
    DEFINES(bool b, std::string n, std::string d, std::string c, bool t):value(b),name(n),description(d), category(c), typeOption(t)
    {
    };

    bool operator== (const DEFINES& other)
    {
        if (typeOption)
            return name == other.name && category == other.category;
        else
        {
            return name == other.name && category == other.category && description == other.description;
        }
    }


    void addConditions(std::vector< CONDITION > c)
    {
        for (unsigned int i=0; i<c.size(); ++i)
        {
            std::vector< CONDITION >::iterator found= std::find(conditions.begin(), conditions.end(), c[i]);
            if (found == conditions.end()) conditions.push_back(c[i]);
        }
    }

    friend std::ostream& operator << (std::ostream& out, const DEFINES& val)
    {
        out << "[" << val.category << "] -> " << "(" << val.name << ", " << val.description << ")" << std::endl;
        return out;
    }

    bool value;
    std::string name;
    std::string description;
    std::string category;
    std::vector< CONDITION > conditions;

    bool typeOption;
};

class ConfigurationParser
{
public:
    static void Parse(std::ifstream &in, std::vector<DEFINES>  &listOptions);

protected:
    static void removeInitialCharacter(std::string &s, char c);
    static void removeFinalCharacter(std::string &s, char c);
    static void removeComment(std::string &s);

    static void processDescription(std::string &description, std::size_t pos);
    static void processOption(std::string &name, bool &activated, std::size_t pos);
    static void processTextOption(std::string &description, std::string &name, bool &activated, std::size_t pos);
    static void processCondition(std::string &description, bool &presence, TYPE_CONDITION &type, std::size_t pos);
    static void processCategory(std::string &description);
};


}
}
}
#endif
