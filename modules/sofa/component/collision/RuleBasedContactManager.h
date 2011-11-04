/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_RULEBASEDCONTACTMANAGER_H
#define SOFA_COMPONENT_COLLISION_RULEBASEDCONTACTMANAGER_H

#include <sofa/component/collision/DefaultContactManager.h>

namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_MISC_COLLISION_API RuleBasedContactManager : public DefaultContactManager
{
public:
    SOFA_CLASS(RuleBasedContactManager, DefaultContactManager);

    class Rule
    {
    public:
        std::string name1;
        int group1;
        std::string name2;
        int group2;
        std::string response;

        inline friend std::istream& operator >> ( std::istream& in, Rule& r )
        {
            in >> r.name1 >> r.name2 >> r.response;
            if (!r.name1.empty() && r.name1.find_first_not_of("-0123456789") == std::string::npos)
            {
                r.group1 = atoi(r.name1.c_str());
                r.name1.clear();
            }
            else
                r.group1 = 0;
            if (!r.name2.empty() && r.name2.find_first_not_of("-0123456789") == std::string::npos)
            {
                r.group2 = atoi(r.name2.c_str());
                r.name2.clear();
            }
            else
                r.group2 = 0;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const Rule& r )
        {
            if (r.name1.empty())
                out << r.group1;
            else
                out << r.name1;
            out << ' ';
            if (r.name2.empty())
                out << r.group2;
            else
                out << r.name2;
            out << ' ';
            out << r.response<<'\n';
            return out;
        }
        bool match(core::CollisionModel* model1, core::CollisionModel* model2) const
        {
            if (!name1.empty())
            {
                if (name1 != "*" && name1 != model1->getName())
                    return false;
            }
            else
            {
                if (group1 != model1->getGroup())
                    return false;
            }
            if (!name2.empty())
            {
                if (name2 != "*" && name2 != model2->getName())
                    return false;
            }
            else
            {
                if (group2 != model2->getGroup())
                    return false;
            }
            return true;
        }
    };

    Data< helper::vector<Rule> > rules;

    virtual std::string getContactResponse(core::CollisionModel* model1, core::CollisionModel* model2);

protected:
    RuleBasedContactManager();
    ~RuleBasedContactManager();
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
