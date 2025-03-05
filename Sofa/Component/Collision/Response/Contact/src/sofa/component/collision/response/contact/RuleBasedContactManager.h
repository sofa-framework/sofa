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
#include <sofa/component/collision/response/contact/config.h>

#include <sofa/component/collision/response/contact/CollisionResponse.h>
#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::collision::response::contact
{

class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API RuleBasedContactManager : public CollisionResponse
{
public:
    SOFA_CLASS(RuleBasedContactManager, CollisionResponse);

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
                if (!model1->getGroups().contains(group1) )
                    return false;
            }
            if (!name2.empty())
            {
                if (name2 != "*" && name2 != model2->getName())
                    return false;
            }
            else
            {
                if (!model2->getGroups().contains(group2) )
                    return false;
            }
            return true;
        }
    };

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_COLLISION_RESPONSE_CONTACT()
    sofa::core::objectmodel::lifecycle::RenamedData< type::vector<Rule> > rules;





    Data< std::string > d_variables; ///< Define a list of variables to be used inside the rules
    Data< type::vector<Rule> > d_rules;

    virtual std::string getContactResponse(core::CollisionModel* model1, core::CollisionModel* model2) override;

    void createVariableData ( std::string variable );

    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override;

protected:
    RuleBasedContactManager();
    ~RuleBasedContactManager() override;

    std::map<std::string,Data<std::string>* > variablesData;

    std::string replaceVariables(std::string response);
};

} // namespace sofa::component::collision::response::contact
