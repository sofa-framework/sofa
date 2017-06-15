/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <SofaMiscCollision/RuleBasedContactManager.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(RuleBasedContactManager)

int RuleBasedContactManagerClass = core::RegisterObject("Create different response to the collisions based on a set of rules")
        .add< RuleBasedContactManager >()
        .addAlias("RuleBasedCollisionResponse")
        ;

RuleBasedContactManager::RuleBasedContactManager()
    : d_variables(initData(&d_variables, "variables", "Define a list of variables to be used inside the rules"))
    , rules(initData(&rules, "rules", "Ordered list of rules, each with a triplet of strings.\n"
            "The first two define either the name of the collision model, its group number, or * meaning any model.\n"
            "The last string define the response algorithm to use for contacts matched by this rule.\n"
            "Rules are applied in the order they are specified. If none match a given contact, the default response is used.\n"))
{
}

RuleBasedContactManager::~RuleBasedContactManager()
{
    for(std::map<std::string,Data<std::string>*>::iterator it = variablesData.begin(),
        itend = variablesData.end(); it != itend; ++it)
    {
        //this->removeData(it->second);
        delete it->second;
    }
}

void RuleBasedContactManager::createVariableData ( std::string variable )
{
    Data<std::string>* d = new Data<std::string>("", true, false);
    d->setName(variable);
    std::size_t sep = variable.find('_');
    if (sep != std::string::npos)
    {
        // store group names in static set so that pointer to string content is kept valid
        static std::set<std::string> groupNames;
        const std::string& group = *groupNames.insert(variable.substr(0,sep)).first;
        d->setGroup(group.c_str());
    }
    else
    {
        d->setGroup("Variables");
    }
    this->addData(d);
    variablesData[variable] = d;
}

std::string RuleBasedContactManager::replaceVariables(std::string response)
{
    std::string res;
    std::string::size_type next = 0;
    while(next < response.size())
    {
        std::string::size_type var = response.find('$', next);
        if (var == std::string::npos) // no more variables
        {
            res.append(response.substr(next));
            break;
        }
        else
        {
            if (var > next)
                res.append(response.substr(next,var-next));
            std::string::size_type varEnd = response.find('$', var+1);
            if (varEnd == std::string::npos) // parse error
            {
                serr << "Error parsing variables in rule " << response << sendl;
                res.append(response.substr(var));
                break;
            }
            else
            {
                std::string varname = response.substr(var+1,varEnd-var-1);
                std::string varvalue;
                std::map<std::string,Data<std::string>*>::const_iterator it = variablesData.find(varname);
                if (it == variablesData.end())
                {
                    serr << "Unknown variables " << varname << sendl;
                }
                else
                {
                    varvalue = it->second->getValue();
                }
                res.append(varvalue);
                next = varEnd+1;
            }
        }
    }
    msg_info() << "Output response string : " << res ;
    return res;
}


std::string RuleBasedContactManager::getContactResponse(core::CollisionModel* model1, core::CollisionModel* model2)
{
    // Response locally defined on a given CollisionModel takes priority (necessary for the mouse for instance)
    std::string response1 = model1->getContactResponse();
    std::string response2 = model2->getContactResponse();
    if (!response1.empty()) return response1;
    else if (!response2.empty()) return response2;

    const helper::vector<Rule>& r = rules.getValue();
    for (helper::vector<Rule>::const_iterator it = r.begin(), itend = r.end(); it != itend; ++it)
    {
        if (it->match(model1, model2) || it->match(model2, model1))
            return replaceVariables(it->response); // rule it matched
    }
    // no rule matched
    return replaceVariables(DefaultContactManager::getContactResponse(model1, model2));
}

void RuleBasedContactManager::parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    const char* v = arg->getAttribute(d_variables.getName().c_str());
    if (v)
    {
        std::istringstream variablesStr(v);
        std::string var;
        while(variablesStr >> var)
        {
            createVariableData(var);
        }
    }
    Inherit1::parse(arg);
}

} // namespace collision

} // namespace component

} // namespace sofa

