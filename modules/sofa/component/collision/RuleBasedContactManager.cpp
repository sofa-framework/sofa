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
#include <sofa/component/collision/RuleBasedContactManager.h>
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
    : rules(initData(&rules, "rules", "Ordered list of rules, each with a triplet of strings.\n"
            "The first two define either the name of the collision model, its group number, or * meaning any model.\n"
            "The last string define the response algorithm to use for contacts matched by this rule.\n"
            "Rules are applied in the order they are specified. If none match a given contact, the default response is used.\n"))
{
}

RuleBasedContactManager::~RuleBasedContactManager()
{
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
            return it->response; // rule it matched
    }
    // no rule matched
    return DefaultContactManager::getContactResponse(model1, model2);
}

} // namespace collision

} // namespace component

} // namespace sofa

