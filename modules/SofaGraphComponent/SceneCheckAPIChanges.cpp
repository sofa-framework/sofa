/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include <sofa/version.h>
#include <string>
#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base ;

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>

#include "SceneCheckAPIChanges.h"
#include "RequiredPlugin.h"

#include "APIVersion.h"
using sofa::component::APIVersion ;

namespace sofa
{
namespace simulation
{
namespace _scenecheckapichange_
{

//// Here are the message we will all the time reproduct.
#define DEPMSG 0
#define DEPREP 1
std::map<std::string, std::string> s_commonMessages =
{
    {"deprecated-17.12", " has been deprecated since sofa 17.12. Please consider updating your scene as using "
                         " deprecated component may result in poor performance and undefined behavior."
                         " If this component is crucial to you please report that to sofa-dev@ so we can  "
                         " reconsider this component for future re-integration. "
    },
    {"removed-17.12", " has been removed since sofa 17.12. Please consider updating your scene."
                      " If this component is crucial to you please report that to sofa-dev@ so we can  "
                      " reconsider this component for future re-integration. "
    },
} ;


////// Here is the list of component that are removed or deprecated.
/// Component name, the error message to use among
std::map<std::string, std::vector<std::string>> deprecatedComponents =
{
    {"WashingMachineForceField", {"deprecated-17.12"}},

    {"CatmullRomSplineMapping.cpp", {"deprecated-17.12"}},
    {"CenterPointMechanicalMapping.cpp", {"deprecated-17.12"}},
    {"CurveMapping.cpp", {"deprecated-17.12"}},
    {"ExternalInterpolationMapping.cpp", {"deprecated-17.12"}},
    {"ProjectionToLineMapping", {"deprecated-17.12"}},
    {"ProjectionToTargetLineMapping_test", {"deprecated-17.12"}},
    {"ProjectionToPlaneMapping", {"deprecated-17.12"}},
    {"ProjectionToTargetPlaneMapping_test.cpp", {"deprecated-17.12"}},

    /// SofaUserInteraction
    {"AddRecordedCameraPerformer", {"deprecated-17.12"}},
    {"ArticulatedHierarchyBVHController", {"deprecated-17.12"}},
    {"ArticulatedHierarchyController", {"deprecated-17.12"}},
    {"CuttingPoint", {"deprecated-17.12"}},
    {"DisabledContact", {"deprecated-17.12"}},
    {"EdgeSetController", {"deprecated-17.12"}},
    {"FixParticlePerformer", {"deprecated-17.12"}},
    {"GraspingManager", {"deprecated-17.12"}},
    {"InciseAlongPathPerformer", {"deprecated-17.12"}},
    {"InterpolationController", {"deprecated-17.12"}},
    {"NodeToggleController", {"deprecated-17.12"}}
};

const std::string SceneCheckAPIChange::getName()
{
    return "SceneCheckAPIChange";
}

const std::string SceneCheckAPIChange::getDesc()
{
    return "Check for each component that the behavior have not changed since reference version of sofa.";
}

void SceneCheckAPIChange::doInit(Node* node)
{
    std::stringstream version;
    version << SOFA_VERSION / 10000 << "." << SOFA_VERSION / 100 % 100;
    m_currentApiLevel = version.str();

    APIVersion* apiversion {nullptr} ;
    /// 1. Find if there is an APIVersion component in the scene. If there is none, warn the user and set
    /// the version to 17.06 (the last version before it was introduced). If there is one...use
    /// this component to request the API version requested by the scene.
    node->getTreeObject(apiversion) ;
    if(!apiversion)
    {
        msg_info("SceneChecker") << "The 'APIVersion' directive is missing in the current scene. Switching to the default APIVersion level '"<< m_selectedApiLevel <<"' " ;
    }
    else
    {
        m_selectedApiLevel = apiversion->getApiLevel() ;
    }
}

void SceneCheckAPIChange::doCheckOn(Node* node)
{
    for (auto& object : node->object )
    {
        if(m_selectedApiLevel != m_currentApiLevel && m_changesets.find(m_selectedApiLevel) != m_changesets.end())
        {
            for(auto& hook : m_changesets[m_selectedApiLevel])
            {
                hook(object.get());
            }
        }

    }
}


void SceneCheckAPIChange::installDefaultChangeSets()
{
    addHookInChangeSet("17.06", [](Base* o){
        if(o->getClassName() == "RestShapeSpringsForceField" && o->findData("external_rest_shape")->isSet())
            msg_warning(o) << "RestShapeSpringsForceField have changed since 17.06. The parameter 'external_rest_shape' is now a Link. To fix your scene you need to add and '@' in front of the provided path. See PR#315" ;
    }) ;

    addHookInChangeSet("17.06", [](Base* o){
        if(o->getClassName() == "BoxStiffSpringForceField" )
            msg_warning(o) << "BoxStiffSpringForceField have changed since 17.06. To use the old behavior you need to set parameter 'forceOldBehavior=true'" ;
    }) ;

    addHookInChangeSet("17.06", [](Base* o){
        if( deprecatedComponents.find( o->getClassName() ) != deprecatedComponents.end() )
        {
            auto& msg = deprecatedComponents[o->getClassName()] ;
            std::string str = msg[DEPMSG];

            /// Replace the string by the default one.
            if( s_commonMessages.find( str ) != s_commonMessages.end() ){
                str = s_commonMessages[str] ;
            }

            msg_warning(o) << o->getClassName()
                               << str
                               << msg[DEPREP] ;
        }
    }) ;
}

void SceneCheckAPIChange::addHookInChangeSet(const std::string& version, ChangeSetHookFunction fct)
{
    m_changesets[version].push_back(fct) ;
}


} // _scenecheckapichange_

} // namespace simulation

} // namespace sofa

