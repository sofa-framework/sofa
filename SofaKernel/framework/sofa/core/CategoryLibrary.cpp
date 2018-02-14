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

#include "CategoryLibrary.h"

#include <sofa/core/objectmodel/ContextObject.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/collision/CollisionAlgorithm.h>
#include <sofa/core/collision/Pipeline.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/topology/BaseTopologyObject.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/core/loader/BaseLoader.h>

namespace sofa
{

namespace core
{


//-------------------------------------------------------------------------------------------------------
CategoryLibrary::CategoryLibrary( const std::string &categoryName): name(categoryName)
{
}


ComponentLibrary *CategoryLibrary::addComponent(const std::string &componentName, ClassEntry::SPtr entry, const std::vector< std::string > &exampleFiles)
{
    //Special case of Mapping and MechanicalMapping
    bool isMechanicalMapping = (name == "MechanicalMapping");
    bool isMapping           = (name == "Mapping");

    ComponentLibrary* component = createComponent(componentName, entry, exampleFiles);

    //Add the corresponding templates
    std::map<std::string, Creator::SPtr>::iterator itTemplate;

    //It exists Mappings only Mechanical or only Visual. So, we must add the component if only a creator is available for the current category
    bool componentCreationPossible=false;
    //read all the template possible, and remove unused (for Mapping processing)
    std::list<std::string> templates;
    for (itTemplate=entry->creatorMap.begin(); itTemplate!= entry->creatorMap.end(); ++itTemplate)
    {
        const std::string &templateName = itTemplate->first;
        //If the component corresponds to a MechanicalMapping, we must remove the template related to the visual mapping
        if (isMechanicalMapping)
        {
            const std::string nonMechanical = templateName.substr(0,7);
            if (nonMechanical == "Mapping") continue;
        }
        //If the component corresponds to a Mapping, we must remove the template related to the Mechanical Mapping
        else if (isMapping)
        {
            const std::string mechanical    = templateName.substr(0,17);
            if (mechanical == "MechanicalMapping") continue;
        }
        componentCreationPossible=true;
        //component->addTemplate(itTemplate->first);
        if (templateName == (entry->defaultTemplate.empty() ? std::string("Vec3d") : entry->defaultTemplate))
            templates.push_front(templateName); // make sure the default template is first
        else
            templates.push_back(templateName);
    }
    for (std::list<std::string>::const_iterator it = templates.begin(); it != templates.end(); ++it)
        component->addTemplate(*it);
    component->endConstruction();

    //If no constructor is available, we delete the component
    if (!componentCreationPossible)
    {
        delete component;
        component=NULL;
    }
    else
        components.push_back(component);

    return component;
}

void CategoryLibrary::endConstruction()
{
}


const ComponentLibrary *CategoryLibrary::getComponent( const std::string &categoryName) const
{
    for (VecComponentIterator it=components.begin(); it != components.end(); ++it)
    {
        if ((*it)->getName().find(categoryName) != std::string::npos)
            return *it;
    }
    return NULL;
}



void CategoryLibrary::getCategories(const objectmodel::BaseClass* mclass,
                                    std::vector<std::string>& v)
{
    if (mclass->hasParent(objectmodel::ContextObject::GetClass()))
        v.push_back("ContextObject");
    if (mclass->hasParent(visual::VisualModel::GetClass()))
        v.push_back("VisualModel");
    if (mclass->hasParent(BehaviorModel::GetClass()))
        v.push_back("BehaviorModel");
    if (mclass->hasParent(CollisionModel::GetClass()))
        v.push_back("CollisionModel");
    if (mclass->hasParent(behavior::BaseMechanicalState::GetClass()))
        v.push_back("MechanicalState");
    // A Mass is a technically a ForceField, but we don't want it to appear in the ForceField category
    if (mclass->hasParent(behavior::BaseForceField::GetClass()) && !mclass->hasParent(behavior::BaseMass::GetClass()))
        v.push_back("ForceField");
    if (mclass->hasParent(behavior::BaseInteractionForceField::GetClass()))
        v.push_back("InteractionForceField");
    if (mclass->hasParent(behavior::BaseProjectiveConstraintSet::GetClass()))
        v.push_back("ProjectiveConstraintSet");
    if (mclass->hasParent(behavior::BaseConstraintSet::GetClass()))
        v.push_back("ConstraintSet");
    if (mclass->hasParent(BaseMapping::GetClass()))
        v.push_back("Mapping");
    if (mclass->hasParent(DataEngine::GetClass()))
        v.push_back("Engine");
    if (mclass->hasParent(topology::TopologicalMapping::GetClass()))
        v.push_back("TopologicalMapping");
    if (mclass->hasParent(behavior::BaseMass::GetClass()))
        v.push_back("Mass");
    if (mclass->hasParent(behavior::OdeSolver::GetClass()))
        v.push_back("OdeSolver");
    if (mclass->hasParent(behavior::ConstraintSolver::GetClass()))
        v.push_back("ConstraintSolver");
    if (mclass->hasParent(behavior::BaseConstraintCorrection::GetClass()))
        v.push_back("ConstraintSolver");
    if (mclass->hasParent(behavior::LinearSolver::GetClass()))
        v.push_back("LinearSolver");
    if (mclass->hasParent(behavior::BaseAnimationLoop::GetClass()))
        v.push_back("AnimationLoop");
    // Just like Mass and ForceField, we don't want TopologyObject to appear in the Topology category
    if (mclass->hasParent(topology::Topology::GetClass()) && !mclass->hasParent(topology::BaseTopologyObject::GetClass()))
        v.push_back("Topology");
    if (mclass->hasParent(topology::BaseTopologyObject::GetClass()))
        v.push_back("TopologyObject");
    if (mclass->hasParent(behavior::BaseController::GetClass()))
        v.push_back("Controller");
    if (mclass->hasParent(loader::BaseLoader::GetClass()))
        v.push_back("Loader");
    if (mclass->hasParent(collision::CollisionAlgorithm::GetClass()))
        v.push_back("CollisionAlgorithm");
    if (mclass->hasParent(collision::Pipeline::GetClass()))
        v.push_back("CollisionAlgorithm");
    if (mclass->hasParent(collision::Intersection::GetClass()))
        v.push_back("CollisionAlgorithm");
    if (mclass->hasParent(objectmodel::ConfigurationSetting::GetClass()))
        v.push_back("ConfigurationSetting");
    if (v.empty())
        v.push_back("_Miscellaneous");
}


}
}
