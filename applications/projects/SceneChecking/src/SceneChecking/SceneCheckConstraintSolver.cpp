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
#include "SceneCheckConstraintSolver.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/ComponentChange.h>
#include <sofa/simulation/SceneCheckMainRegistry.h>


namespace sofa::_scenechecking_
{

const bool SceneCheckConstraintSolverRegistered = sofa::simulation::SceneCheckMainRegistry::addToRegistry(SceneCheckConstraintSolver::newSPtr());

using sofa::core::objectmodel::BaseContext;
using sofa::core::objectmodel::BaseObjectDescription;
using sofa::core::ObjectFactory;


SceneCheckConstraintSolver::SceneCheckConstraintSolver()
{
    /// Add a callback to be n
    ObjectFactory::getInstance()->setBeforeCreateCallback([this](BaseContext* o, BaseObjectDescription *arg) {
        const std::string typeNameInScene = arg->getAttribute("type", "");
        if ( typeNameInScene == "GenericConstraintSolver" )
        {
            const std::string solverType = arg->getAttribute("resolutionMethod", "ProjectedGaussSeidel");
            std::string newName = "ConstraintSolver";
            if (solverType == "ProjectedGaussSeidel")
                newName = std::string("ProjectedGaussSeidel") + newName;
            else if (solverType == "UnbuiltGaussSeidel")
                newName = std::string("UnbuiltGaussSeidel") + newName;
            else if (solverType == "NonsmoothNonlinearConjugateGradient")
                newName = std::string("NNCG") + newName;
            else
                newName = std::string("ProjectedGaussSeidel") + newName;
            m_targetDescription["type"] = newName;
            std::vector<std::string> attributes;
            arg->getAttributeList(attributes);
            for (const auto & attr : attributes)
            {
                if (attr == "resolutionMethod" || attr == "type")
                    continue;

                const std::string attrValue = arg->getAttribute(attr,"");

                if (attr == "newtonIterations")
                    m_targetDescription["maxIterations"] =  attrValue;
                else if (! attrValue.empty())
                    m_targetDescription[attr] = attrValue;
            }

            //Now correct args to run simulation anyway
            const std::string newtonValue = arg->getAttribute("newtonIterations","");
            if (!newtonValue.empty())
                arg->setAttribute("maxIterations", newtonValue);

            arg->removeAttribute("resolutionMethod");
            arg->removeAttribute("newtonIterations");
            arg->setAttribute("type",newName);
        }

    });
}

SceneCheckConstraintSolver::~SceneCheckConstraintSolver()
{

}

const std::string SceneCheckConstraintSolver::getName()
{
    return "SceneCheckConstraintSolver";
}

const std::string SceneCheckConstraintSolver::getDesc()
{
    return "Check if a Component has been created using an Alias.";
}

void SceneCheckConstraintSolver::doPrintSummary()
{
    if ( this->m_targetDescription.empty() )
    {
        return;
    }
    
    std::stringstream usingAliasesWarning;
    usingAliasesWarning << "This scene is using a GenericConstraintSolver which is now an abstract class."<< msgendl;
    usingAliasesWarning << "The real types are now specific to the solver. The object has been automatically replaced for you." <<msgendl;
    usingAliasesWarning << "To remove this warning an insure future compatibility, please replace the GenericConstraintSolver creation call to :"<< msgendl;

    usingAliasesWarning << "  (python) : addObject(\""<<m_targetDescription["type"]<<"\"";
    for (const auto & arg : m_targetDescription)
    {
        if (arg.first == "type")
            continue;
        usingAliasesWarning <<", "<< arg.first<<"=\"" << arg.second<<"\"";
    }
    usingAliasesWarning << ")" << msgendl;


    usingAliasesWarning << "  (xml)    : <"<<m_targetDescription["type"];
    for (const auto & arg : m_targetDescription)
    {
        if (arg.first == "type")
            continue;
        usingAliasesWarning <<" "<< arg.first<<"=\"" << arg.second<<"\"";
    }
    usingAliasesWarning << " />" << msgendl;

    msg_warning(this->getName()) << usingAliasesWarning.str();

}

} // namespace sofa::_scenechecking_
