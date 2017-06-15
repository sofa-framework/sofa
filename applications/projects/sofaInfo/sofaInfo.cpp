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
#include <SofaSimulationTree/init.h>
#include <SofaSimulationTree/TreeSimulation.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>
#include <sofa/core/ObjectFactory.h>

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int /*argc*/, char** argv)
{
    sofa::simulation::tree::init();
    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();

    if (argv[1] == NULL)
    {
        std::cout << "Usage: sofaInfo FILE" << std::endl;
        return -1;
    }

    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());

    sofa::simulation::Node::SPtr groot = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(argv[1]));

    if (groot==NULL)
    {
        groot = sofa::simulation::getSimulation()->createNewGraph("");
    }

    sofa::helper::vector<sofa::core::objectmodel::Base*> objects;
    std::set<std::string> classNames;
    std::set<std::string> targets;

    groot->getTreeObjects<sofa::core::objectmodel::Base>(&objects);

    // get the classes and targets of the scene
    for (unsigned int i=0; i<objects.size(); i++)
    {
        sofa::core::ObjectFactory::ClassEntry& entry = sofa::core::ObjectFactory::getInstance()->getEntry(objects[i]->getClassName());
        if (entry.creatorMap.empty())
        {
            classNames.insert(entry.className);

            sofa::core::ObjectFactory::CreatorMap::iterator it = entry.creatorMap.find(objects[i]->getTemplateName());
            if (it != entry.creatorMap.end() && *it->second->getTarget())
            {
                targets.insert(it->second->getTarget());
            }
        }
    }

    std::set<std::string>::const_iterator it = classNames.begin();
    std::set<std::string>::const_iterator end = classNames.end();
    std::cout << "=== CLASSES ===" << std::endl;
    while (it != end)
    {
        std::cout << (*it) << std::endl;
        it++;
    }

    it = targets.begin();
    end = targets.end();
    std::cout << std::endl << "=== TARGETS ===" << std::endl;
    while (it != end)
    {
        std::cout << (*it) << std::endl;
        it++;
    }

    if (groot!=NULL)
        sofa::simulation::getSimulation()->unload(groot);

    sofa::simulation::tree::cleanup();
    return 0;
}
