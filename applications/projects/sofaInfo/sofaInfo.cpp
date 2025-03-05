/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/init.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/common/init.h>
#include <sofa/simulation/graph/init.h>

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int /*argc*/, char** argv)
{
    sofa::simulation::common::init();
    sofa::simulation::graph::init();
    sofa::component::init();

    if (argv[1] == nullptr)
    {
        std::cout << "Usage: sofaInfo FILE" << std::endl;
        return -1;
    }

    sofa::simulation::Node::SPtr groot = sofa::simulation::node::load(argv[1]);

    if (groot == nullptr)
    {
        groot = sofa::simulation::getSimulation()->createNewGraph("");
    }

    sofa::type::vector<sofa::core::objectmodel::Base*> objects;
    std::set<std::string> classNames;
    std::set<std::string> targets;

    groot->getTreeObjects<sofa::core::objectmodel::Base>(&objects);

    // get the classes and targets of the scene
    for (const auto& object : objects)
    {
        sofa::core::ObjectFactory::ClassEntry& entry = sofa::core::ObjectFactory::getInstance()->getEntry(object->getClassName());
        if (!entry.creatorMap.empty())
        {
            classNames.insert(entry.className);

            const auto it = entry.creatorMap.find(object->getTemplateName());
            if (it != entry.creatorMap.end() && *it->second->getTarget())
            {
                targets.insert(it->second->getTarget());
            }
        }
    }

    std::cout << "=== CLASSES ===" << std::endl;
    std::cout << sofa::helper::join(classNames, "\n");

    std::cout << std::endl << "=== TARGETS ===" << std::endl;
    std::cout << sofa::helper::join(targets, "\n");

    if (groot != nullptr)
    {
        sofa::simulation::node::unload(groot);
    }

    sofa::simulation::common::cleanup();
    sofa::simulation::graph::cleanup();
    return 0;
}
