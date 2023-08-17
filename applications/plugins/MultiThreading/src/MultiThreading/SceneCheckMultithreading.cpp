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
#include <MultiThreading/SceneCheckMultithreading.h>
#include <sofa/simulation/Node.h>

#include <MultiThreading/ParallelImplementationsRegistry.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/SceneCheckMainRegistry.h>

namespace multithreading::_scenechecking_
{

const bool SceneCheckMultithreadingRegistered = sofa::simulation::SceneCheckMainRegistry::addToRegistry(SceneCheckMultithreading::newSPtr());

std::shared_ptr<SceneCheckMultithreading> SceneCheckMultithreading::newSPtr()
{
    return std::make_shared<SceneCheckMultithreading>();
}

const std::string SceneCheckMultithreading::getName()
{
    return "SceneCheckMultithreading";
}

const std::string SceneCheckMultithreading::getDesc()
{
    return "Check if there are opportunities to use multithreading, and potentially improve the performances,"
           " by replacing components by their parallel implementations";
}

void SceneCheckMultithreading::doInit(sofa::simulation::Node* node)
{
    SOFA_UNUSED(node);

    m_summary.clear();
}

void SceneCheckMultithreading::doCheckOn(sofa::simulation::Node* node)
{
    if (node == nullptr)
        return;

    for (auto& object : node->object )
    {
        if (object)
        {
            const auto parallelImplementation =
                multithreading::ParallelImplementationsRegistry::findParallelImplementation(object->getClassName());

            if (!parallelImplementation.empty())
            {
                if (sofa::core::ObjectFactory::getInstance()->hasCreator(parallelImplementation))
                {
                    const auto& entry = sofa::core::ObjectFactory::getInstance()->getEntry(parallelImplementation);
                    auto it = entry.creatorMap.find(object->getTemplateName());
                    if (it != entry.creatorMap.end())
                    {
                        std::string seq = object->getClassName();
                        if (!object->getTemplateName().empty())
                        {
                            seq += "[" + object->getTemplateName() + "]";
                        }
                        std::string par = entry.className;
                        if (!it->first.empty())
                        {
                            par += "[" + it->first + "]";
                        }
                        m_summary.insert({seq, par });
                    }
                }
                else
                {
                    msg_error(object.get()) << "The component has a equivalent parallel implementation '"
                        << parallelImplementation << "' but it cannot be found in the object factory";
                }
            }
        }
    }
}

void SceneCheckMultithreading::doPrintSummary()
{
    if (!m_summary.empty())
    {
        std::stringstream ss;
        for (const auto& [seq, par] : m_summary)
        {
            ss << "\t" << seq << " -> " << par << msgendl;
        }
        msg_advice(this->getName()) << "This scene is using components implemented sequentially while "
            << "a parallel implementation is available. Using the parallel implementation may improve "
            "the performances. Here is the list of sequential components in your scene and "
            "their parallel equivalent: " << msgendl << ss.str();
    }
}
}


