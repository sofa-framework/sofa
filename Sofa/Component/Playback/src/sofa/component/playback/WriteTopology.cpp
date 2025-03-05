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
#include <sofa/component/playback/WriteTopology.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/BaseMapping.h>

namespace sofa::component::playback
{

using namespace defaulttype;

void registerWriteTopology(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Write topology containers information to file at each timestep.")
        .add< WriteTopology >());
}

WriteTopologyCreator::WriteTopologyCreator(const core::ExecParams* params)
    :Visitor(params)
    ,sceneName("")
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , recordContainers(true)
    , recordShellContainers(false)
    , createInMapping(false)
    , counterWriteTopology(0)
{
}

WriteTopologyCreator::WriteTopologyCreator(const std::string &n, bool _writeContainers, bool _writeShellContainers, bool _createInMapping, const core::ExecParams* params, int c)
    :Visitor(params)
    , sceneName(n)
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , recordContainers(_writeContainers)
    , recordShellContainers(_writeShellContainers)
    , createInMapping(_createInMapping)
    , counterWriteTopology(c)
{
}


//Create a Write Topology component each time a BaseMeshTopology is found
simulation::Visitor::Result WriteTopologyCreator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::core::topology::BaseMeshTopology* topo = dynamic_cast<sofa::core::topology::BaseMeshTopology *>( gnode->getMeshTopologyLink());
    if (!topo)   return simulation::Visitor::RESULT_CONTINUE;
    //We have a meshTopology
    addWriteTopology(topo, gnode);
    return simulation::Visitor::RESULT_CONTINUE;
}


void WriteTopologyCreator::addWriteTopology(core::topology::BaseMeshTopology* topology, simulation::Node* gnode)
{
    const sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping;
    context->get(mapping);
    if ( createInMapping || mapping == nullptr)
    {
        sofa::component::playback::WriteTopology::SPtr wt;
        context->get(wt, this->subsetsToManage, core::objectmodel::BaseContext::Local);

        if (wt.get() == nullptr)
        {
            wt = sofa::core::objectmodel::New<WriteTopology>();
            gnode->addObject(wt);
            wt->d_writeContainers.setValue(recordContainers);
            wt->d_writeShellContainers.setValue(recordShellContainers);
            for (const auto& subset : this->subsetsToManage)
            {
                wt->addTag(subset);
            }
        }

        std::ostringstream ofilename;
        ofilename << sceneName << "_" << counterWriteTopology << "_" << topology->getName()  << "_topology" << extension ;

        wt->d_filename.setValue(ofilename.str()); wt->init(); wt->f_listening.setValue(true);  //Activated at init

        ++counterWriteTopology;

    }
}



//if state is true, we activate all the write states present in the scene.
simulation::Visitor::Result WriteTopologyActivator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::component::playback::WriteTopology *wt = gnode->get< sofa::component::playback::WriteTopology >(this->subsetsToManage);
    if (wt != nullptr) { changeStateWriter(wt);}
    return simulation::Visitor::RESULT_CONTINUE;
}

void WriteTopologyActivator::changeStateWriter(sofa::component::playback::WriteTopology* wt)
{
    if (!state) wt->reset();
    wt->f_listening.setValue(state);
}

} // namespace sofa::component::playback
