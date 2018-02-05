/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaGeneralLoader/ReadTopology.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(ReadTopology)

using namespace defaulttype;

int ReadTopologyClass = core::RegisterObject("Read topology containers informations from file at each timestep")
        .add< ReadTopology >();

ReadTopologyCreator::ReadTopologyCreator(const core::ExecParams* params)
    :Visitor(params)
    , sceneName("")
#ifdef SOFA_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , createInMapping(false)
    , init(true)
    , counterReadTopology(0)
{
}

ReadTopologyCreator::ReadTopologyCreator(const std::string &n, bool _createInMapping, const core::ExecParams* params, bool i, int c)
    :Visitor(params)
    , sceneName(n)
#ifdef SOFA_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , createInMapping(_createInMapping)
    , init(i)
    , counterReadTopology(c)
{
}

//Create a Read Topology component each time a BaseMeshTopology is found
simulation::Visitor::Result ReadTopologyCreator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::core::topology::BaseMeshTopology* topo = gnode->getMeshTopology();
    if (!topo)   return simulation::Visitor::RESULT_CONTINUE;
    //We have a meshTopology
    addReadTopology(topo, gnode);
    return simulation::Visitor::RESULT_CONTINUE;
}

void ReadTopologyCreator::addReadTopology(core::topology::BaseMeshTopology* topology, simulation::Node* gnode)
{
    sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping;
    context->get(mapping);
    if (createInMapping || mapping== NULL)
    {
        sofa::component::misc::ReadTopology::SPtr rt;
        context->get(rt, this->subsetsToManage, core::objectmodel::BaseContext::Local);
        if (rt == NULL)
        {
            rt = sofa::core::objectmodel::New<ReadTopology>();
            gnode->addObject(rt);
            for (core::objectmodel::TagSet::iterator it=this->subsetsToManage.begin(); it != this->subsetsToManage.end(); ++it)
                rt->addTag(*it);
        }

        std::ostringstream ofilename;
        ofilename << sceneName << "_" << counterReadTopology << "_" << topology->getName()  << "_topology" << extension ;

        rt->f_filename.setValue(ofilename.str());  rt->f_listening.setValue(false); //Deactivated only called by extern functions
        if (init) rt->init();

        ++counterReadTopology;
    }
}

///if state is true, we activate all the write states present in the scene.
simulation::Visitor::Result ReadTopologyActivator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::component::misc::ReadTopology *rt = gnode->get< sofa::component::misc::ReadTopology >(this->subsetsToManage);
    if (rt != NULL) { changeTopologyReader(rt);}

    return simulation::Visitor::RESULT_CONTINUE;
}

void ReadTopologyActivator::changeTopologyReader(sofa::component::misc::ReadTopology* rt)
{
    if (!state) rt->reset();
    rt->f_listening.setValue(state);
}


//if state is true, we activate all the write states present in the scene. If not, we activate all the readers.
simulation::Visitor::Result ReadTopologyModifier::processNodeTopDown( simulation::Node* gnode)
{
    using namespace sofa::defaulttype;

    sofa::component::misc::ReadTopology* rt = gnode->get< sofa::component::misc::ReadTopology>(this->subsetsToManage);
    if (rt != NULL) {changeTimeReader(rt);}

    return simulation::Visitor::RESULT_CONTINUE;
}

} // namespace misc

} // namespace component

} // namespace sofa
