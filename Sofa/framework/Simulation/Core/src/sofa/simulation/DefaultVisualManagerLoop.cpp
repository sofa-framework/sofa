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
#include <sofa/simulation/DefaultVisualManagerLoop.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/Node.h>

namespace sofa::core::objectmodel {
    template class sofa::core::objectmodel::SingleLink< sofa::simulation::DefaultVisualManagerLoop, simulation::Node, BaseLink::FLAG_STOREPATH>;
}

namespace sofa::simulation
{

int DefaultVisualManagerLoopClass = core::RegisterObject("The simplest Visual Loop Manager, created by default when user do not put on scene")
        .add< DefaultVisualManagerLoop >()
        ;

DefaultVisualManagerLoop::DefaultVisualManagerLoop() :
    l_node(initLink("targetNode","Link to the scene's node where the rendering will take place"))
{
}

DefaultVisualManagerLoop::~DefaultVisualManagerLoop()
{

}

void DefaultVisualManagerLoop::init()
{
    if (!l_node)
        l_node = dynamic_cast<simulation::Node*>(this->getContext());
}


void DefaultVisualManagerLoop::initStep(sofa::core::ExecParams* params)
{
    if ( !l_node ) return;
    l_node->execute<VisualInitVisitor>(params);
    // Do a visual update now as it is not done in load() anymore
    /// \todo Separate this into another method?
    l_node->execute<VisualUpdateVisitor>(params);
}

void DefaultVisualManagerLoop::updateStep(sofa::core::ExecParams* params)
{
    if ( !l_node ) return;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("UpdateVisual");
#endif

    l_node->execute<VisualUpdateVisitor>(params);
    
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("UpdateVisual");
#endif
}

void DefaultVisualManagerLoop::updateContextStep(sofa::core::visual::VisualParams* vparams)
{
    UpdateVisualContextVisitor vis(vparams);
    vis.execute(l_node);
}

void DefaultVisualManagerLoop::drawStep(sofa::core::visual::VisualParams* vparams)
{
    if ( !l_node ) return;
    if (l_node->visualManager.empty())
    {
        vparams->pass() = sofa::core::visual::VisualParams::Std;
        VisualDrawVisitor act ( vparams );
        act.setTags(this->getTags());
        l_node->execute ( &act );
        vparams->pass() = sofa::core::visual::VisualParams::Transparent;
        VisualDrawVisitor act2 ( vparams );
        act2.setTags(this->getTags());
        l_node->execute ( &act2 );
    }
    else
    {
        Node::Sequence<core::visual::VisualManager>::iterator begin = l_node->visualManager.begin(), end = l_node->visualManager.end(), it;
        for (it = begin; it != end; ++it)
            (*it)->preDrawScene(vparams);
        bool rendered = false; // true if a manager did the rendering
        for (it = begin; it != end; ++it)
            if ((*it)->drawScene(vparams))
            {
                rendered = true;
                break;
            }
        if (!rendered) // do the rendering
        {
            vparams->pass() = sofa::core::visual::VisualParams::Std;

            VisualDrawVisitor act ( vparams );
            act.setTags(this->getTags());
            l_node->execute ( &act );
            vparams->pass() = sofa::core::visual::VisualParams::Transparent;
            VisualDrawVisitor act2 ( vparams );
            act2.setTags(this->getTags());
            l_node->execute ( &act2 );
        }
        Node::Sequence<core::visual::VisualManager>::reverse_iterator rbegin = l_node->visualManager.rbegin(), rend = l_node->visualManager.rend(), rit;
        for (rit = rbegin; rit != rend; ++rit)
            (*rit)->postDrawScene(vparams);
    }
}

void DefaultVisualManagerLoop::computeBBoxStep(sofa::core::visual::VisualParams* vparams, SReal* minBBox, SReal* maxBBox, bool init)
{
    VisualComputeBBoxVisitor act(vparams);
    if ( l_node )
        l_node->execute ( act );

    if (init)
    {
        minBBox[0] = (SReal)(act.minBBox[0]);
        minBBox[1] = (SReal)(act.minBBox[1]);
        minBBox[2] = (SReal)(act.minBBox[2]);
        maxBBox[0] = (SReal)(act.maxBBox[0]);
        maxBBox[1] = (SReal)(act.maxBBox[1]);
        maxBBox[2] = (SReal)(act.maxBBox[2]);
    }
    else
    {
        if ((SReal)(act.minBBox[0]) < minBBox[0] ) minBBox[0] = (SReal)(act.minBBox[0]);
        if ((SReal)(act.minBBox[1]) < minBBox[1] ) minBBox[1] = (SReal)(act.minBBox[1]);
        if ((SReal)(act.minBBox[2]) < minBBox[2] ) minBBox[2] = (SReal)(act.minBBox[2]);
        if ((SReal)(act.maxBBox[0]) > maxBBox[0] ) maxBBox[0] = (SReal)(act.maxBBox[0]);
        if ((SReal)(act.maxBBox[1]) > maxBBox[1] ) maxBBox[1] = (SReal)(act.maxBBox[1]);
        if ((SReal)(act.maxBBox[2]) > maxBBox[2] ) maxBBox[2] = (SReal)(act.maxBBox[2]);
    }
}

} // namespace sofa
