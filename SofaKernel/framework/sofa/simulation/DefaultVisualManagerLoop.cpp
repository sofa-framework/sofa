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
#include <sofa/simulation/DefaultVisualManagerLoop.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/VisualVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>


#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{

namespace simulation
{

SOFA_DECL_CLASS(DefaultVisualManagerLoop)

int DefaultVisualManagerLoopClass = core::RegisterObject("The simplest Visual Loop Manager, created by default when user do not put on scene")
        .add< DefaultVisualManagerLoop >()
        ;

DefaultVisualManagerLoop::DefaultVisualManagerLoop(simulation::Node* _gnode)
    : Inherit()
    , gRoot(_gnode)
{
    //assert(gRoot);
}

DefaultVisualManagerLoop::~DefaultVisualManagerLoop()
{

}

void DefaultVisualManagerLoop::init()
{
    if (!gRoot)
        gRoot = dynamic_cast<simulation::Node*>(this->getContext());
}


void DefaultVisualManagerLoop::initStep(sofa::core::ExecParams* params)
{
    if ( !gRoot ) return;
    gRoot->execute<VisualInitVisitor>(params);
    // Do a visual update now as it is not done in load() anymore
    /// \todo Separate this into another method?
    gRoot->execute<VisualUpdateVisitor>(params);
}

void DefaultVisualManagerLoop::updateStep(sofa::core::ExecParams* params)
{
    if ( !gRoot ) return;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("UpdateVisual");
#endif
    sofa::helper::AdvancedTimer::begin("UpdateVisual");

    // 03/09/14: mapping update should already be performed by animation
//    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
//    gRoot->execute<UpdateMappingVisitor>(params);
//    sofa::helper::AdvancedTimer::step("UpdateMappingEndEvent");
//    {
//        SReal dt=gRoot->getDt();
//        UpdateMappingEndEvent ev ( dt );
//        PropagateEventVisitor act ( params, &ev );
//        gRoot->execute ( act );
//    }
//    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");

    gRoot->execute<VisualUpdateVisitor>(params);
    sofa::helper::AdvancedTimer::end("UpdateVisual");
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("UpdateVisual");
#endif

}

void DefaultVisualManagerLoop::updateContextStep(sofa::core::visual::VisualParams* vparams)
{
    UpdateVisualContextVisitor vis(vparams);
    vis.execute(gRoot);
}

void DefaultVisualManagerLoop::drawStep(sofa::core::visual::VisualParams* vparams)
{
    if ( !gRoot ) return;
    if (gRoot->visualManager.empty())
    {
        vparams->pass() = sofa::core::visual::VisualParams::Std;
        VisualDrawVisitor act ( vparams );
        act.setTags(this->getTags());
        gRoot->execute ( &act );
        vparams->pass() = sofa::core::visual::VisualParams::Transparent;
        VisualDrawVisitor act2 ( vparams );
        act2.setTags(this->getTags());
        gRoot->execute ( &act2 );
    }
    else
    {
        Node::Sequence<core::visual::VisualManager>::iterator begin = gRoot->visualManager.begin(), end = gRoot->visualManager.end(), it;
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
            gRoot->execute ( &act );
            vparams->pass() = sofa::core::visual::VisualParams::Transparent;
            VisualDrawVisitor act2 ( vparams );
            act2.setTags(this->getTags());
            gRoot->execute ( &act2 );
        }
        Node::Sequence<core::visual::VisualManager>::reverse_iterator rbegin = gRoot->visualManager.rbegin(), rend = gRoot->visualManager.rend(), rit;
        for (rit = rbegin; rit != rend; ++rit)
            (*rit)->postDrawScene(vparams);
    }
}

void DefaultVisualManagerLoop::computeBBoxStep(sofa::core::visual::VisualParams* vparams, SReal* minBBox, SReal* maxBBox, bool init)
{
    VisualComputeBBoxVisitor act(vparams);
    if ( gRoot )
        gRoot->execute ( act );
//    cerr<<"DefaultVisualManagerLoop::computeBBoxStep, xm= " << act.minBBox[0] <<", xM= " << act.maxBBox[0] << endl;
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



} // namespace simulation

} // namespace sofa
