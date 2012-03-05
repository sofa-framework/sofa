/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/common/DefaultVisualManagerLoop.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/simulation/common/UpdateMappingEndEvent.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>


#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{

namespace simulation
{

SOFA_DECL_CLASS(DefaultVisualManagerLoop);

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

    /*
    Okay, the next line looks weird and useless. Actually I put it here because
    VisualDrawVisitorMultiPass does exist, but is not accessed in this project.
    Therefore the compilator skip the class and raise some linking error in
    sofa_opengl_visual (coz this other project needs that class).
    That's why I've had to create an instance of VisualDrawVisitorMultiPass, just to tell
    the compilator to link it correctly.
    Now you know the full story. By the way, if you have any other solution (compilation option?)
    feel free to correct it :) (I'm quite sure that any other solution would be much more elegant
    than this but after all, that was a funny story huh?)
    */
    VisualDrawVisitorMultiPass whyTheHeckDoIExistIMNotEvenUsedThatDoesNTMakeSense();
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
    simulation::Visitor::printNode(std::string("UpdateVisual"));
#endif
    sofa::helper::AdvancedTimer::begin("UpdateVisual");
    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    gRoot->execute<UpdateMappingVisitor>(params);
    sofa::helper::AdvancedTimer::step("UpdateMappingEndEvent");
    {
        double dt=gRoot->getDt();
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        gRoot->execute ( act );
    }
    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");
    gRoot->execute<VisualUpdateVisitor>(params);
    sofa::helper::AdvancedTimer::end("UpdateVisual");
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode(std::string("UpdateVisual"));
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

        gRoot->execute ( &act );
        vparams->pass() = sofa::core::visual::VisualParams::Transparent;
        VisualDrawVisitor act2 ( vparams );
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
            gRoot->execute ( &act );
            vparams->pass() = sofa::core::visual::VisualParams::Transparent;
            VisualDrawVisitor act2 ( vparams );
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
