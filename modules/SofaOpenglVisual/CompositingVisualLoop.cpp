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
/*
 * CompositingVisualLoop.cpp
 *
 *  Created on: 16 janv. 2012
 *      Author: Jeremy Ringard
 */


#include <SofaOpenglVisual/CompositingVisualLoop.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaBaseVisual/VisualStyle.h>
#include <sofa/core/visual/DisplayFlags.h>
#include <sofa/simulation/VisualVisitor.h>


#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

SOFA_DECL_CLASS(CompositingVisualLoop)

int CompositingVisualLoopClass = core::RegisterObject("Visual loop enabling multipass rendering. Needs multiple fbo data and a compositing shader")
        .add< CompositingVisualLoop >()
        ;

CompositingVisualLoop::CompositingVisualLoop(simulation::Node* _gnode)
    : simulation::DefaultVisualManagerLoop(_gnode),
      vertFilename(initData(&vertFilename, (std::string) "shaders/compositing.vert", "vertFilename", "Set the vertex shader filename to load")),
      fragFilename(initData(&fragFilename, (std::string) "shaders/compositing.frag", "fragFilename", "Set the fragment shader filename to load"))
{
    //assert(gRoot);
}

CompositingVisualLoop::~CompositingVisualLoop()
{}

void CompositingVisualLoop::initVisual()
{}

void CompositingVisualLoop::init()
{
    if (!gRoot)
        gRoot = dynamic_cast<simulation::Node*>(this->getContext());
}

//should not be called if scene file is well formed
void CompositingVisualLoop::defaultRendering(sofa::core::visual::VisualParams* vparams)
{
    vparams->pass() = sofa::core::visual::VisualParams::Std;
    sofa::simulation::VisualDrawVisitor act ( vparams );
    gRoot->execute ( &act );
    vparams->pass() = sofa::core::visual::VisualParams::Transparent;
    sofa::simulation::VisualDrawVisitor act2 ( vparams );
    gRoot->execute ( &act2 );
}

void CompositingVisualLoop::drawStep(sofa::core::visual::VisualParams* vparams)
{
    if ( !gRoot ) return;

    sofa::core::visual::tristate renderingState;
    //vparams->displayFlags().setShowRendering(false);
    component::visualmodel::VisualStyle::SPtr visualStyle = NULL;
    gRoot->get(visualStyle);
    const sofa::core::visual::DisplayFlags &backupFlags = vparams->displayFlags();
    const sofa::core::visual::DisplayFlags &currentFlags = visualStyle->displayFlags.getValue();
    vparams->displayFlags() = sofa::core::visual::merge_displayFlags(backupFlags, currentFlags);
    renderingState = vparams->displayFlags().getShowRendering();

    if (!(vparams->displayFlags().getShowRendering()))
    {
        dmsg_info() << "Advanced Rendering is OFF" ;

        defaultRendering(vparams);
        return;
    }
    else{
        dmsg_info() << "Advanced Rendering is ON" ;
    }
    //should not happen: the compositing loop relies on one or more rendered passes done by the VisualManagerPass component
    if (gRoot->visualManager.empty())
    {
        serr << "CompositingVisualLoop: no VisualManagerPass found. Disable multipass rendering." << sendl;
        defaultRendering(vparams);
    }

    //rendering sequence: call each VisualManagerPass elements, then composite the frames
    else
    {
#ifdef SOFA_HAVE_GLEW
        if (renderingState == sofa::core::visual::tristate::false_value || renderingState == sofa::core::visual::tristate::neutral_value) return;

        sofa::simulation::Node::Sequence<core::visual::VisualManager>::iterator begin = gRoot->visualManager.begin(), end = gRoot->visualManager.end(), it;
        //preDraw sequence
        it=begin;
        for (it = begin; it != end; ++it)
        {
            (*it)->preDrawScene(vparams);
            VisualManagerPass* currentVMP=dynamic_cast<VisualManagerPass*>(*it);
            if( currentVMP!=NULL && !currentVMP->isPrerendered())
            {
                msg_info() << "final pass is "<<currentVMP->getName()<< "end of predraw loop"  ;
                break;
            }
        }
        //Draw sequence
        bool rendered = false; // true if a manager did the rendering
        for (it = begin; it != end; ++it)
            if ((*it)->drawScene(vparams))	{ rendered = true; 	break;	}

        if (!rendered) // do the rendering
        {
            msg_error() << "No visualManager rendered the scene. Please make sure the final visualManager(Secondary)Pass has a renderToScreen=\"true\" attribute" ;
        }

        //postDraw sequence
        sofa::simulation::Node::Sequence<core::visual::VisualManager>::reverse_iterator rbegin = gRoot->visualManager.rbegin(), rend = gRoot->visualManager.rend(), rit;
        for (rit = rbegin; rit != rend; ++rit)
            (*rit)->postDrawScene(vparams);

        // cleanup OpenGL state
        for (int i=0; i<4; ++i)
        {
            glActiveTexture(GL_TEXTURE0+i);
            glDisable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        glDisable(GL_LIGHTING);
        glUseProgramObjectARB(0);
        glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBufferARB(GL_ARRAY_BUFFER, 0);
        //glViewport(vparams->viewport()[0],vparams->viewport()[1],vparams->viewport()[2],vparams->viewport()[3]);
#endif
    }
}


} // namespace visualmodel
} // namespace component
} //sofa
