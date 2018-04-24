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
 * CompositingVisualLoop.h
 *
 *  Created on: 16 janv. 2012
 *      Author: Jeremy Ringard
 */

#ifndef SOFA_SIMULATION_COMPOSITINGVISUALLOOP_H
#define SOFA_SIMULATION_COMPOSITINGVISUALLOOP_H
#include "config.h"

#include <sofa/simulation/DefaultVisualManagerLoop.h>
#include <sofa/core/visual/VisualParams.h>

#ifdef SOFA_HAVE_GLEW
#include <SofaOpenglVisual/OglShader.h>
#include <SofaOpenglVisual/VisualManagerPass.h>
#endif

#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/Event.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \Compositing visual loop: render multiple passes and composite them into one single rendered frame
 */

class SOFA_OPENGL_VISUAL_API CompositingVisualLoop : public simulation::DefaultVisualManagerLoop
{
public:
    SOFA_CLASS(CompositingVisualLoop,simulation::DefaultVisualManagerLoop);

    ///Files where vertex shader is defined
    sofa::core::objectmodel::DataFileName vertFilename;
    ///Files where fragment shader is defined
    sofa::core::objectmodel::DataFileName fragFilename;

private:

    void traceFullScreenQuad();
    void defaultRendering(sofa::core::visual::VisualParams* vparams);

protected:
    CompositingVisualLoop(simulation::Node* gnode = NULL);

    virtual ~CompositingVisualLoop();

public:

    virtual void init() override;
    virtual void initVisual() override;
    virtual void drawStep(sofa::core::visual::VisualParams* vparams) override;
};

} // namespace visualmodel

} // namespace component

} //sofa
#endif  /* SOFA_SIMULATION_COMPOSITINGVISUALLOOP_H */
