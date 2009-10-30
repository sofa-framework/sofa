/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
/*
 * PostProcessManager.h
 *
 *  Created on: 12 janv. 2009
 *      Author: froy
 */

#ifndef SOFA_COMPONENT_POSTPROCESSMANAGER_H_
#define SOFA_COMPONENT_POSTPROCESSMANAGER_H_

#include <sofa/core/VisualManager.h>
#include <sofa/helper/gl/FrameBufferObject.h>
#include <sofa/component/visualmodel/OglShader.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_COMPONENT_VISUALMODEL_API PostProcessManager : public core::VisualManager
{
public:
    SOFA_CLASS(PostProcessManager,core::VisualModel);

private:
    static const std::string DEPTH_OF_FIELD_VERTEX_SHADER;
    static const std::string DEPTH_OF_FIELD_FRAGMENT_SHADER;
    Data<double> zNear, zFar;
    helper::gl::FrameBufferObject fbo;
    OglShader* dofShader;
    bool postProcessEnabled;

public:
    ///Files where vertex shader is defined
    sofa::core::objectmodel::DataFileName vertFilename;
    ///Files where fragment shader is defined
    sofa::core::objectmodel::DataFileName fragFilename;

    PostProcessManager();
    virtual ~PostProcessManager();

    void init() ;
    void reinit() { };
    void initVisual();
    void update() { };

    void preDrawScene(helper::gl::VisualParameters* vp);
    bool drawScene(helper::gl::VisualParameters* vp);
    void postDrawScene(helper::gl::VisualParameters* vp);

    void handleEvent(sofa::core::objectmodel::Event* event);
};

} //visualmodel

} //component

} //sofa

#endif /* SOFA_COMPONENT_POSTPROCESSMANAGER_H_ */
