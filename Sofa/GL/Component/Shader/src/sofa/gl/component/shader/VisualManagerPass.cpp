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
#include <sofa/gl/component/shader/VisualManagerPass.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::gl::component::shader
{

using namespace sofa::gl;
using namespace simulation;
using namespace core::visual;

void registerVisualManagerPass(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Render pass element: render the relevant tagged objects in a FBO.")
        .add< VisualManagerPass >());
}

VisualManagerPass::VisualManagerPass()
    : factor(initData(&factor, 1.0f, "factor","set the resolution factor for the output pass. default value:1.0")),
      renderToScreen(initData(&renderToScreen, "renderToScreen", "if true, this pass will be displayed on screen (only one renderPass in the scene must be defined as renderToScreen)")),
      outputName(initData(&outputName, "outputName","name the output texture"))
{
    if(factor.getValue()==0.0f)
    {
        msg_warning("VisualManagerPass") << this->getName()<<":\"factor\" attribute shall not be null. Using 1.0 instead...";
        factor.setValue(1.0f);
    }

    prerendered=false;
    //listen by default, in order to get the keys to activate shadows
    if(!f_listening.isSet())
        f_listening.setValue(true);
}

std::string VisualManagerPass::getOutputName()
{
    if(outputName.getValue().empty())
        return this->getName();
    else
        return outputName.getValue();
}

VisualManagerPass::~VisualManagerPass()
{}

bool VisualManagerPass::checkMultipass(sofa::core::objectmodel::BaseContext* con)
{
    const sofa::gl::component::shader::CompositingVisualLoop* isMultipass=nullptr;
    isMultipass= con->core::objectmodel::BaseContext::get<sofa::gl::component::shader::CompositingVisualLoop>();
    return (isMultipass!=nullptr);
}

void VisualManagerPass::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    multiPassEnabled=checkMultipass(context);
}

/* herited from VisualModel */
void VisualManagerPass::doInitVisual(const core::visual::VisualParams*)
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    passWidth = (GLint) ((float)viewport[2]*factor.getValue());
    passHeight = (GLint)((float)viewport[3] * factor.getValue());

    fbo = std::unique_ptr<sofa::gl::FrameBufferObject>(
                new FrameBufferObject(true, true, true, true));
    fbo->init(passWidth, passHeight);
}

void VisualManagerPass::fwdDraw(core::visual::VisualParams* )
{
}

void VisualManagerPass::bwdDraw(core::visual::VisualParams* )
{
}

void VisualManagerPass::draw(const core::visual::VisualParams* )
{
}
/***************************/

void VisualManagerPass::preDrawScene(VisualParams* vp)
{
    if(renderToScreen.getValue() || (!multiPassEnabled))
        return;

    //const VisualParams::Viewport& viewport = vp->viewport();
    fbo->setSize(passWidth, passHeight);
    fbo->start();

    glViewport(0,0,passWidth,passHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glEnable(GL_DEPTH_BUFFER_BIT);

    //render opaque meshes
    vp->pass() = sofa::core::visual::VisualParams::Std;
    VisualDrawVisitor act ( vp );
    act.setTags(this->getTags());
    act.execute ( getContext() );
    //render transparent meshes
    vp->pass() = sofa::core::visual::VisualParams::Transparent;
    VisualDrawVisitor act2 ( vp );
    act2.setTags(this->getTags());
    act2.execute ( getContext() );

    fbo->stop();
    prerendered=true;
}

bool VisualManagerPass::drawScene(VisualParams* vp)
{
    if(!multiPassEnabled)
        return false;

    if(renderToScreen.getValue())
    {
        glViewport(0,0,passWidth,passHeight);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glEnable(GL_DEPTH_BUFFER_BIT);

        //render opaque meshes
        vp->pass() = sofa::core::visual::VisualParams::Std;
        VisualDrawVisitor act ( vp );
        act.setTags(this->getTags());
        act.execute ( getContext() );

        //render transparent meshes
        vp->pass() = sofa::core::visual::VisualParams::Transparent;
        VisualDrawVisitor act2 ( vp );
        act2.setTags(this->getTags());
        act2.execute ( getContext() );

        return true;
    }
    else
        return false;
}

void VisualManagerPass::postDrawScene(VisualParams* /*vp*/)
{
    prerendered=false;
}


//keyboard event management. Not sure what I'm gonna do with that for the moment, but I'm quite sure it should be useful in the future
void VisualManagerPass::handleEvent(sofa::core::objectmodel::Event* /*event*/)
{
}

bool VisualManagerPass::hasFilledFbo()
{
    return prerendered;
}


} // namespace sofa::gl::component::shader
