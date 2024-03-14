/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/gui/qt/GLPickHandler.h>


#include <sofa/simulation/InitVisitor.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/gl/gl.h>
#include <sofa/gui/component/performer/ComponentMouseInteraction.h>
#include <sofa/component/collision/geometry/SphereModel.h>
#include <sofa/component/collision/geometry/TriangleModel.h>
#include <sofa/component/collision/response/contact/RayContact.h>
#include <sofa/component/setting/MouseButtonSetting.h>

#include <iostream>
#include <limits>


namespace sofa::gui::qt
{

using namespace sofa::component::collision::geometry;
using namespace sofa::gui::common;
using namespace sofa::gui::component::performer;

GLPickHandler::GLPickHandler(double defaultLength)
    : Inherit(defaultLength)
    , _fboAllocated(false)
    , _fbo(true,true,true,false,0)
{

}


GLPickHandler::~GLPickHandler()
{


}

void GLPickHandler::allocateSelectionBuffer(int width, int height)
{
    /*called when shift key is pressed */
    assert(_fboAllocated == false );

    static bool firstTime=true;
    if (firstTime)
    {
        _fboParams.depthInternalformat = GL_DEPTH_COMPONENT24;
#if defined(GL_VERSION_3_0)
        if (GLEW_VERSION_3_0)
        {
            _fboParams.colorInternalformat = GL_RGBA32F;
        }
        else
#endif //  (GL_VERSION_3_0)
        {
            _fboParams.colorInternalformat = GL_RGBA16;
        }
        _fboParams.colorFormat         = GL_RGBA;
        _fboParams.colorType           = GL_FLOAT;

        _fbo.setFormat(_fboParams);
        firstTime=false;
    }
    _fbo.init(width,height);

    _fboAllocated = true;
}

void GLPickHandler::destroySelectionBuffer()
{
    /*called when shift key is released */
    assert(_fboAllocated);

    _fbo.destroy();

    _fboAllocated = false;
}

//WARNING: do not use this method with Ogre
BodyPicked GLPickHandler::findCollisionUsingColourCoding(const type::Vec3& origin,
        const type::Vec3& direction)
{
    assert(_fboAllocated);
    BodyPicked result;

    result.dist =  0;
    type::RGBAColor color;
    const int x = mousePosition.x;
    const int y = mousePosition.screenHeight - mousePosition.y;
    TriangleCollisionModel<defaulttype::Vec3Types>* tmodel;
    SphereCollisionModel<defaulttype::Vec3Types>* smodel;
    _fbo.start();
    if(renderCallback)
    {
        renderCallback->render(ColourPickingVisitor::ENCODE_COLLISIONELEMENT );
        glReadPixels(x,y,1,1,_fboParams.colorFormat,_fboParams.colorType, &color[0]);
        decodeCollisionElement(color,result);
        renderCallback->render(ColourPickingVisitor::ENCODE_RELATIVEPOSITION );
        glReadPixels(x,y,1,1,_fboParams.colorFormat,_fboParams.colorType, &color[0]);
        if( ( tmodel = dynamic_cast<TriangleCollisionModel<defaulttype::Vec3Types>*>(result.body) ) != nullptr )
        {
            decodePosition(result,color,tmodel,result.indexCollisionElement);
        }
        if( ( smodel = dynamic_cast<SphereCollisionModel<defaulttype::Vec3Types>*>(result.body)) != nullptr)
        {
            decodePosition(result, color,smodel,result.indexCollisionElement);
        }
        result.rayLength = (result.point-origin)*direction;
    }
    _fbo.stop();

    return result;
}

} // namespace sofa::gui::qt
