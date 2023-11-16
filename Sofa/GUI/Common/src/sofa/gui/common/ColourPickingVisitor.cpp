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
#include <sofa/gui/common/ColourPickingVisitor.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/Node.h>

#include <sofa/component/collision/geometry/SphereModel.h>
#include <sofa/component/collision/geometry/TriangleModel.h>

#if SOFA_GUI_COMMON_HAVE_SOFA_GL == 1
#include <sofa/gl/gl.h>
#include <sofa/gl/BasicShapes.h>
#endif // SOFA_GUI_COMMON_HAVE_SOFA_GL  == 1

namespace sofa::gui::common
{

using namespace sofa::type;
using namespace sofa::component::collision::geometry;
using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using sofa::gui::component::performer::BodyPicked;

namespace
{
constexpr float threshold = std::numeric_limits<float>::min();
}

void decodeCollisionElement(const RGBAColor& colour, BodyPicked& body)
{

    if( colour[0] > threshold || colour[1] > threshold || colour[2] > threshold  ) // make sure we are not picking the background...
    {
        type::vector<core::CollisionModel*> listCollisionModel;
        if (body.body) body.body->getContext()->get<core::CollisionModel>(&listCollisionModel,BaseContext::SearchRoot);
        const std::size_t totalCollisionModel = listCollisionModel.size();
        const int indexListCollisionModel = (int) ( colour[0] * (float)totalCollisionModel + 0.5) - 1;
        if(indexListCollisionModel >= 0 && indexListCollisionModel < (int)listCollisionModel.size())
        {
            body.body = listCollisionModel[indexListCollisionModel];
            body.indexCollisionElement = (unsigned int) ( colour[1] * body.body->getSize() + 0.5 );
        }
    }
    else
    {
        body.body = nullptr;
        body.indexCollisionElement= 0;
    }

}

void decodePosition(BodyPicked& body, const RGBAColor& colour, const TriangleCollisionModel<sofa::defaulttype::Vec3Types>* model,
        const unsigned int index)
{

    if( colour[0] > threshold || colour[1] > threshold || colour[2] > threshold  )
    {
        const sofa::component::collision::geometry::Triangle t(const_cast<TriangleCollisionModel<sofa::defaulttype::Vec3Types>*>(model),index);
        body.point = (t.p1()*colour[0]) + (t.p2()*colour[1]) + (t.p3()*colour[2]) ;

    }

}

void decodePosition(BodyPicked& body, const RGBAColor& colour, const SphereCollisionModel<sofa::defaulttype::Vec3Types> *model,
        const unsigned int index)
{
    SOFA_UNUSED(colour);

    const Sphere s(const_cast<SphereCollisionModel<sofa::defaulttype::Vec3Types>*>(model),index);
    body.point = s.center();
}

simulation::Visitor::Result ColourPickingVisitor::processNodeTopDown(simulation::Node* node)
{

    for_each(this, node, node->collisionModel, &ColourPickingVisitor::processCollisionModel);
    return RESULT_CONTINUE;
}

void decodeCollisionElement(const sofa::type::Vec4f& colour, BodyPicked& body)
{
    decodeCollisionElement(sofa::type::RGBAColor::fromVec4(colour), body);
}

void decodePosition(BodyPicked& body, const sofa::type::Vec4f& colour, const TriangleCollisionModel<sofa::defaulttype::Vec3Types>* model,
        const unsigned int index)
{
    decodePosition(body, sofa::type::RGBAColor::fromVec4(colour), model, index);
}

void decodePosition(BodyPicked& body, const sofa::type::Vec4f& colour, const SphereCollisionModel<sofa::defaulttype::Vec3Types> *model,
        const unsigned int index)
{
    decodePosition(body, sofa::type::RGBAColor::fromVec4(colour), model, index);
}

void ColourPickingVisitor::processCollisionModel(simulation::Node*  node , core::CollisionModel* o)
{
    using namespace core::objectmodel;
    TriangleCollisionModel<sofa::defaulttype::Vec3Types>* tmodel = nullptr;
    SphereCollisionModel<sofa::defaulttype::Vec3Types>*   smodel = nullptr;
    if((tmodel = dynamic_cast<TriangleCollisionModel<sofa::defaulttype::Vec3Types>*>(o)) != nullptr )
        processTriangleModel(node,tmodel);
    if( (smodel = dynamic_cast<SphereCollisionModel<sofa::defaulttype::Vec3Types>*>(o) ) != nullptr )
        processSphereModel(node,smodel);
}

void ColourPickingVisitor::processTriangleModel(simulation::Node * node, sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types> * tmodel)
{
#if SOFA_GUI_COMMON_HAVE_SOFA_GL  == 1
    using namespace sofa::core::collision;
    using namespace sofa::defaulttype;
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    type::vector<Vec3> points;
    type::vector<Vec3> normals;
    std::vector<sofa::type::RGBAColor> colours;
    type::vector<core::CollisionModel*> listCollisionModel;
    type::vector<core::CollisionModel*>::iterator iter;
    float r,g;

    const int size = tmodel->getSize();

    node->get< sofa::core::CollisionModel >( &listCollisionModel, BaseContext::SearchRoot);
    iter = std::find(listCollisionModel.begin(), listCollisionModel.end(), tmodel);
    const std::size_t totalCollisionModel = listCollisionModel.size();
    const int indexCollisionModel = std::distance(listCollisionModel.begin(),iter ) + 1 ;

    switch( method )
    {
    case ENCODE_COLLISIONELEMENT:
        r = (float)indexCollisionModel / (float)totalCollisionModel;
        for( int i=0 ; i<size; i++)
        {
            g = (float)i / (float)size;
            sofa::component::collision::geometry::Triangle t(tmodel,i);
            normals.push_back(t.n() );
            points.push_back( t.p1() );
            points.push_back( t.p2() );
            points.push_back( t.p3() );
            colours.emplace_back( r,g,0,1 );
            colours.emplace_back( r,g,0,1 );
            colours.emplace_back( r,g,0,1 );
        }
        break;
    case ENCODE_RELATIVEPOSITION:
        for( int i=0 ; i<size; i++)
        {
            sofa::component::collision::geometry::Triangle t(tmodel,i);
            normals.push_back(t.n() );
            points.push_back( t.p1() );
            points.push_back( t.p2() );
            points.push_back( t.p3() );
            colours.emplace_back( 1,0,0,1 );
            colours.emplace_back( 0,1,0,1 );
            colours.emplace_back( 0,0,1,1 );
        }
        break;
    default: assert(false);
    }
    vparams->drawTool()->drawTriangles(points,normals,colours);
#else
    SOFA_UNUSED(node);
    SOFA_UNUSED(tmodel);
#endif // SOFA_GUI_COMMON_HAVE_SOFA_GL  == 1
}

void ColourPickingVisitor::processSphereModel(simulation::Node * node, sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types> * smodel)
{
#if SOFA_GUI_COMMON_HAVE_SOFA_GL  == 1
    typedef Sphere::Coord Coord;

    if( method == ENCODE_RELATIVEPOSITION ) return; // we pick the center of the sphere.

    type::vector<core::CollisionModel*> listCollisionModel;

    node->get< sofa::core::CollisionModel >( &listCollisionModel, BaseContext::SearchRoot);
    const std::size_t totalCollisionModel = listCollisionModel.size();
    const type::vector<core::CollisionModel*>::iterator iter = std::find(listCollisionModel.begin(), listCollisionModel.end(), smodel);
    const int indexCollisionModel = std::distance(listCollisionModel.begin(),iter ) + 1 ;
    const float red = (float)indexCollisionModel / (float)totalCollisionModel;
    // Check topological modifications

    const int npoints = smodel->getMechanicalState()->getSize();
    std::vector<Vec3> points;
    std::vector<float> radius;
    for (int i=0; i<npoints; i++)
    {
        Sphere t(smodel,i);
        Coord p = t.p();
        points.push_back(p);
        radius.push_back((float)t.r());
    }
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    float ratio;
    for( int i=0; i<npoints; i++)
    {
        Vec3 p = points[i];

        glPushMatrix();
        ratio = (float)i / (float)npoints;
        glColor4f(red,ratio,0,1);
		sofa::gl::drawSphere(p, radius[i], 32, 16);

        glPopMatrix();
    }
#else
    SOFA_UNUSED(node);
    SOFA_UNUSED(smodel);
#endif // SOFA_GUI_COMMON_HAVE_SOFA_GL  == 1
}

} // namespace sofa::gui::common
