/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/gui/ColourPickingVisitor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/system/gl.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{
namespace gui
{

using namespace sofa::component::collision;
using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;
using namespace sofa::core::collision;

namespace
{
const float threshold = std::numeric_limits<float>::min();
}

void decodeCollisionElement(const sofa::defaulttype::Vec4f colour,  sofa::component::collision::BodyPicked& body)
{

    if( colour[0] > threshold || colour[1] > threshold || colour[2] > threshold  ) // make sure we are not picking the background...
    {

        helper::vector<core::CollisionModel*> listCollisionModel;
        //sofa::simulation::getSimulation()->getContext()->get<core::CollisionModel>(&listCollisionModel,BaseContext::SearchRoot);
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
        body.body = NULL;
        body.indexCollisionElement= 0;
    }

}

void decodePosition(BodyPicked& body, const sofa::defaulttype::Vec4f colour, const TriangleModel* model,
        const unsigned int index)
{

    if( colour[0] > threshold || colour[1] > threshold || colour[2] > threshold  )
    {
        component::collision::Triangle t(const_cast<TriangleModel*>(model),index);
        body.point = (t.p1()*colour[0]) + (t.p2()*colour[1]) + (t.p3()*colour[2]) ;

    }

}

void decodePosition(BodyPicked& body, const sofa::defaulttype::Vec4f /*colour*/, const SphereModel *model,
        const unsigned int index)
{
    Sphere s(const_cast<SphereModel*>(model),index);
    body.point = s.center();
}

simulation::Visitor::Result ColourPickingVisitor::processNodeTopDown(simulation::Node* node)
{

#ifdef SOFA_SUPPORT_MOVING_FRAMES
    glPushMatrix();
    double glMatrix[16];
    node->getPositionInWorld().writeOpenGlMatrix(glMatrix);
    glMultMatrixd( glMatrix );
#endif
    for_each(this, node, node->collisionModel, &ColourPickingVisitor::processCollisionModel);
#ifdef SOFA_SUPPORT_MOVING_FRAMES
    glPopMatrix();
#endif
    return RESULT_CONTINUE;
}

void ColourPickingVisitor::processCollisionModel(simulation::Node*  node , core::CollisionModel* o)
{
    using namespace core::objectmodel;
    TriangleModel* tmodel = NULL;
    SphereModel*   smodel = NULL;
    if((tmodel = dynamic_cast<TriangleModel*>(o)) != NULL )
        processTriangleModel(node,tmodel);
    if( (smodel = dynamic_cast<SphereModel*>(o) ) != NULL )
        processSphereModel(node,smodel);
}

void ColourPickingVisitor::processTriangleModel(simulation::Node * node, sofa::component::collision::TriangleModel * tmodel)
{
#ifndef SOFA_NO_OPENGL
    using namespace sofa::core::collision;
    using namespace sofa::defaulttype;
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    helper::vector<Vector3> points;
    helper::vector<Vector3> normals;
    helper::vector< Vec<4,float> > colours;
    helper::vector<core::CollisionModel*> listCollisionModel;
    helper::vector<core::CollisionModel*>::iterator iter;
    float r,g;

    int size = tmodel->getSize();

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
            component::collision::Triangle t(tmodel,i);
            normals.push_back(t.n() );
            points.push_back( t.p1() );
            points.push_back( t.p2() );
            points.push_back( t.p3() );
            colours.push_back( Vec<4,float>(r,g,0,1) );
            colours.push_back( Vec<4,float>(r,g,0,1) );
            colours.push_back( Vec<4,float>(r,g,0,1) );
        }
        break;
    case ENCODE_RELATIVEPOSITION:
        for( int i=0 ; i<size; i++)
        {
            component::collision::Triangle t(tmodel,i);
            normals.push_back(t.n() );
            points.push_back( t.p1() );
            points.push_back( t.p2() );
            points.push_back( t.p3() );
            colours.push_back( Vec<4,float>(1,0,0,1) );
            colours.push_back( Vec<4,float>(0,1,0,1) );
            colours.push_back( Vec<4,float>(0,0,1,1) );
        }
        break;
    default: assert(false);
    }
    vparams->drawTool()->drawTriangles(points,normals,colours);
#endif /* SOFA_NO_OPENGL */
}

void ColourPickingVisitor::processSphereModel(simulation::Node * node, sofa::component::collision::SphereModel * smodel)
{
#ifndef SOFA_NO_OPENGL
    typedef Sphere::Coord Coord;

    if( method == ENCODE_RELATIVEPOSITION ) return; // we pick the center of the sphere.

    helper::vector<core::CollisionModel*> listCollisionModel;

    node->get< sofa::core::CollisionModel >( &listCollisionModel, BaseContext::SearchRoot);
    const std::size_t totalCollisionModel = listCollisionModel.size();
    helper::vector<core::CollisionModel*>::iterator iter = std::find(listCollisionModel.begin(), listCollisionModel.end(), smodel);
    const int indexCollisionModel = std::distance(listCollisionModel.begin(),iter ) + 1 ;
    float red = (float)indexCollisionModel / (float)totalCollisionModel;
    // Check topological modifications

    const int npoints = smodel->getMechanicalState()->getSize();
    std::vector<Vector3> points;
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
        Vector3 p = points[i];

        glPushMatrix();
        ratio = (float)i / (float)npoints;
        glColor4f(red,ratio,0,1);
		helper::gl::drawSphere(p, radius[i], 32, 16);

        glPopMatrix();
    }
#endif /* SOFA_NO_OPENGL */
}



}
}
