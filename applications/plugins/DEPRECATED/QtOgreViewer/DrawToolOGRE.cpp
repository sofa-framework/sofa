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


#include "DrawToolOGRE.h"

#include <OgreRenderOperation.h>
#include <OgreMaterial.h>
#include <Ogre.h>

namespace sofa
{

namespace core
{

namespace visual
{

using namespace defaulttype;

int DrawToolOGRE::mMaterialName = 0;
int DrawToolOGRE::mEntityName = 0;
int DrawToolOGRE::mMeshName = 0;

DrawToolOGRE::DrawToolOGRE():
    pOgreDraw(NULL)
    ,pOgreDrawInternal(NULL)
    ,pOgreDrawSave(NULL)
{
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DrawToolOGRE::~DrawToolOGRE()
{

}

void DrawToolOGRE::setSceneMgr(Ogre::SceneManager *_scnMgr)
{
    pSceneMgr = _scnMgr;
    assert( pSceneMgr );
    while( !sceneNodeStack.empty() )
    {
        sceneNodeStack.pop();
    }
    sceneNodeStack.push(pSceneMgr->getRootSceneNode());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawPoints(const std::vector<Vector3> &points, float size, const Vec<4,float> colour=Vec<4,float>(1.0f,1.0f,1.0f,1.0f))
{
    if (!pOgreDraw) return;

    setMaterial(colour);
    pCurrentMaterial.getPointer()->getTechnique(0)->setPointSize(size);
    pOgreDraw->begin(pCurrentMaterial->getName(), Ogre::RenderOperation::OT_POINT_LIST);
    for (unsigned int i=0; i<points.size(); ++i)
    {
        Vector3 point = points[i];
        pOgreDraw->position(point[0],point[1],point[2]);

    }
    pOgreDraw->end();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawLines(const std::vector<Vector3> &points, float size, const Vec<4,float> colour)
{
    if (!pOgreDraw) return;

    setMaterial(colour);
    //Set Line Width: not possible with Ogre, with current version
    pOgreDraw->begin(pCurrentMaterial->getName(), Ogre::RenderOperation::OT_LINE_LIST);

    for (unsigned int i=0; i<points.size()/2; ++i)
    {
        drawPoint(points[2*i], colour);
        drawPoint(points[2*i+1], colour);
        pOgreDraw->index(2*i);
        pOgreDraw->index(2*i+1);
    }
    pOgreDraw->end();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawLines(const std::vector<Vector3> &points, const std::vector< defaulttype::Vec<2,int> > &index, float size, const Vec<4,float> colour=Vec<4,float>(1.0f,1.0f,1.0f,1.0f))
{
    if (!pOgreDraw) return;

    setMaterial(colour);
    pCurrentMaterial.getPointer()->getTechnique(0)->setPointSize(size);
    pOgreDraw->begin(pCurrentMaterial->getName(), Ogre::RenderOperation::OT_LINE_LIST);

    for (unsigned int i=0; i<points.size(); ++i)
    {
        drawPoint(points[i], colour);
    }

    for (unsigned int i=0; i<index.size(); ++i)
    {
        pOgreDraw->index(index[i][0]);
        pOgreDraw->index(index[i][1]);
    }
    pOgreDraw->end();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawTriangles(const std::vector<Vector3> &points, const Vec<4,float> colour)
{
    if (!pOgreDraw) return;
    setMaterial(colour);
    pOgreDraw->begin(pCurrentMaterial->getName(), Ogre::RenderOperation::OT_TRIANGLE_LIST);

    for (unsigned int i=0; i<points.size()/3; ++i)
    {
        const Vector3& a = points[ 3*i+0 ];
        const Vector3& b = points[ 3*i+1 ];
        const Vector3& c = points[ 3*i+2 ];
        Vector3 n = cross((b-a),(c-a));
        n.normalize();
        drawPoint(a,n,colour);
        drawPoint(b,n,colour);
        drawPoint(c,n,colour);
        pOgreDraw->triangle(3*i, 3*i+1, 3*i+2);
    }
    pOgreDraw->end();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawTriangles(const std::vector<Vector3> &points, const Vector3 normal, const Vec<4,float> colour)
{
    if (!pOgreDraw) return;
    setMaterial(colour);
    pOgreDraw->begin(pCurrentMaterial->getName(), Ogre::RenderOperation::OT_TRIANGLE_LIST);

    for (unsigned int i=0; i<points.size()/3; ++i)
    {
        drawPoint(points[ 3*i   ],colour);
        drawPoint(points[ 3*i+1 ],colour);
        drawPoint(points[ 3*i+2 ],colour);
        pOgreDraw->triangle(3*i, 3*i+1, 3*i+2);
    }
    pOgreDraw->end();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawTriangles(const std::vector<Vector3> &points, const std::vector< defaulttype::Vec<3,int> > &index,
        const std::vector<Vector3> &normal, const Vec<4,float> colour=Vec<4,float>(1.0f,1.0f,1.0f,1.0f))
{
    if (!pOgreDraw) return;
    setMaterial(colour);
    pOgreDraw->begin(pCurrentMaterial->getName(), Ogre::RenderOperation::OT_TRIANGLE_LIST);

    for (unsigned int i=0; i<index.size(); ++i)
    {
        drawTriangle(points[ index[i][0] ],points[ index[i][1] ],points[ index[i][2] ],normal[i],colour);
        pOgreDraw->triangle(index[i][0],index[i][1],index[i][2]);
    }
    pOgreDraw->end();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawTriangles(const std::vector<Vector3> &points,
        const std::vector<Vector3> &normal, const std::vector< Vec<4,float> > &colour)
{
    const unsigned int nbTriangles=points.size()/3;
    bool computeNormals= (normal.size() != nbTriangles);
    if (nbTriangles == 0) return;
    if (!pOgreDraw) return;
//            setMaterial(colour);
    pOgreDraw->begin(pCurrentMaterial->getName(), Ogre::RenderOperation::OT_TRIANGLE_LIST);

    for (unsigned int i=0; i<nbTriangles; ++i)
    {
        if (!computeNormals)
        {
            drawTriangle(points[3*i+0],points[3*i+1],points[3*i+2],normal[i],
                    colour[3*i+0],colour[3*i+1],colour[3*i+2]);
        }
        else
        {
            const Vector3& a = points[ 3*i+0 ];
            const Vector3& b = points[ 3*i+1 ];
            const Vector3& c = points[ 3*i+2 ];
            Vector3 n = cross((b-a),(c-a));
            n.normalize();

            drawPoint(a,n,colour[3*i+0]);
            drawPoint(b,n,colour[3*i+1]);
            drawPoint(c,n,colour[3*i+2]);
        }
        pOgreDraw->triangle(3*i+0,3*i+1,3*i+2);
    }
    pOgreDraw->end();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawTriangleStrip(const std::vector<Vector3> &points,
        const std::vector<Vector3>  &normal,
        const Vec<4,float> colour)
{
    if (!pOgreDraw) return;
    setMaterial(colour);
    pOgreDraw->begin(pCurrentMaterial->getName(), Ogre::RenderOperation::OT_TRIANGLE_STRIP);

    for (unsigned int i=0; i<normal.size(); ++i)
    {
        drawOgreVertexPosition(points[2*i]);
        drawOgreVertexNormal(normal[i]);
        drawOgreVertexColour(colour);

        drawOgreVertexPosition(points[2*i+1]);
        drawOgreVertexNormal(normal[i]);
        drawOgreVertexColour(colour);

        pOgreDraw->index(2*i);
        pOgreDraw->index(2*i+1);
    }
    pOgreDraw->end();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawTriangleFan(const std::vector<Vector3> &points,
        const std::vector<Vector3>  &normal,
        const Vec<4,float> colour)
{
    if (points.size() < 3) return;
    if (!pOgreDraw) return;
    setMaterial(colour);
    pOgreDraw->begin(pCurrentMaterial->getName(), Ogre::RenderOperation::OT_TRIANGLE_FAN);

    drawOgreVertexPosition(points[0]);
    drawOgreVertexNormal(normal[0]);
    drawOgreVertexColour(colour);

    drawOgreVertexPosition(points[1]);
    drawOgreVertexNormal(normal[0]);
    drawOgreVertexColour(colour);

    drawOgreVertexPosition(points[2]);
    drawOgreVertexNormal(normal[0]);
    drawOgreVertexColour(colour);

    pOgreDraw->index(0);
    pOgreDraw->index(1);
    pOgreDraw->index(2);

    for (unsigned int i=3; i<points.size(); ++i)
    {
        drawOgreVertexPosition(points[i]);
        drawOgreVertexNormal(normal[i]);
        drawOgreVertexColour(colour);
        pOgreDraw->index(i);
    }

    pOgreDraw->end();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawFrame(const Vector3& position, const Quaternion &orientation, const Vec<3,float> &size)
{
    setPolygonMode(0,false);
    setLightingEnabled(true);
    SReal matrix[16];
    orientation.writeOpenGlMatrix(matrix);

    Vector3 X(matrix[0*4+0], matrix[0*4+1],matrix[0*4+2]);
    Vector3 Y(matrix[1*4+0], matrix[1*4+1],matrix[1*4+2]);
    Vector3 Z(matrix[2*4+0], matrix[2*4+1],matrix[2*4+2]);

    drawArrow(position, position+X*size[0], 0.1*size[0], Vec<4,float>(1.0f,0.0f,0.0f,1.0f),16);
    drawArrow(position, position+Y*size[1], 0.1*size[1], Vec<4,float>(0.0f,1.0f,0.0f,1.0f),16);
    drawArrow(position, position+Z*size[2], 0.1*size[2], Vec<4,float>(0.0f,0.0f,1.0f,1.0f),16);

    setLightingEnabled(false);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawSpheres(const std::vector<Vector3> &points, float radius, const Vec<4,float> colour)
{
    setMaterial(colour);
    for (unsigned int i=0; i<points.size(); ++i)
        drawSphere(points[i], radius);
    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawSpheres(const std::vector<Vector3> &points, const std::vector<float>& radius, const Vec<4,float> colour)
{
    setMaterial(colour);
    for (unsigned int i=0; i<points.size(); ++i)
        drawSphere(points[i], radius[i]);
    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawCone(const Vector3& p1, const Vector3 &p2, float radius1, float radius2, const Vec<4,float> colour, int subd)
{
    Vector3 tmp = p2-p1;
    setMaterial(colour);
    /* create Vectors p and q, co-planar with the cylinder's cross-sectional disk */
    Vector3 p=tmp;
    if (fabs(p[0]) + fabs(p[1]) < 0.00001*tmp.norm())
        p[0] += 1.0;
    else
        p[2] += 1.0;
    Vector3 q;
    q = p.cross(tmp);
    p = tmp.cross(q);
    /* do the normalization outside the segment loop */
    p.normalize();
    q.normalize();

    int i2;
    float theta, st, ct;
    /* build the cylinder from rectangular subd */
    std::vector<Vector3> points;
    std::vector<Vec<4,int> > indices;
    std::vector<Vector3> normals;

    std::vector<Vector3> pointsCloseCylinder1;
    std::vector<Vector3> normalsCloseCylinder1;
    std::vector<Vector3> pointsCloseCylinder2;
    std::vector<Vector3> normalsCloseCylinder2;

    Vector3 dir=p1-p2; dir.normalize();
    pointsCloseCylinder1.push_back(p1);
    normalsCloseCylinder1.push_back(dir);
    pointsCloseCylinder2.push_back(p2);
    normalsCloseCylinder2.push_back(-dir);

    for (i2=0 ; i2<=subd ; i2++)
    {
        /* sweep out a circle */
        theta =  i2 * 2.0 * 3.14 / subd;
        st = sin(theta);
        ct = cos(theta);
        /* construct normal */
        tmp = p*ct+q*st;
        /* set the normal for the two subseqent points */
        normals.push_back(tmp);

        /* point on disk 1 */
        Vector3 w(p1);
        w += tmp*radius1;
        points.push_back(w);
        pointsCloseCylinder1.push_back(w);
        normalsCloseCylinder1.push_back(dir);

        /* point on disk 2 */
        w=p2;
        w += tmp*radius2;
        points.push_back(w);
        pointsCloseCylinder2.push_back(w);
        normalsCloseCylinder2.push_back(-dir);
    }
    pointsCloseCylinder1.push_back(pointsCloseCylinder1[1]);
    normalsCloseCylinder1.push_back(normalsCloseCylinder1[1]);
    pointsCloseCylinder2.push_back(pointsCloseCylinder2[1]);
    normalsCloseCylinder2.push_back(normalsCloseCylinder2[1]);

    drawTriangleStrip(points, normals,colour);
    if (radius1 > 0) drawTriangleFan(pointsCloseCylinder1, normalsCloseCylinder1,colour);
    if (radius2 > 0) drawTriangleFan(pointsCloseCylinder2, normalsCloseCylinder2,colour);

    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawCube( const float& radius, const Vec<4,float>& colour, const int& subd)
{
    // X Axis
    drawCylinder( Vector3(-1.0, -1.0, -1.0), Vector3(1.0, -1.0, -1.0), radius, colour, subd);
    drawCylinder( Vector3(-1.0,  1.0, -1.0), Vector3(1.0,  1.0, -1.0), radius, colour, subd);
    drawCylinder( Vector3(-1.0, -1.0,  1.0), Vector3(1.0, -1.0,  1.0), radius, colour, subd);
    drawCylinder( Vector3(-1.0,  1.0,  1.0), Vector3(1.0,  1.0,  1.0), radius, colour, subd);
    // Y Axis
    drawCylinder( Vector3(-1.0, -1.0, -1.0), Vector3(-1.0, 1.0, -1.0), radius, colour, subd);
    drawCylinder( Vector3(-1.0, -1.0,  1.0), Vector3(-1.0, 1.0,  1.0), radius, colour, subd);
    drawCylinder( Vector3( 1.0, -1.0, -1.0), Vector3( 1.0, 1.0, -1.0), radius, colour, subd);
    drawCylinder( Vector3( 1.0, -1.0,  1.0), Vector3( 1.0, 1.0,  1.0), radius, colour, subd);
    // Z Axis
    drawCylinder( Vector3(-1.0, -1.0, -1.0), Vector3(-1.0, -1.0, 1.0), radius, colour, subd);
    drawCylinder( Vector3(-1.0,  1.0, -1.0), Vector3(-1.0,  1.0, 1.0), radius, colour, subd);
    drawCylinder( Vector3( 1.0, -1.0, -1.0), Vector3( 1.0, -1.0, 1.0), radius, colour, subd);
    drawCylinder( Vector3( 1.0,  1.0, -1.0), Vector3( 1.0,  1.0, 1.0), radius, colour, subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawCylinder(const Vector3& p1, const Vector3 &p2, float radius, const Vec<4,float> colour, int subd)
{
    drawCone( p1,p2,radius,radius,colour,subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawArrow(const Vector3& p1, const Vector3 &p2, float radius, const Vec<4,float> colour,  int subd)
{
    Vector3 p3 = p1*.2+p2*.8;
    drawCylinder( p1,p3,radius,colour,subd);
    drawCone( p3,p2,radius*2.5,0,colour,subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawPlus ( const float& radius, const Vec<4,float>& colour, const int& subd)
{
    drawCylinder( Vector3(-1.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), radius, colour, subd);
    drawCylinder( Vector3(0.0, -1.0, 0.0), Vector3(0.0, 1.0, 0.0), radius, colour, subd);
    drawCylinder( Vector3(0.0, 0.0, -1.0), Vector3(0.0, 0.0, 1.0), radius, colour, subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawPoint(const Vector3 &p, const Vec<4,float> &c)
{
    if (!pOgreDraw) return;
    drawOgreVertexPosition(p);
    drawOgreVertexColour(c);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawPoint(const Vector3 &p, const Vector3 &n, const Vec<4,float> &c)
{
    if (!pOgreDraw) return;
    drawOgreVertexPosition(p);
    drawOgreVertexNormal(p);
    drawOgreVertexColour(c);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal,
        const Vec<4,float> &c1, const Vec<4,float> &c2, const Vec<4,float> &c3)
{
    if (!pOgreDraw) return;
    drawOgreVertexPosition(p1);
    drawOgreVertexNormal(normal);
    drawOgreVertexColour(c1);
    drawOgreVertexPosition(p2);
    drawOgreVertexNormal(normal);
    drawOgreVertexColour(c2);
    drawOgreVertexPosition(p3);
    drawOgreVertexNormal(normal);
    drawOgreVertexColour(c3);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawTriangle( const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,
        const Vector3 &normal, const  Vec<4,float> &c)
{
    if (!pOgreDraw) return;
    drawOgreVertexPosition(p1);
    drawOgreVertexNormal(normal);
    drawOgreVertexColour(c);

    drawOgreVertexPosition(p2);
    drawOgreVertexNormal(normal);
    drawOgreVertexColour(c);
    drawOgreVertexPosition(p3);
    drawOgreVertexNormal(normal);
    drawOgreVertexColour(c);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawSphere( const Vector3 &p, float radius)
{
    std::ostringstream s;
    s << "entity[" << mEntityName++ <<"]";
    Ogre::Entity* sph = pSceneMgr->createEntity( s.str(), "mesh/ball.mesh" );//pSceneMgr->createEntity(s.str().c_str(), Ogre::SceneManager::PT_SPHERE );
    s.str("");
    s << "material[" << mMaterialName-1 << "]" ;
    sph->setMaterialName(s.str());

    Ogre::SceneNode* node = pSceneMgr->getRootSceneNode()->createChildSceneNode();
    node->setScale(radius,radius,radius);
    node->setPosition(p[0],p[1],p[2]);
    node->attachObject(sph);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawOgreVertexPosition(const Vector3 &p)
{
    pOgreDraw->position(p[0],p[1],p[2]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawOgreVertexNormal(const Vector3 &p)
{
    pOgreDraw->normal(p[0],p[1],p[2]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::drawOgreVertexColour(const Vec<4,float> &p)
{
    pOgreDraw->colour(p[0],p[1],p[2],p[3]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::setPolygonMode(int _mode, bool _wireframe)
{
    mPolygonMode = _mode;
    mWireFrameEnabled = _wireframe;

    if(pCurrentMaterial.isNull() ) return;
    if (!mPolygonMode)
    {
        if (mWireFrameEnabled)
        {
            pCurrentMaterial->getTechnique(0)->getPass(0)->setPolygonMode(Ogre::PM_WIREFRAME);//there is also PM_POINTS
            pCurrentMaterial->setCullingMode(Ogre::CULL_NONE);
        }
        else
        {
            pCurrentMaterial->getTechnique(0)->getPass(0)->setPolygonMode(Ogre::PM_SOLID);//there is also PM_POINTS
            pCurrentMaterial->setCullingMode(Ogre::CULL_NONE);
        }
    }
    else if (mPolygonMode == 1)
    {
        if (mWireFrameEnabled)
        {
            pCurrentMaterial->getTechnique(0)->getPass(0)->setPolygonMode(Ogre::PM_WIREFRAME);
            pCurrentMaterial->setCullingMode(Ogre::CULL_CLOCKWISE);
        }
        else
        {
            pCurrentMaterial->getTechnique(0)->getPass(0)->setPolygonMode(Ogre::PM_SOLID);
            pCurrentMaterial->setCullingMode(Ogre::CULL_CLOCKWISE);
        }
    }
    else if (mPolygonMode == 2)
    {
        if (mWireFrameEnabled)
        {
            pCurrentMaterial->getTechnique(0)->getPass(0)->setPolygonMode(Ogre::PM_WIREFRAME);
            pCurrentMaterial->setCullingMode(Ogre::CULL_ANTICLOCKWISE);
        }
        else
        {
            pCurrentMaterial->getTechnique(0)->getPass(0)->setPolygonMode(Ogre::PM_SOLID);
            pCurrentMaterial->setCullingMode(Ogre::CULL_ANTICLOCKWISE);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::setLightingEnabled(bool _isAnabled)
{
    mLightEnabled = _isAnabled;
    if(pCurrentMaterial.isNull()) return;
    pCurrentMaterial.getPointer()->setLightingEnabled(getLightEnabled());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::setMaterial(const Vec<4,float> &colour,std::string name)
{
    //Get the Material
    if (name.empty())
    {
        std::ostringstream s;
        s << "material[" << mMaterialName++ << "]" ;
        pCurrentMaterial = Ogre::MaterialManager::getSingleton().create(s.str(), "General");
    }
    else
        pCurrentMaterial = Ogre::MaterialManager::getSingleton().getByName(name);

    //Light
    pCurrentMaterial.getPointer()->setLightingEnabled(getLightEnabled());

    //Culling
    switch( mPolygonMode )
    {
    case 0:
        pCurrentMaterial.getPointer()->getTechnique(0)->setCullingMode(Ogre::CULL_NONE);
        break;
    case 1:
        pCurrentMaterial.getPointer()->getTechnique(0)->setCullingMode(Ogre::CULL_CLOCKWISE);
        break;
    case 2:
        pCurrentMaterial.getPointer()->getTechnique(0)->setCullingMode(Ogre::CULL_ANTICLOCKWISE);
        break;
    }

    //Blending
    if (colour[3] < 1)
    {
        pCurrentMaterial.getPointer()->setDepthWriteEnabled(false);
        pCurrentMaterial.getPointer()->getTechnique(0)->setSceneBlending(Ogre::SBT_TRANSPARENT_ALPHA);
        pCurrentMaterial.getPointer()->setLightingEnabled(false);
        pCurrentMaterial.getPointer()->setCullingMode(Ogre::CULL_NONE);
    }
    else
        pCurrentMaterial.getPointer()->setDepthWriteEnabled(true);

    //Shading
    pCurrentMaterial.getPointer()->getTechnique(0)->setShadingMode(Ogre::SO_PHONG);

    //Colour
    pCurrentMaterial.getPointer()->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(colour[0],colour[1],colour[2],colour[3]));
    pCurrentMaterial.getPointer()->getTechnique(0)->getPass(0)->setAmbient(Ogre::ColourValue(colour[0],colour[1],colour[2],colour[3]));
    pCurrentMaterial.getPointer()->getTechnique(0)->getPass(0)->setSelfIllumination(Ogre::ColourValue(0,0,0,1));
    pCurrentMaterial.getPointer()->getTechnique(0)->getPass(0)->setSpecular(Ogre::ColourValue(1,1,1,1));
    pCurrentMaterial.getPointer()->getTechnique(0)->getPass(0)->setShininess(Ogre::Real(45));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::resetMaterial(const Vec<4,float> &colour,std::string name)
{
    this->setMaterial(colour, name);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolOGRE::clear()
{
    if (pOgreDraw) pOgreDraw->clear();
    for (int i=0; i<mEntityName; ++i)
    {
        std::ostringstream s;
        s << "entity[" << i <<"]";
        Ogre::SceneNode* n=(Ogre::SceneNode*)(pSceneMgr->getEntity(s.str())->getParentNode());
        pSceneMgr->destroyEntity(s.str());
        pSceneMgr->destroySceneNode(n);
    }

    for (int i=0; i<mMaterialName; ++i)
    {
        std::ostringstream s;
        s << "material[" << i <<"]" ;
        Ogre::MaterialManager::getSingleton().remove(s.str());
    }

    for( int i=0; i<mMeshName; ++i)
    {
        std::ostringstream s;
        s << "mesh[" << i <<"]";
        Ogre::MeshManager::getSingleton().remove(s.str());
    }

    mEntityName=0;
    mMaterialName=0;
    mMeshName=0;
}

void DrawToolOGRE::pushMatrix()
{
    if(pOgreDrawInternal == NULL)
    {
        pOgreDrawInternal = (Ogre::ManualObject*)pSceneMgr->createMovableObject("drawUtilityInternal","ManualObject");
    }
    pOgreDrawInternal->clear();
    // save the OgreDraw pointer
    pOgreDrawSave = pOgreDraw;
    // drawing of primitive is handled to OgreDrawInternal
    pOgreDraw = pOgreDrawInternal;
    Ogre::SceneNode* currentSceneNode = sceneNodeStack.top();
    sceneNodeStack.push(currentSceneNode->createChildSceneNode());
}

void DrawToolOGRE::popMatrix()
{
    assert(pOgreDrawInternal);

    //create the entity corresponding to what has been drawn so far by
    //pOgreDrawInternal
    std::ostringstream s;
    s << "mesh[" << mMeshName++ <<"]";
    Ogre::MeshPtr ogreMesh = pOgreDrawInternal->convertToMesh(s.str(), "General");
    s.str("");
    s << "entity[" << mEntityName++ <<"]";
    Ogre::Entity *e = pSceneMgr->createEntity(s.str(), ogreMesh->getName());

    //restore drawing of primitives to OgreDraw
    pOgreDraw = pOgreDrawSave;

    //attach the entity to the top stack node.
    Ogre::SceneNode* currentSceneNode = sceneNodeStack.top();
    currentSceneNode->attachObject(e);
    //reduce stack size by one
    sceneNodeStack.pop();
    assert(!sceneNodeStack.empty());
}
void DrawToolOGRE::multMatrix(float* glTransform)
{
    Ogre::Matrix4 t( glTransform[0], glTransform[1], glTransform[2], glTransform[3],
            glTransform[4], glTransform[5], glTransform[6], glTransform[7],
            glTransform[8], glTransform[9], glTransform[10], glTransform[11],
            glTransform[12], glTransform[13], glTransform[14], glTransform[15]);
    t = t.transpose();
    Ogre::SceneNode* currentSceneNode = sceneNodeStack.top();
    currentSceneNode->setOrientation(t.extractQuaternion());
    currentSceneNode->setPosition(t.getTrans());
}

void DrawToolOGRE::scale(float s)
{
    Ogre::SceneNode* currentSceneNode = sceneNodeStack.top();
    currentSceneNode->setScale(s,s,s);
}


} // namespace gl

} // namespace helper

} // namespace sofa
