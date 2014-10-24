/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/


#include <sofa/core/visual/DrawToolGL.h>

#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/Cylinder.h>
#include <sofa/helper/gl/template.h>
#include <math.h>

namespace sofa
{

namespace core
{

namespace visual
{

using namespace sofa::defaulttype;
using namespace sofa::helper::gl;

DrawToolGL::DrawToolGL()
{
    clear();
    mLightEnabled = false;
    mWireFrameEnabled = false;
    mPolygonMode = 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DrawToolGL::~DrawToolGL()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawPoints(const std::vector<Vector3> &points, float size, const Vec<4,float> colour=Vec<4,float>(1.0f,1.0f,1.0f,1.0f))
{
    setMaterial(colour);
    glPointSize(size);
    glDisable(GL_LIGHTING);
    glBegin(GL_POINTS);
    {
        for (unsigned int i=0; i<points.size(); ++i)
        {
            drawPoint(points[i], colour);
        }
    } glEnd();
    if (getLightEnabled()) glEnable(GL_LIGHTING);
    resetMaterial(colour);
    glPointSize(1);
}

void DrawToolGL::drawPoints(const std::vector<Vector3> &points, float size, const std::vector<Vec4f> colour)
{
    glPointSize(size);
    glDisable(GL_LIGHTING);
    glBegin(GL_POINTS);
    {
        for (unsigned int i=0; i<points.size(); ++i)
        {
            setMaterial(colour[i]);
            drawPoint(points[i], colour[i]);
            if (getLightEnabled()) glEnable(GL_LIGHTING);
            resetMaterial(colour[i]);
        }
    } glEnd();
    glPointSize(1);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawLines(const std::vector<Vector3> &points, float size, const Vec<4,float> colour)
{
    setMaterial(colour);
    glLineWidth(size);
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    {
        for (unsigned int i=0; i<points.size()/2; ++i)
        {
            drawPoint(points[2*i]  , colour );
            drawPoint(points[2*i+1], colour );
        }
    } glEnd();
    if (getLightEnabled()) glEnable(GL_LIGHTING);
    resetMaterial(colour);
    glLineWidth(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawLines(const std::vector<Vector3> &points, const std::vector< defaulttype::Vec<2,int> > &index, float size, const Vec<4,float> colour=Vec<4,float>(1.0f,1.0f,1.0f,1.0f))
{
    setMaterial(colour);
    glLineWidth(size);
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    {
        for (unsigned int i=0; i<index.size(); ++i)
        {
            drawPoint(points[ index[i][0] ], colour );
            drawPoint(points[ index[i][1] ], colour );
        }
    } glEnd();
    if (getLightEnabled()) glEnable(GL_LIGHTING);
    resetMaterial(colour);
    glLineWidth(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points, const Vec<4,float> colour)
{
    setMaterial(colour);
    glBegin(GL_TRIANGLES);
    {
        for (unsigned int i=0; i<points.size()/3; ++i)
        {
            const Vector3& a = points[ 3*i+0 ];
            const Vector3& b = points[ 3*i+1 ];
            const Vector3& c = points[ 3*i+2 ];
            Vector3 n = cross((b-a),(c-a));
            n.normalize();
            drawTriangle(a,b,c,n,colour);
        }
    } glEnd();
    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points, const Vector3 normal, const Vec<4,float> colour)
{
    setMaterial(colour);
    glBegin(GL_TRIANGLES);
    {
        for (unsigned int i=0; i<points.size()/3; ++i)
            drawTriangle(points[ 3*i+0 ],points[ 3*i+1 ],points[ 3*i+2 ], normal, colour);
    } glEnd();
    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points, const std::vector< defaulttype::Vec<3,int> > &index,
        const std::vector<Vector3> &normal, const Vec<4,float> colour=Vec<4,float>(1.0f,1.0f,1.0f,1.0f))
{
    setMaterial(colour);
    glBegin(GL_TRIANGLES);
    {
        for (unsigned int i=0; i<index.size(); ++i)
        {
            drawTriangle(points[ index[i][0] ],points[ index[i][1] ],points[ index[i][2] ],normal[i],colour);
        }
    } glEnd();
    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points,
        const std::vector<Vector3> &normal, const std::vector< Vec<4,float> > &colour)
{
    const std::size_t nbTriangles=points.size()/3;
    bool computeNormals= (normal.size() != nbTriangles);
    if (nbTriangles == 0) return;
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    setMaterial(colour[0]);
    glBegin(GL_TRIANGLES);
    {
        for (std::size_t i=0; i<nbTriangles; ++i)
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
        }
    } glEnd();
    glDisable(GL_COLOR_MATERIAL);
    resetMaterial(colour[0]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangleStrip(const std::vector<Vector3> &points,
        const std::vector<Vector3>  &normal,
        const Vec<4,float> colour)
{
    setMaterial(colour);
    glBegin(GL_TRIANGLE_STRIP);
    {
        for (unsigned int i=0; i<normal.size(); ++i)
        {
            glNormalT(normal[i]);
            glVertexNv<3>(points[2*i].ptr());
            glVertexNv<3>(points[2*i+1].ptr());
        }
    } glEnd();
    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangleFan(const std::vector<Vector3> &points,
        const std::vector<Vector3>  &normal,
        const Vec<4,float> colour)
{
    if (points.size() < 3) return;
    setMaterial(colour);
    glBegin(GL_TRIANGLE_FAN);

    glNormalT(normal[0]);
    glVertexNv<3>(points[0].ptr());
    glVertexNv<3>(points[1].ptr());
    glVertexNv<3>(points[2].ptr());

    for (unsigned int i=3; i<points.size(); ++i)
    {
        glNormalT(normal[i]);
        glVertexNv<3>(points[i].ptr());
    }

    glEnd();
    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawFrame(const Vector3& position, const Quaternion &orientation, const Vec<3,float> &size)
{
    setPolygonMode(0,false);
    helper::gl::Axis::draw(position, orientation, size);
}
void DrawToolGL::drawFrame(const Vector3& position, const Quaternion &orientation, const Vec<3,float> &size, const Vec4f &colour)
{
    setPolygonMode(0,false);
    helper::gl::Axis::draw(position, orientation, size, colour, colour, colour);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawSpheres(const std::vector<Vector3> &points, float radius, const Vec<4,float> colour)
{
    setMaterial(colour);
    for (unsigned int i=0; i<points.size(); ++i)
        drawSphere(points[i], radius);

    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawSpheres(const std::vector<Vector3> &points, const std::vector<float>& radius, const Vec<4,float> colour)
{
    setMaterial(colour);
    for (unsigned int i=0; i<points.size(); ++i)
        drawSphere(points[i], radius[i]);

    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawCapsule(const Vector3& p1, const Vector3 &p2, float radius,const Vec<4,float> colour, int subd){
    Vector3 tmp = p2-p1;
    setMaterial(colour);
    /* create Vectors p and q, co-planar with the capsules's cross-sectional disk */
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
    /* build the cylinder part of the capsule from rectangular subd */
    std::vector<Vector3> points;
    std::vector<Vector3> normals;

    for (i2=0 ; i2<=subd ; i2++)
    {
        /* sweep out a circle */
        float theta =  (float)( i2 * 2.0f * M_PI / subd );
        float st = sin(theta);
        float ct = cos(theta);
        /* construct normal */
        tmp = p*ct+q*st;
        /* set the normal for the two subseqent points */
        normals.push_back(tmp);

        Vector3 w(p1);
        w += tmp*fabs(radius);
        points.push_back(w);

        w=p2;
        w += tmp*fabs(radius);
        points.push_back(w);
    }

    //we draw here the cylinder part
    drawTriangleStrip(points, normals,colour);

    //now we must draw the two hemispheres
    //but it's easier to draw spheres...
    drawSphere(p1,radius);
    drawSphere(p2,radius);

    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawCone(const Vector3& p1, const Vector3 &p2, float radius1, float radius2, const Vec<4,float> colour, int subd)
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
    
    /* build the cylinder from rectangular subd */
    std::vector<Vector3> points;
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
        float theta =  (float)( i2 * 2.0f * M_PI / subd );
        float st = sin(theta);
        float ct = cos(theta);
        /* construct normal */
        tmp = p*ct+q*st;
        /* set the normal for the two subseqent points */
        normals.push_back(tmp);

        /* point on disk 1 */
        Vector3 w(p1);
        w += tmp*fabs(radius1);
        points.push_back(w);
        pointsCloseCylinder1.push_back(w);
        normalsCloseCylinder1.push_back(dir);

        /* point on disk 2 */
        w=p2;
        w += tmp*fabs(radius2);
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

void DrawToolGL::drawCube( const float& radius, const Vec<4,float>& colour, const int& subd)
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

void DrawToolGL::drawCylinder(const Vector3& p1, const Vector3 &p2, float radius, const Vec<4,float> colour, int subd)
{
    drawCone( p1,p2,radius,radius,colour,subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawArrow(const Vector3& p1, const Vector3 &p2, float radius, const Vec<4,float> colour,  int subd)
{

    Vector3 p3 = p1*.2+p2*.8;
    drawCylinder( p1,p3,radius,colour,subd);
    drawCone( p3,p2,radius*2.5f,0,colour,subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawPlus ( const float& radius, const Vec<4,float>& colour, const int& subd)
{
    drawCylinder( Vector3(-1.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), radius, colour, subd);
    drawCylinder( Vector3(0.0, -1.0, 0.0), Vector3(0.0, 1.0, 0.0), radius, colour, subd);
    drawCylinder( Vector3(0.0, 0.0, -1.0), Vector3(0.0, 0.0, 1.0), radius, colour, subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawPoint(const Vector3 &p, const Vec<4,float> &c)
{
#ifdef PS3
	// bit of a hack we force to enter our emulation of draw immediate
	// because glColor4f already exists in OGL ES.
    glColor3f(c[0],c[1],c[2]);
#else 
	glColor4f(c[0],c[1],c[2],c[3]);
#endif
    glVertexNv<3>(p.ptr());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawPoint(const Vector3 &p, const Vector3 &n, const Vec<4,float> &c)
{
#ifdef PS3
	// bit of a hack we force to enter our emulation of draw immediate
	// because glColor4f already exists in OGL ES.
    glColor3f(c[0],c[1],c[2]);
#else 
	glColor4f(c[0],c[1],c[2],c[3]);
#endif
    glNormalT(n);
    glVertexNv<3>(p.ptr());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal,
        const Vec<4,float> &c1, const Vec<4,float> &c2, const Vec<4,float> &c3)
{
    glNormalT(normal);
    glColor4fv(c1.ptr());
    glVertexNv<3>(p1.ptr());
    glColor4fv(c2.ptr());
    glVertexNv<3>(p2.ptr());
    glColor4fv(c3.ptr());
    glVertexNv<3>(p3.ptr());
}


void DrawToolGL::drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3,
        const Vec<4,float> &c1, const Vec<4,float> &c2, const Vec<4,float> &c3)
{
    glNormalT(normal1);
    glColor4fv(c1.ptr());
    glVertexNv<3>(p1.ptr());
    glNormalT(normal2);
    glColor4fv(c2.ptr());
    glVertexNv<3>(p2.ptr());
    glNormalT(normal3);
    glColor4fv(c3.ptr());
    glVertexNv<3>(p3.ptr());
}


void DrawToolGL::drawTriangle( const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,
        const Vector3 &normal, const  Vec<4,float> &c)
{
    glNormalT(normal);
    glColor4fv(c.ptr());
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
}


void DrawToolGL::drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal)
{
    glNormalT(normal);
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal,
        const Vec<4,float> &c1, const Vec<4,float> &c2, const Vec<4,float> &c3, const Vec<4,float> &c4)
{
    glNormalT(normal);
    glColor4fv(c1.ptr());
    glVertexNv<3>(p1.ptr());
    glColor4fv(c2.ptr());
    glVertexNv<3>(p2.ptr());
    glColor4fv(c3.ptr());
    glVertexNv<3>(p3.ptr());
    glColor4fv(c4.ptr());
    glVertexNv<3>(p4.ptr());
}


void DrawToolGL::drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3, const Vector3 &normal4,
        const Vec<4,float> &c1, const Vec<4,float> &c2, const Vec<4,float> &c3, const Vec<4,float> &c4)
{
    glNormalT(normal1);
    glColor4fv(c1.ptr());
    glVertexNv<3>(p1.ptr());
    glNormalT(normal2);
    glColor4fv(c2.ptr());
    glVertexNv<3>(p2.ptr());
    glNormalT(normal3);
    glColor4fv(c3.ptr());
    glVertexNv<3>(p3.ptr());
    glNormalT(normal4);
    glColor4fv(c4.ptr());
    glVertexNv<3>(p4.ptr());
}


void DrawToolGL::drawQuad( const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal, const  Vec<4,float> &c)
{
    glNormalT(normal);
    glColor4fv(c.ptr());
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
    glVertexNv<3>(p4.ptr());
}


void DrawToolGL::drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal)
{
    glNormalT(normal);
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
    glVertexNv<3>(p4.ptr());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawSphere( const Vector3 &p, float radius)
{
    glPushMatrix();
    glTranslated(p[0], p[1], p[2]);
    glutSolidSphere(radius, 32, 16);
    glPopMatrix();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawBoundingBox( const Vector3 &min, const Vector3 &max )
{
    glBegin( GL_LINES );

    // 0-1
    glVertex3f( (float)min[0], (float)min[1], (float)min[2] );
    glVertex3f( (float)max[0], (float)min[1], (float)min[2] );
    // 2-3
    glVertex3f( (float)max[0], (float)max[1], (float)min[2] );
    glVertex3f( (float)min[0], (float)max[1], (float)min[2] );
    // 4-5
    glVertex3f( (float)min[0], (float)min[1], (float)max[2] );
    glVertex3f( (float)max[0], (float)min[1], (float)max[2] );
    // 6-7
    glVertex3f( (float)max[0], (float)max[1], (float)max[2] );
    glVertex3f( (float)min[0], (float)max[1], (float)max[2] );
    // 0-3
    glVertex3f( (float)min[0], (float)min[1], (float)min[2] );
    glVertex3f( (float)min[0], (float)max[1], (float)min[2] );
    // 1-2
    glVertex3f( (float)max[0], (float)min[1], (float)min[2] );
    glVertex3f( (float)max[0], (float)max[1], (float)min[2] );
    // 4-7
    glVertex3f( (float)min[0], (float)min[1], (float)max[2] );
    glVertex3f( (float)min[0], (float)max[1], (float)max[2] );
    // 5-6
    glVertex3f( (float)max[0], (float)min[1], (float)max[2] );
    glVertex3f( (float)max[0], (float)max[1], (float)max[2] );
    // 0-4
    glVertex3f( (float)min[0], (float)min[1], (float)min[2] );
    glVertex3f( (float)min[0], (float)min[1], (float)max[2] );
    // 1-5
    glVertex3f( (float)max[0], (float)min[1], (float)min[2] );
    glVertex3f( (float)max[0], (float)min[1], (float)max[2] );
    // 2-6
    glVertex3f( (float)max[0], (float)max[1], (float)min[2] );
    glVertex3f( (float)max[0], (float)max[1], (float)max[2] );
    // 3-7
    glVertex3f( (float)min[0], (float)max[1], (float)min[2] );
    glVertex3f( (float)min[0], (float)max[1], (float)max[2] );

    glEnd();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::setPolygonMode(int _mode, bool _wireframe)
{
    mPolygonMode=_mode;
    mWireFrameEnabled=_wireframe;
    if (!mPolygonMode)
    {
        if (mWireFrameEnabled) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        else                  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    else if (mPolygonMode == 1)
    {
        if (mWireFrameEnabled) glPolygonMode(GL_FRONT, GL_LINE);
        else                  glPolygonMode(GL_FRONT, GL_FILL);
    }
    else if (mPolygonMode == 2)
    {
        if (mWireFrameEnabled) glPolygonMode(GL_BACK, GL_LINE);
        else                  glPolygonMode(GL_BACK, GL_FILL);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::setLightingEnabled(bool _isAnabled)
{
    mLightEnabled = _isAnabled;
    if (this->getLightEnabled()) glEnable(GL_LIGHTING);
    else glDisable(GL_LIGHTING);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::setMaterial(const Vec<4,float> &colour,std::string)
{
    glColor4f(colour[0],colour[1],colour[2],colour[3]);
    glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, &colour[0]);
    static const float emissive[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
    static const float specular[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
    glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive);
    glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular);
    glMaterialf  (GL_FRONT_AND_BACK, GL_SHININESS, 20);
    if (colour[3] < 1)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(0);
    }
    else
    {
        glDisable(GL_BLEND);
        glDepthMask(1);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::resetMaterial(const Vec<4,float> &colour,std::string)
{
    if (colour[3] < 1)
    {
        glDisable(GL_BLEND);
        glDepthMask(1);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::clear()
{
    helper::gl::Axis::clear();
    helper::gl::Cylinder::clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::pushMatrix()
{
    glPushMatrix();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::popMatrix()
{
    glPopMatrix();
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::multMatrix(float* glTransform )
{
    glMultMatrix(glTransform);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::scale( float s )
{
    glScale(s,s,s);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::writeOverlayText( int x, int y, unsigned fontSize, const Vec4f &color, const char* text )
{
    GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport );


    static const float letterSize = (float)glutStrokeWidth( GLUT_STROKE_ROMAN, 'm' );

    float scale = fontSize / letterSize;

    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);


    glPushAttrib( GL_LIGHTING_BIT );
    glEnable( GL_COLOR_MATERIAL );

    glPushAttrib( GL_ENABLE_BIT );
    glEnable( GL_LINE_SMOOTH );
    glEnable( GL_POLYGON_SMOOTH );
    glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );



    glColor4f( color[0], color[1], color[2], color[3] );


    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, viewport[2], 0, viewport[3] );

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();


    glTranslated(x,viewport[3]-y-fontSize,0);


    glScalef( scale, scale, scale );

    glLineWidth( fontSize/20.0f );

    for( const char*c = text ; *c ; ++c )
    {
        glutStrokeCharacter ( GLUT_STROKE_ROMAN, *c );
    }

    glPopAttrib(); // GL_ENABLE_BIT
    glPopAttrib(); // GL_LIGHTING_BIT

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

}



} // namespace visual

} // namespace core

} // namespace sofa
