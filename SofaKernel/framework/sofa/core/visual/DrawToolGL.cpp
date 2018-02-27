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
#define SOFA_HELPER_GL_DRAWTOOLGL_CPP

#include <sofa/core/visual/DrawToolGL.h>

#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/helper/gl/BasicShapesGL.inl>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/Cylinder.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/glText.inl>
#include <cmath>

namespace sofa
{

namespace helper
{

namespace gl
{

template class SOFA_CORE_API BasicShapesGL_Sphere< sofa::defaulttype::Vector3 >;
template class SOFA_CORE_API BasicShapesGL_FakeSphere< sofa::defaulttype::Vector3 >;

} // namespace gl

} // namespace helper

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

void DrawToolGL::init()
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawPoints(const std::vector<Vector3> &points, float size, const Vec<4,float>& colour=Vec<4,float>(1.0f,1.0f,1.0f,1.0f))
{
    setMaterial(colour);
    glPointSize(size);
    disableLighting();
    glBegin(GL_POINTS);
    {
        for (unsigned int i=0; i<points.size(); ++i)
        {
            internalDrawPoint(points[i], colour);
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    resetMaterial(colour);
    glPointSize(1);
}

void DrawToolGL::drawPoints(const std::vector<Vector3> &points, float size, const std::vector<Vec4f>& colour)
{
    glPointSize(size);
    disableLighting();
    glBegin(GL_POINTS);
    {
        for (unsigned int i=0; i<points.size(); ++i)
        {
            setMaterial(colour[i]);
            internalDrawPoint(points[i], colour[i]);
            if (getLightEnabled()) enableLighting();
            resetMaterial(colour[i]);
        }
    } glEnd();
    glPointSize(1);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::internalDrawLine(const Vector3 &p1, const Vector3 &p2, const Vec4f& colour)
{
    internalDrawPoint(p1, colour );
    internalDrawPoint(p2, colour );
}

void DrawToolGL::drawLine(const Vector3 &p1, const Vector3 &p2, const Vec4f& colour)
{
    glBegin(GL_LINES);
    internalDrawLine(p1,p2,colour);
    glEnd();
}

void DrawToolGL::drawLines(const std::vector<Vector3> &points, float size, const Vec<4,float>& colour)
{
    setMaterial(colour);
    glLineWidth(size);
    disableLighting();
    glBegin(GL_LINES);
    {
        for (unsigned int i=0; i<points.size()/2; ++i)
        {
            internalDrawLine(points[2*i],points[2*i+1]  , colour );
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    resetMaterial(colour);
    glLineWidth(1);
}

void DrawToolGL::drawLines(const std::vector<Vector3> &points, float size, const std::vector<Vec<4,float> >& colours)
{
    glLineWidth(size);
    disableLighting();
    glBegin(GL_LINES);
    {
        for (unsigned int i=0; i<points.size()/2; ++i)
        {
            setMaterial(colours[i]);
            internalDrawLine(points[2*i],points[2*i+1]  , colours[i] );
            resetMaterial(colours[i]);
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    glLineWidth(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawLines(const std::vector<Vector3> &points, const std::vector< defaulttype::Vec<2,int> > &index, float size, const Vec<4,float>& colour=Vec<4,float>(1.0f,1.0f,1.0f,1.0f))
{
    setMaterial(colour);
    glLineWidth(size);
    disableLighting();
    glBegin(GL_LINES);
    {
        for (unsigned int i=0; i<index.size(); ++i)
        {
            internalDrawLine(points[ index[i][0] ],points[ index[i][1] ], colour );
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    resetMaterial(colour);
    glLineWidth(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawLineStrip(const std::vector<Vector3> &points, float size, const Vec<4,float>& colour)
{
    setMaterial(colour);
    glLineWidth(size);
    disableLighting();
    glBegin(GL_LINE_STRIP);
    {
        for (unsigned int i=0; i<points.size(); ++i)
        {
            internalDrawPoint(points[i]  , colour );
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    resetMaterial(colour);
    glLineWidth(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawLineLoop(const std::vector<Vector3> &points, float size, const Vec<4,float>& colour)
{
    setMaterial(colour);
    glLineWidth(size);
    disableLighting();
    glBegin(GL_LINE_LOOP);
    {
        for (unsigned int i=0; i<points.size(); ++i)
        {
            internalDrawPoint(points[i]  , colour );
        }
    } glEnd();
    if (getLightEnabled()) enableLighting();
    resetMaterial(colour);
    glLineWidth(1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points, const Vec<4,float>& colour)
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

void DrawToolGL::drawTriangles(const std::vector<Vector3> &points, const Vector3& normal, const Vec<4,float>& colour)
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
        const std::vector<Vector3> &normal, const Vec<4,float>& colour=Vec<4,float>(1.0f,1.0f,1.0f,1.0f))
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

                internalDrawPoint(a,n,colour[3*i+0]);
                internalDrawPoint(b,n,colour[3*i+1]);
                internalDrawPoint(c,n,colour[3*i+2]);

            }
        }
    } glEnd();
    glDisable(GL_COLOR_MATERIAL);
    resetMaterial(colour[0]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawTriangleStrip(const std::vector<Vector3> &points,
        const std::vector<Vector3>  &normal,
        const Vec<4,float>& colour)
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
        const Vec<4,float>& colour)
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

void DrawToolGL::drawSpheres(const std::vector<Vector3> &points, float radius, const Vec<4,float>& colour)
{
    setMaterial(colour);

    m_sphereUtil.draw(points, radius);

    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawSpheres(const std::vector<Vector3> &points, const std::vector<float>& radius, const Vec<4,float>& colour)
{
    setMaterial(colour);

    m_sphereUtil.draw(points, radius);

    resetMaterial(colour);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawFakeSpheres(const std::vector<Vector3> &points, float radius, const Vec<4, float>& colour)
{
    setMaterial(colour);

    m_fakeSphereUtil.draw(points, radius);

    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawFakeSpheres(const std::vector<Vector3> &points, const std::vector<float>& radius, const Vec<4, float>& colour)
{
    setMaterial(colour);

    m_fakeSphereUtil.draw(points, radius);

    resetMaterial(colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawCapsule(const Vector3& p1, const Vector3 &p2, float radius,const Vec<4,float>& colour, int subd){
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

void DrawToolGL::drawCone(const Vector3& p1, const Vector3 &p2, float radius1, float radius2, const Vec<4,float>& colour, int subd)
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

void DrawToolGL::drawCylinder(const Vector3& p1, const Vector3 &p2, float radius, const Vec<4,float>& colour, int subd)
{
    drawCone( p1,p2,radius,radius,colour,subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawArrow(const Vector3& p1, const Vector3 &p2, float radius, const Vec<4,float>& colour, int subd)
{
    Vector3 p3 = p1*.2+p2*.8;
    drawCylinder( p1,p3,radius,colour,subd);
    drawCone( p3,p2,radius*2.5f,0,colour,subd);
}


void DrawToolGL::drawArrow(const Vector3& p1, const Vector3 &p2, float radius, float coneLength, const Vec<4,float>& colour, int subd)
{
    // fixed coneLength ; cone can be stretched or when its length depends on the total arrow length

    Vector3 a = p2 - p1;
    SReal n = a.norm();
    if( coneLength >= n )
        drawCone( p1,p2,radius*2.5f,0,colour,subd);
    else
    {
        a /= n; // normalizing
        Vector3 p3 = p2 - coneLength*a;
        drawCylinder( p1,p3,radius,colour,subd);
        drawCone( p3,p2,radius*2.5f,0,colour,subd);
    }

}

void DrawToolGL::drawCross(const Vector3&p, float length, const Vec4f& colour)
{
    std::vector<sofa::defaulttype::Vector3> bounds;

    for ( unsigned int i=0 ; i<3 ; i++ )
    {
        sofa::defaulttype::Vector3 p0 = p;
        sofa::defaulttype::Vector3 p1 = p;

        p0[i] -= length;
        p1[i] += length;

        bounds.push_back(p0);
        bounds.push_back(p1);
    }
    drawLines(bounds, 1, colour);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawPlus ( const float& radius, const Vec<4,float>& colour, const int& subd)
{
    drawCylinder( Vector3(-1.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), radius, colour, subd);
    drawCylinder( Vector3(0.0, -1.0, 0.0), Vector3(0.0, 1.0, 0.0), radius, colour, subd);
    drawCylinder( Vector3(0.0, 0.0, -1.0), Vector3(0.0, 0.0, 1.0), radius, colour, subd);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::internalDrawPoint(const Vector3 &p, const Vec<4,float> &c)
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

void DrawToolGL::internalDrawPoint(const Vector3 &p, const Vector3 &n, const Vec<4,float> &c)
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


void DrawToolGL::drawPoint(const Vector3 &p, const Vec<4,float> &c)
{
    glBegin(GL_POINTS);
    internalDrawPoint(p,c);
    glEnd();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawPoint(const Vector3 &p, const Vector3 &n, const Vec<4,float> &c)
{
    glBegin(GL_POINTS);
    internalDrawPoint(p, n, c);
    glEnd();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::internalDrawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
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


void DrawToolGL::internalDrawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
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


void DrawToolGL::internalDrawTriangle( const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,
        const Vector3 &normal, const  Vec<4,float> &c)
{
    glNormalT(normal);
    glColor4fv(c.ptr());
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
}


void DrawToolGL::internalDrawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal)
{
    glNormalT(normal);
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
}

void DrawToolGL::drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal,
        const Vec<4,float> &c1, const Vec<4,float> &c2, const Vec<4,float> &c3)
{
    glBegin(GL_TRIANGLES);
    internalDrawTriangle(p1, p2, p3, normal, c1, c2, c3);
    glEnd();
}


void DrawToolGL::drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3,
        const Vec<4,float> &c1, const Vec<4,float> &c2, const Vec<4,float> &c3)
{
    glBegin(GL_TRIANGLES);
    internalDrawTriangle(p1, p2, p3, normal1, normal2, normal3, c1, c2, c3);
    glEnd();
}


void DrawToolGL::drawTriangle( const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,
        const Vector3 &normal, const  Vec<4,float> &c)
{
    glBegin(GL_TRIANGLES);
    internalDrawTriangle(p1, p2, p3, normal, c);
    glEnd();
}


void DrawToolGL::drawTriangle(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,
        const Vector3 &normal)
{
    glBegin(GL_TRIANGLES);
    internalDrawTriangle(p1, p2, p3, normal);
    glEnd();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::internalDrawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal)
{
    glNormalT(normal);
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
    glVertexNv<3>(p4.ptr());
}

void DrawToolGL::internalDrawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal, const Vec4f &c)
{
    glNormalT(normal);
    glColor4fv(c.ptr());
    glVertexNv<3>(p1.ptr());
    glVertexNv<3>(p2.ptr());
    glVertexNv<3>(p3.ptr());
    glVertexNv<3>(p4.ptr());
}

void DrawToolGL::internalDrawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal,
        const Vec4f &c1, const Vec4f &c2, const Vec4f &c3, const Vec4f &c4)
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

void DrawToolGL::internalDrawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3, const Vector3 &normal4,
        const Vec4f &c1, const Vec4f &c2, const Vec4f &c3, const Vec4f &c4)
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


void DrawToolGL::drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal,
        const Vec<4,float> &c1, const Vec<4,float> &c2, const Vec<4,float> &c3, const Vec<4,float> &c4)
{
    glBegin(GL_QUADS);
    internalDrawQuad(p1, p2, p3, p4, normal, c1, c2, c3, c4);
    glEnd();
}


void DrawToolGL::drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal1, const Vector3 &normal2, const Vector3 &normal3, const Vector3 &normal4,
        const Vec<4,float> &c1, const Vec<4,float> &c2, const Vec<4,float> &c3, const Vec<4,float> &c4)
{
    glBegin(GL_QUADS);
    internalDrawQuad(p1, p2, p3, p4, normal1, normal2, normal3, normal4, c1, c2, c3, c4);
    glEnd();
}


void DrawToolGL::drawQuad( const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal, const  Vec<4,float> &c)
{
    glBegin(GL_QUADS);
    internalDrawQuad(p1, p2, p3, p4, normal, c);
    glEnd();
}


void DrawToolGL::drawQuad(const Vector3 &p1,const Vector3 &p2,const Vector3 &p3,const Vector3 &p4,
        const Vector3 &normal)
{
    glBegin(GL_QUADS);
    internalDrawQuad(p1, p2, p3, p4, normal);
    glEnd();
}

void DrawToolGL::drawQuads(const std::vector<Vector3> &points, const Vec4f& colour)
{
    setMaterial(colour);
    glBegin(GL_QUADS);
    {
        for (unsigned int i=0; i<points.size()/4; ++i)
        {
            const Vector3& a = points[ 4*i+0 ];
            const Vector3& b = points[ 4*i+1 ];
            const Vector3& c = points[ 4*i+2 ];
            const Vector3& d = points[ 4*i+3 ];
            Vector3 n = cross((b-a),(c-a));
            n.normalize();
            internalDrawQuad(a,b,c,d,n,colour);
        }
    } glEnd();
    resetMaterial(colour);
}

void DrawToolGL::drawTetrahedron(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &p3, const Vec4f &colour)
{
    setMaterial(colour);
    glBegin(GL_TRIANGLES);
    {
        this->drawTriangle(p0,p1,p2, cross((p1-p0),(p2-p0)), colour);
        this->drawTriangle(p0,p1,p3, cross((p1-p0),(p3-p0)), colour);
        this->drawTriangle(p0,p2,p3, cross((p2-p0),(p3-p0)), colour);
        this->drawTriangle(p1,p2,p3, cross((p2-p1),(p3-p1)), colour);
    } glEnd();
    resetMaterial(colour);
}

void DrawToolGL::drawTetrahedra(const std::vector<Vector3> &points, const Vec4f &colour)
{
    setMaterial(colour);
    glBegin(GL_TRIANGLES);
    for (std::vector<Vector3>::const_iterator it = points.begin(), end = points.end(); it != end;)
    {
        const Vector3& p0 = *(it++);
        const Vector3& p1 = *(it++);
        const Vector3& p2 = *(it++);
        const Vector3& p3 = *(it++);

        //this->drawTetrahedron(p0,p1,p2,p3,colour); // not recommanded as it will call glBegin/glEnd <number of tetra> times
        this->drawTriangle(p0, p1, p2, cross((p1 - p0), (p2 - p0)), colour);
        this->drawTriangle(p0, p1, p3, cross((p1 - p0), (p3 - p0)), colour);
        this->drawTriangle(p0, p2, p3, cross((p2 - p0), (p3 - p0)), colour);
        this->drawTriangle(p1, p2, p3, cross((p2 - p1), (p3 - p1)), colour);
    }
    glEnd();
    resetMaterial(colour);
}

void DrawToolGL::drawScaledTetrahedra(const std::vector<Vector3> &points, const Vec4f &colour, const float scale)
{
    setMaterial(colour);
    glBegin(GL_TRIANGLES);
    for (std::vector<Vector3>::const_iterator it = points.begin(), end = points.end(); it != end;)
    {
        const Vector3& p0 = *(it++);
        const Vector3& p1 = *(it++);
        const Vector3& p2 = *(it++);
        const Vector3& p3 = *(it++);

        Vector3 center = (p0 + p1 + p2 + p3) / 4.0;

        Vector3 np0 = ((p0 - center)*scale) + center;
        Vector3 np1 = ((p1 - center)*scale) + center;
        Vector3 np2 = ((p2 - center)*scale) + center;
        Vector3 np3 = ((p3 - center)*scale) + center;

        //this->drawTetrahedron(p0,p1,p2,p3,colour); // not recommanded as it will call glBegin/glEnd <number of tetra> times
        this->drawTriangle(np0, np1, np2, cross((p1 - p0), (p2 - p0)), colour);
        this->drawTriangle(np0, np1, np3, cross((p1 - p0), (p3 - p0)), colour);
        this->drawTriangle(np0, np2, np3, cross((p2 - p0), (p3 - p0)), colour);
        this->drawTriangle(np1, np2, np3, cross((p2 - p1), (p3 - p1)), colour);
    }
    glEnd();
    resetMaterial(colour);
}


void DrawToolGL::drawHexahedron(const Vector3 &p0, const Vector3 &p1, const Vector3 &p2, const Vector3 &p3,
                                const Vector3 &p4, const Vector3 &p5, const Vector3 &p6, const Vector3 &p7,
                                const Vec4f &colour)
{
    //{{0,1,2,3}, {4,7,6,5}, {1,0,4,5},{1,5,6,2},  {2,6,7,3}, {0,3,7,4}}
    setMaterial(colour);
    glBegin(GL_QUADS);
    {
        this->drawQuad(p0, p1, p2, p3, cross((p1 - p0), (p2 - p0)), colour);
        this->drawQuad(p4, p7, p6, p5, cross((p7 - p5), (p6 - p5)), colour);
        this->drawQuad(p1, p0, p4, p5, cross((p0 - p1), (p4 - p1)), colour);
        this->drawQuad(p1, p5, p6, p2, cross((p5 - p1), (p6 - p1)), colour);
        this->drawQuad(p2, p6, p7, p3, cross((p6 - p2), (p7 - p2)), colour);
        this->drawQuad(p0, p3, p7, p4, cross((p3 - p0), (p7 - p0)), colour);
    } glEnd();
    resetMaterial(colour);
}

void DrawToolGL::drawHexahedra(const std::vector<Vector3> &points, const Vec4f& colour)
{
    setMaterial(colour);

    glBegin(GL_QUADS);
    for (std::vector<Vector3>::const_iterator it = points.begin(), end = points.end(); it != end;)
    {
        const Vector3& p0 = *(it++);
        const Vector3& p1 = *(it++);
        const Vector3& p2 = *(it++);
        const Vector3& p3 = *(it++);
        const Vector3& p4 = *(it++);
        const Vector3& p5 = *(it++);
        const Vector3& p6 = *(it++);
        const Vector3& p7 = *(it++);

        //this->drawHexahedron(p0,p1,p2,p3,p4,p5,p6,p7,colour); // not recommanded as it will call glBegin/glEnd <number of hexa> times
        this->drawQuad(p0, p1, p2, p3, cross((p1 - p0), (p2 - p0)), colour);
        this->drawQuad(p4, p7, p6, p5, cross((p7 - p5), (p6 - p5)), colour);
        this->drawQuad(p1, p0, p4, p5, cross((p0 - p1), (p4 - p1)), colour);
        this->drawQuad(p1, p5, p6, p2, cross((p5 - p1), (p6 - p1)), colour);
        this->drawQuad(p2, p6, p7, p3, cross((p6 - p2), (p7 - p2)), colour);
        this->drawQuad(p0, p3, p7, p4, cross((p3 - p0), (p7 - p0)), colour);
    }
    glEnd();
    resetMaterial(colour);
}

void DrawToolGL::drawScaledHexahedra(const std::vector<Vector3> &points, const Vec4f& colour, const float scale)
{
    setMaterial(colour);

    glBegin(GL_QUADS);
    for (std::vector<Vector3>::const_iterator it = points.begin(), end = points.end(); it != end;)
    {
        const Vector3& p0 = *(it++);
        const Vector3& p1 = *(it++);
        const Vector3& p2 = *(it++);
        const Vector3& p3 = *(it++);
        const Vector3& p4 = *(it++);
        const Vector3& p5 = *(it++);
        const Vector3& p6 = *(it++);
        const Vector3& p7 = *(it++);

        //barycenter
        Vector3 center = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7)/8.0;

        Vector3 np0 = ((p0 - center)*scale) + center;
        Vector3 np1 = ((p1 - center)*scale) + center;
        Vector3 np2 = ((p2 - center)*scale) + center;
        Vector3 np3 = ((p3 - center)*scale) + center;
        Vector3 np4 = ((p4 - center)*scale) + center;
        Vector3 np5 = ((p5 - center)*scale) + center;
        Vector3 np6 = ((p6 - center)*scale) + center;
        Vector3 np7 = ((p7 - center)*scale) + center;

        //this->drawHexahedron(p0,p1,p2,p3,p4,p5,p6,p7,colour); // not recommanded as it will call glBegin/glEnd <number of hexa> times
        this->drawQuad(np0, np1, np2, np3, cross((p1 - p0), (p2 - p0)), colour);
        this->drawQuad(np4, np7, np6, np5, cross((p7 - p5), (p6 - p5)), colour);
        this->drawQuad(np1, np0, np4, np5, cross((p0 - p1), (p4 - p1)), colour);
        this->drawQuad(np1, np5, np6, np2, cross((p5 - p1), (p6 - p1)), colour);
        this->drawQuad(np2, np6, np7, np3, cross((p6 - p2), (p7 - p2)), colour);
        this->drawQuad(np0, np3, np7, np4, cross((p3 - p0), (p7 - p0)), colour);
    }
    glEnd();
    resetMaterial(colour);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::drawSphere( const Vector3 &p, float radius)
{
    glPushMatrix();
    helper::gl::drawSphere(p, radius, 32, 16);
    glPopMatrix();
}

void DrawToolGL::drawEllipsoid(const Vector3 &p, const Vector3 &radii)
{
    glPushMatrix();
    helper::gl::drawEllipsoid(p, (float)radii[0], (float)radii[1], (float)radii[2], 32, 16);
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
    if (this->getLightEnabled()) enableLighting();
    else disableLighting();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DrawToolGL::setMaterial(const Vec<4,float> &colour)
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

void DrawToolGL::resetMaterial(const Vec<4,float> &colour)
{
    if (colour[3] < 1)
    {
        resetMaterial();
    }
}

void DrawToolGL::resetMaterial()
{
    glDisable(GL_BLEND);
    glDepthMask(1);
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

void DrawToolGL::translate(float x, float y, float z)
{
    glTranslatef(x, y, z);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::writeOverlayText( int x, int y, unsigned fontSize, const Vec4f &color, const char* text )
{
    GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport );

    static const float letterSize = 0.5;

    float scale = fontSize / letterSize;

    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);

    glPushAttrib( GL_LIGHTING_BIT );
    glEnable( GL_COLOR_MATERIAL );

    glPushAttrib( GL_ENABLE_BIT );

    glDisable(GL_CULL_FACE);

    glColor4f( color[0], color[1], color[2], color[3] );

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, viewport[2], viewport[3], 0 );

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glTranslated(x,y,0);

    glScalef( scale, scale, scale );

//    glLineWidth( fontSize/20.0f );

    helper::gl::GlText::textureDraw_Overlay(text);

    glPopAttrib(); // GL_ENABLE_BIT
    glPopAttrib(); // GL_LIGHTING_BIT

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

}

void DrawToolGL::enableBlending()
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void DrawToolGL::disableBlending()
{
    glDisable(GL_BLEND);
}

void DrawToolGL::enableLighting()
{
    glEnable(GL_LIGHTING);
}

void DrawToolGL::disableLighting()
{
    glDisable(GL_LIGHTING);
}

void DrawToolGL::enableDepthTest()
{
    glEnable(GL_DEPTH_TEST);
}

void DrawToolGL::disableDepthTest()
{
    glDisable(GL_DEPTH_TEST);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DrawToolGL::draw3DText(const Vector3 &p, float scale, const Vec4f &color, const char* text)
{
    glColor4fv(color.ptr());

    sofa::helper::gl::GlText::draw(text, p, (double)scale);
}

void DrawToolGL::draw3DText_Indices(const helper::vector<Vector3> &positions, float scale, const Vec4f &color)
{
    glColor4f(color[0], color[1], color[2], color[3]);

    sofa::helper::gl::GlText::textureDraw_Indices(positions, scale);
}

void DrawToolGL::saveLastState()
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
}

void DrawToolGL::restoreLastState()
{
    glPopAttrib();
}

void DrawToolGL::readPixels(int x, int y, int w, int h, float* rgb, float* z)
{
    if(rgb != NULL && sizeof(*rgb) == 3 * sizeof(float) * w * h)
        glReadPixels(x, y, w, h, GL_RGB, GL_FLOAT, rgb);

    if(z != NULL && sizeof(*z) == sizeof(float) * w * h)
        glReadPixels(x, y, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, z);
}

} // namespace visual

} // namespace core

} // namespace sofa
