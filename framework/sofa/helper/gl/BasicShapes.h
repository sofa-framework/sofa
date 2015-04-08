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
#ifndef SOFA_HELPER_GL_BASICSHAPES_H
#define SOFA_HELPER_GL_BASICSHAPES_H

#ifndef SOFA_NO_OPENGL

#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/glu.h>
#include <math.h>

namespace sofa
{

namespace helper
{

namespace gl
{

template <typename V>
void drawCone(const V& p1, const V& p2, const float& radius1, const float& radius2, const int subd=8)
{
    V tmp = p2-p1;

    /* create Vectors p and q, co-planar with the cylinder's cross-sectional disk */
    V p=tmp;
    if (p[0] == 0.0 && p[1] == 0.0)
        p[0] += 1.0;
    else
        p[2] += 1.0;
    V q;
    q = p.cross(tmp);
    p = tmp.cross(q);
    /* do the normalization outside the segment loop */
    p.normalize();
    q.normalize();

    int i2;
    float theta, st, ct;
    /* build the cylinder from rectangular subd */
    glBegin(GL_QUAD_STRIP);
    for (i2=0 ; i2<=subd ; i2++)
    {
        /* sweep out a circle */
        theta =  (float)(i2 * 2.0 * M_PI / subd);
        st = (float)sin(theta);
        ct = (float)cos(theta);
        /* construct normal */
        tmp = p*ct+q*st;
        /* set the normal for the two subseqent points */
        helper::gl::glNormalT(tmp);
        /* point on disk 1 */
        V w(p1);
        w += tmp*radius1;
        helper::gl::glVertexT(w);
        /* point on disk 2 */
        w=p2;
        w += tmp*radius2;
        helper::gl::glVertexT(w);
    }
    glEnd();
}


template <typename V>
void drawCylinder(const V& p1, const V& p2, const float& rad, const int subd=8)
{
    drawCone( p1,p2,rad,rad,subd);
}


template <typename V>
void drawArrow(const V& p1, const V& p2, const float& rad, const int subd=8)
{
    V p3 = p1*.2+p2*.8;
    drawCylinder( p1,p3,rad,subd);
    drawCone( p3,p2,rad*2.5f,0.f,subd);
}


template <typename V>
void drawSphere(const V& center, const float& rad, const int subd1=8, const int subd2=8)
{
    GLUquadricObj*	sphere = gluNewQuadric();
    gluQuadricDrawStyle(sphere, GLU_FILL);
    gluQuadricOrientation(sphere, GLU_OUTSIDE);
    gluQuadricNormals(sphere, GLU_SMOOTH);
    glPushMatrix();
    helper::gl::glTranslateT( center );
    gluSphere(sphere,2.0*rad,subd1,subd2);
    glPopMatrix();
    // 		delete sphere;
}

template <typename V>
void drawWireSphere(const V& center, const float& rad, const int subd1=8, const int subd2=8)
{
    GLUquadricObj*	sphere = gluNewQuadric();
    gluQuadricDrawStyle(sphere, GLU_LINE);
    gluQuadricOrientation(sphere, GLU_OUTSIDE);
    glPushMatrix();
    helper::gl::glTranslateT( center );
    gluSphere(sphere,2.0*rad,subd1,subd2);
    glPopMatrix();
    // 		delete sphere;
}

template <typename V>
void drawTorus(const float* coordinateMatrix, const float& bodyRad=0.0,  const float& rad=1.0, const int precision=20,
               const V& color=sofa::defaulttype::Vec3i(255,215,180))
{
    GLUquadricObj* disk = gluNewQuadric();
    glColor3ub(color.x(), color.y(), color.z());
    gluQuadricDrawStyle(disk, GLU_FILL);
    gluQuadricOrientation(disk, GLU_OUTSIDE);
    gluQuadricNormals(disk, GLU_SMOOTH);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(coordinateMatrix);
    //gluDisk(disk, 2.0*bodyRad, 2.0*rad, 10, 10);

    float rr=1.5f*bodyRad;
    double dv=2*M_PI/precision;
    double dw=2*M_PI/precision;
    double v=0.0f;
    double w=0.0f;

    while(w < 2*M_PI+dw)
    {
        v=0.0f;
        glBegin(GL_TRIANGLE_STRIP);
        // inner loop
        while(v<2*M_PI+dv)
        {
            glNormal3d( (rad+rr*cos(v))*cos(w)-(rad+bodyRad*cos(v))*cos(w),
                        (rad+rr*cos(v))*sin(w)-(rad+bodyRad*cos(v))*sin(w),
                        (rr*sin(v)-bodyRad*sin(v)));
            glVertex3d((rad+bodyRad*cos(v))*cos(w),
                       (rad+bodyRad*cos(v))*sin(w),
                       bodyRad*sin(v));
            glNormal3d( (rad+rr*cos(v+dv))*cos(w+dw)-(rad+bodyRad*cos(v+dv))*cos(w+dw),
                        (rad+rr*cos(v+dv))*sin(w+dw)-(rad+bodyRad*cos(v+dv))*sin(w+dw),
                        rr*sin(v+dv)-bodyRad*sin(v+dv));
            glVertex3d((rad+bodyRad*cos(v+dv))*cos(w+dw),
                       (rad+bodyRad*cos(v+dv))*sin(w+dw),
                       bodyRad*sin(v+dv));
            v+=dv;
        } // inner loop
        glEnd();
        w+=dw;
    }
    glPopMatrix();
}

template <typename V>
void drawEmptyParallelepiped(const V& vert1, const V& vert2, const V& vert3, const V& vert4, const V& vecFromFaceToOppositeFace, const float& rad=1.0, const int precision=8,
                             const V& color=sofa::defaulttype::Vec3i(255,0,0))
{
	GLUquadricObj* parallelepiped = gluNewQuadric();
    glColor3ub(255, 255, 255);
    gluQuadricDrawStyle(parallelepiped, GLU_FILL);
    gluQuadricOrientation(parallelepiped, GLU_OUTSIDE);
    gluQuadricNormals(parallelepiped, GLU_SMOOTH);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

	//Vertices of the parallelepiped
	drawSphere(vert1,2.0*rad);
	drawSphere(vert2,2.0*rad);
	drawSphere(vert3,2.0*rad);
	drawSphere(vert4,2.0*rad);
	drawSphere(vert1 + vecFromFaceToOppositeFace,2.0*rad);
	drawSphere(vert2 + vecFromFaceToOppositeFace,2.0*rad);
	drawSphere(vert3 + vecFromFaceToOppositeFace,2.0*rad);
	drawSphere(vert4 + vecFromFaceToOppositeFace,2.0*rad);

	glColor3ub(color.x(), color.y(), color.z());
	//First face
	drawCylinder(vert1,vert2,rad,precision);
	drawCylinder(vert2,vert3,rad,precision);
	drawCylinder(vert3,vert4,rad,precision);
	drawCylinder(vert4,vert1,rad,precision);
	
	//The opposite face
	drawCylinder(vert1 + vecFromFaceToOppositeFace,vert2 + vecFromFaceToOppositeFace,rad,precision);
	drawCylinder(vert2 + vecFromFaceToOppositeFace,vert3 + vecFromFaceToOppositeFace,rad,precision);
	drawCylinder(vert3 + vecFromFaceToOppositeFace,vert4 + vecFromFaceToOppositeFace,rad,precision);
	drawCylinder(vert4 + vecFromFaceToOppositeFace,vert1 + vecFromFaceToOppositeFace,rad,precision);
	
	//Connect the two faces
	drawCylinder(vert1,vert1 + vecFromFaceToOppositeFace,rad,precision);
	drawCylinder(vert2,vert2 + vecFromFaceToOppositeFace,rad,precision);
	drawCylinder(vert3,vert3 + vecFromFaceToOppositeFace,rad,precision);
	drawCylinder(vert4,vert4 + vecFromFaceToOppositeFace,rad,precision);

	glPopMatrix();
}

} //gl
} //helper
} //sofa

#endif /* SOFA_NO_OPENGL */

#endif
