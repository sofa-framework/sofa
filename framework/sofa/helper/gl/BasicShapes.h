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

#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/glu.h>
#include <math.h>

namespace sofa
{

namespace helper
{

namespace gl
{

using namespace sofa::defaulttype;


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


}
}
}
#endif
