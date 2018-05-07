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
#include <sofa/helper/gl/Axis.h>

#include <sofa/helper/system/gl.h>

#include <cassert>
#include <algorithm>
#include <iostream>


namespace sofa
{

namespace helper
{

namespace gl
{

static const int quadricDiscretisation = 16;
//GLuint Axis::displayList;
//GLUquadricObj *Axis::quadratic = NULL;
std::map < std::pair<std::pair<float,float>,float>, Axis* > Axis::axisMap; // great idea but no more valid when creating a new opengl context when switching sofa viewer

void Axis::initDraw()
{
    if (quadratic) return;

    Vector3 L= length;
    SReal Lmin = L[0];
    if (L[1]<Lmin) Lmin = L[1];
    if (L[2]<Lmin) Lmin = L[2];
    SReal Lmax = L[0];
    if (L[1]>Lmax) Lmax = L[1];
    if (L[2]>Lmax) Lmax = L[2];
    if (Lmax > Lmin*2 && Lmin > 0.0)
        Lmax = Lmin*2;
    if (Lmax > Lmin*2)
        Lmin = Lmax/(SReal)1.414;
    Vector3 l(Lmin / (SReal)10, Lmin / (SReal)10, Lmin / (SReal)10);
    Vector3 lc(Lmax / (SReal)5, Lmax / (SReal)5, Lmax / (SReal)5); // = L / 5;
    Vector3 Lc = lc;


	if(quadratic==NULL)

    quadratic=gluNewQuadric();
    
	gluQuadricNormals(quadratic, GLU_SMOOTH);
    gluQuadricTexture(quadratic, GL_TRUE);
	
    displayLists=glGenLists(3);

    glNewList(displayLists, GL_COMPILE);



    // Center
    gluSphere(quadratic,l[0],quadricDiscretisation,quadricDiscretisation/2);

    if (L[0] > 0.0)
    {
        // X Axis
//        glColor3f(1,0,0);
        glRotatef(90,0,1,0);
        gluCylinder(quadratic,l[0],l[0],L[0],quadricDiscretisation,quadricDiscretisation);
        glRotatef(-90,0,1,0);

        glTranslated(L[0],0,0);
        glRotatef(90,0,1,0);
        gluDisk(quadratic,0,lc[0],quadricDiscretisation,quadricDiscretisation);
        gluCylinder(quadratic,lc[0],0,Lc[0],quadricDiscretisation,quadricDiscretisation);
        glRotatef(-90,0,1,0);
        glTranslated(-L[0],0,0);
    }

    glEndList();




    glNewList(displayLists+1, GL_COMPILE);

    if (L[1] > 0.0)
    {
        // Y Axis
//        glColor3f(0,1,0);
        glRotatef(-90,1,0,0);
        gluCylinder(quadratic,l[1],l[1],L[1],quadricDiscretisation,quadricDiscretisation);
        glRotatef(90,1,0,0);

        glTranslated(0,L[1],0);
        glRotatef(-90,1,0,0);
        gluDisk(quadratic,0,lc[1],quadricDiscretisation,quadricDiscretisation);
        gluCylinder(quadratic,lc[1],0,Lc[1],quadricDiscretisation,quadricDiscretisation);
        glRotatef(90,1,0,0);
        glTranslated(0,-L[1],0);
    }
	
    glEndList();

    glNewList(displayLists+2, GL_COMPILE);

    if (L[2] > 0.0)
    {
        // Z Axis
//        glColor3f(0,0,1);
        gluCylinder(quadratic,l[2],l[2],L[2],quadricDiscretisation,quadricDiscretisation);

        glTranslated(0,0,L[2]);
        gluDisk(quadratic,0,lc[2],quadricDiscretisation,quadricDiscretisation);
        gluCylinder(quadratic,lc[2],0,Lc[2],quadricDiscretisation,quadricDiscretisation);
        glTranslated(0,0,-L[2]);
    }
    glEndList();
}

void Axis::draw( const Vec4f& colorX, const Vec4f& colorY, const Vec4f& colorZ )
{
    initDraw();

    glPushAttrib(GL_LIGHTING_BIT);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);

    glPushMatrix();

    glMultMatrixd(matTransOpenGL);

    // X Axis

#ifndef PS3
	glColor4f( colorX[0], colorX[1], colorX[2], colorX[3] );
    glCallList(displayLists);
#else
	// No static list on PS3 yet
	Vector3 L= length;
    SReal Lmin = L[0];
    if (L[1]<Lmin) Lmin = L[1];
    if (L[2]<Lmin) Lmin = L[2];
    SReal Lmax = L[0];
    if (L[1]>Lmax) Lmax = L[1];
    if (L[2]>Lmax) Lmax = L[2];
    if (Lmax > Lmin*2 && Lmin > 0.0)
        Lmax = Lmin*2;
    if (Lmax > Lmin*2)
        Lmin = Lmax/(SReal)1.414;
    Vector3 l(Lmin / (SReal)10, Lmin / (SReal)10, Lmin / (SReal)10);
    Vector3 lc(Lmax / (SReal)5, Lmax / (SReal)5, Lmax / (SReal)5); // = L / 5;
    Vector3 Lc = lc;
	glColor3f( colorX[0], colorX[1], colorX[2] );
	    // Center
    gluSphere(quadratic,l[0],quadricDiscretisation,quadricDiscretisation/2);

    if (L[0] > 0.0)
    {
        // X Axis
        glRotatef(90,0,1,0);
        gluCylinder(quadratic,l[0],l[0],L[0],quadricDiscretisation,quadricDiscretisation);
        glRotatef(-90,0,1,0);

        glTranslated(L[0],0,0);
        glRotatef(90,0,1,0);
        gluDisk(quadratic,0,lc[0],quadricDiscretisation,quadricDiscretisation);
        gluCylinder(quadratic,lc[0],0,Lc[0],quadricDiscretisation,quadricDiscretisation);
        glRotatef(-90,0,1,0);
        glTranslated(-L[0],0,0);
    }
#endif
   
	// Y Axis
#ifndef PS3
	glColor4f( colorY[0], colorY[1], colorY[2], colorY[3] );
	glCallList(displayLists+1);
#else
	glColor3f( colorY[0], colorY[1], colorY[2] );
	if (L[1] > 0.0)
    {
        // Y Axis
        glRotatef(-90,1,0,0);
        gluCylinder(quadratic,l[1],l[1],L[1],quadricDiscretisation,quadricDiscretisation);
        glRotatef(90,1,0,0);

        glTranslated(0,L[1],0);
        glRotatef(-90,1,0,0);
        gluDisk(quadratic,0,lc[1],quadricDiscretisation,quadricDiscretisation);
        gluCylinder(quadratic,lc[1],0,Lc[1],quadricDiscretisation,quadricDiscretisation);
        glRotatef(90,1,0,0);
        glTranslated(0,-L[1],0);
    }
#endif
   
	// Z Axis
#ifndef PS3
	glColor4f( colorZ[0], colorZ[1], colorZ[2], colorZ[3] );
    glCallList(displayLists+2);
#else
	glColor3f( colorZ[0], colorZ[1], colorZ[2] );
	if (L[2] > 0.0)
    {
        // Z Axis
        gluCylinder(quadratic,l[2],l[2],L[2],quadricDiscretisation,quadricDiscretisation);
        glTranslated(0,0,L[2]);
        gluDisk(quadratic,0,lc[2],quadricDiscretisation,quadricDiscretisation);
        gluCylinder(quadratic,lc[2],0,Lc[2],quadricDiscretisation,quadricDiscretisation);
        glTranslated(0,0,-L[2]);
    }
#endif


    glPopMatrix();

    glPopAttrib();

}

void Axis::update(const double *mat)
{
    std::copy(mat,mat+16, matTransOpenGL);
}

void Axis::update(const Vector3& center, const double orient[4][4])
{
    matTransOpenGL[0] = orient[0][0];
    matTransOpenGL[1] = orient[0][1];
    matTransOpenGL[2] = orient[0][2];
    matTransOpenGL[3] = 0;

    matTransOpenGL[4] = orient[1][0];
    matTransOpenGL[5] = orient[1][1];
    matTransOpenGL[6] = orient[1][2];
    matTransOpenGL[7] = 0;

    matTransOpenGL[8] = orient[2][0];
    matTransOpenGL[9] = orient[2][1];
    matTransOpenGL[10]= orient[2][2];
    matTransOpenGL[11] = 0;

    matTransOpenGL[12] = center[0];
    matTransOpenGL[13] = center[1];
    matTransOpenGL[14] = center[2];
    matTransOpenGL[15] = 1;
}

void Axis::update(const Vector3& center, const Quaternion& orient)
{
    orient.writeOpenGlMatrix(matTransOpenGL);
    matTransOpenGL[12] = center[0];
    matTransOpenGL[13] = center[1];
    matTransOpenGL[14] = center[2];
}

Axis::Axis(SReal len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(Vector3(0,0,0),  Quaternion(1,0,0,0));
}

Axis::Axis(const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(Vector3(0,0,0),  Quaternion(1,0,0,0));
}

Axis::Axis(const Vector3& center, const Quaternion& orient, const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(center, orient);
}

Axis::Axis(const Vector3& center, const double orient[4][4], const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(center, orient);
}

Axis::Axis(const double *mat, const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(mat);
}

Axis::Axis(const Vector3& center, const Quaternion& orient, SReal len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(center, orient);
}
Axis::Axis(const Vector3& center, const double orient[4][4], SReal len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(center, orient);
}

Axis::Axis(const double *mat, SReal len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(mat);
}

Axis::~Axis()
{
    if (quadratic != NULL)
        gluDeleteQuadric(quadratic);
}

Axis* Axis::get(const Vector3& len)
{
    Axis*& a = axisMap[std::make_pair(std::make_pair((float)len[0],(float)len[1]),(float)len[2])];
    if (a==NULL)
        a = new Axis(len);
    return a;
}

void Axis::draw(const Vector3& center, const Quaternion& orient, const Vector3& len, const Vec4f& colorX, const Vec4f& colorY, const Vec4f& colorZ )
{
    Axis* a = get(len);
    a->update(center, orient);
    a->draw( colorX, colorY, colorZ );
}

void Axis::draw(const Vector3& center, const double orient[4][4], const Vector3& len, const Vec4f& colorX, const Vec4f& colorY, const Vec4f& colorZ)
{
    Axis* a = get(len);
    a->update(center, orient);
    a->draw( colorX, colorY, colorZ );
}

void Axis::draw(const double *mat, const Vector3& len, const Vec4f& colorX, const Vec4f& colorY, const Vec4f& colorZ)
{
    Axis* a = get(len);
    a->update(mat);
    a->draw( colorX, colorY, colorZ );
}

void Axis::draw(const Vector3& center, const Quaternion& orient, SReal len, const Vec4f& colorX, const Vec4f& colorY, const Vec4f& colorZ)
{
    Axis* a = get(Vector3(len,len,len));
    a->update(center, orient);
    a->draw( colorX, colorY, colorZ );
}

void Axis::draw(const Vector3& center, const double orient[4][4], SReal len, const Vec4f& colorX, const Vec4f& colorY, const Vec4f& colorZ)
{
    Axis* a = get(Vector3(len,len,len));
    a->update(center, orient);
    a->draw( colorX, colorY, colorZ );
}

void Axis::draw(const double *mat, SReal len, const Vec4f& colorX, const Vec4f& colorY, const Vec4f& colorZ)
{
    Axis* a = get(Vector3(len,len,len));
    a->update(mat);
    a->draw( colorX, colorY, colorZ );
}

void Axis::draw(const Vector3& p1, const Vector3& p2, const double& r )
{
    Vector3 v = p2-p1;
    Axis::draw(p1, p1+v*0.9, r,r );
    Axis::draw(p1+v*0.9,p2, 2.0*r,0.0 );
}

void Axis::draw(const Vector3& p1, const Vector3& p2, const double& r1, const double& r2 )
{
    int i;
    double theta;
    Vec3d n,p,q,perp;

    double theta2 = 2*3.141592653;
    double m = 16; //precision

    /* Normal pointing from p1 to p2 */
    n = p1-p2;

    /*
    Create two perpendicular vectors perp and q
    on the plane of the disk:
    */

    if      (n[1] != 0 || n[2] != 0)  perp = Vec3d(1,0,0);
    else                              perp = Vec3d(0,1,0);


    q = perp.cross(n);
    perp = n.cross(q);
    perp.normalize();
    q.normalize();

    glBegin(GL_QUAD_STRIP);
    for (i=0; i<=m; i++)
    {
        theta = i * (theta2) / m;

        n = perp*cos(theta) + q*sin(theta);
        n.normalize();

        p = p1 + n*r1;
        glNormal3d(n[0],n[1],n[2]);   glTexCoord2d(i/(double)m,0.0);
        glVertex3d(p[0],p[1],p[2]);

        p = p2 + n*r2;
        glNormal3d(n[0],n[1],n[2]);   glTexCoord2d(i/(double)m,1.0);
        glVertex3d(p[0],p[1],p[2]);

    }
    glEnd();

    glBegin(GL_TRIANGLE_FAN);
    glVertex3d(p1[0],p1[1],p1[2]);
    for (i=0; i<=m && r1 != 0; i++)
    {
        theta = i * (theta2) / m;

        n = perp*cos(theta) + q*sin(theta);
        n.normalize();

        p = p1 + n*r1;
        glNormal3d(n[0],n[1],n[2]);  glTexCoord2d(i/(double)m,0.0);
        glVertex3d(p[0],p[1],p[2]);
    }
    glEnd();

    glBegin(GL_TRIANGLE_FAN);
    glVertex3d(p2[0],p2[1],p2[2]);
    for (i=0; i<=m && r2 != 0; i++)
    {
        theta = i * (theta2) / m;

        n = perp*cos(theta) + q*sin(theta);
        n.normalize();

        p = p2 + n*r2;
        glNormal3d(n[0],n[1],n[2]);  glTexCoord2d(i/(double)m,1.0);
        glVertex3d(p[0],p[1],p[2]);
    }
    glEnd();

}

} // namespace gl

} // namespace helper

} // namespace sofa

