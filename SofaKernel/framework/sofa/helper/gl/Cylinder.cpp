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
#include <sofa/helper/gl/Cylinder.h>

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
//GLuint Cylinder::displayList;
//GLUquadricObj *Cylinder::quadratic = NULL;
std::map < std::pair<std::pair<float,float>,float>, Cylinder* > Cylinder::CylinderMap;

void Cylinder::initDraw()
{
#ifdef PS3
	// No display lists
	if (quadratic==NULL)
	{
		quadratic=gluNewQuadric();
		gluQuadricNormals(quadratic, GLU_SMOOTH);
		gluQuadricTexture(quadratic, GL_TRUE);
	}
#else
    if (quadratic!=NULL) return;

    quadratic=gluNewQuadric();
    gluQuadricNormals(quadratic, GLU_SMOOTH);
    gluQuadricTexture(quadratic, GL_TRUE);

    displayList=glGenLists(1);

    glNewList(displayList, GL_COMPILE);
#endif

    glColor3f(1,1,0);
    //gluSphere(quadratic,l[0],quadricDiscretisation,quadricDiscretisation/2);

    if (length[0] > 0.0)
    {
        glRotatef(90,0,1,0);
        glTranslated(0,0,-length[0]/2);
        gluSphere(quadratic,length[0]/15,quadricDiscretisation,quadricDiscretisation/2);
        gluCylinder(quadratic,length[0]/15,length[0]/15,length[0],quadricDiscretisation,quadricDiscretisation);
        glTranslated(0,0,length[0]);
        gluSphere(quadratic,length[0]/15,quadricDiscretisation,quadricDiscretisation/2);
        glTranslated(0,0,-length[0]/2);
        glRotatef(-90,0,1,0);
    }
    if (length[1] > 0.0)
    {
        glRotatef(90,1,0,0);
        glTranslated(0,0,-length[1]/2);
        gluSphere(quadratic,length[1]/15,quadricDiscretisation,quadricDiscretisation/2);
        gluCylinder(quadratic,length[1]/15,length[1]/15,length[1],quadricDiscretisation,quadricDiscretisation);
        glTranslated(0,0,length[1]);
        gluSphere(quadratic,length[1]/15,quadricDiscretisation,quadricDiscretisation/2);
        glTranslated(0,0,-length[1]/2);
        glRotatef(-90,1,0,0);

    }
    if (length[2] > 0.0)
    {
        glTranslated(0,0,-length[2]/2);
        gluSphere(quadratic,length[2]/15,quadricDiscretisation,quadricDiscretisation/2);
        gluCylinder(quadratic,length[2]/15,length[2]/15,length[2],quadricDiscretisation,quadricDiscretisation);
        glTranslated(0,0,length[2]);
        gluSphere(quadratic,length[2]/15,quadricDiscretisation,quadricDiscretisation/2);
        glTranslated(0,0,-length[2]/2);
    }

#ifndef PS3
	glEndList();
#endif
}

void Cylinder::draw()
{
    initDraw();

    glPushMatrix();
    glPushAttrib(GL_ENABLE_BIT);

    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glMultMatrixd(matTransOpenGL);
    glCallList(displayList);

    glPopAttrib();
    glPopMatrix();
}

void Cylinder::update(const double *mat)
{
    std::copy(mat,mat+16, matTransOpenGL);
}

void Cylinder::update(const Vector3& center, const double orient[4][4])
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

void Cylinder::update(const Vector3& center, const Quaternion& orient)
{
    orient.writeOpenGlMatrix(matTransOpenGL);
    matTransOpenGL[12] = center[0];
    matTransOpenGL[13] = center[1];
    matTransOpenGL[14] = center[2];
}

Cylinder::Cylinder(SReal len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(Vector3(0,0,0),  Quaternion(1,0,0,0));
}

Cylinder::Cylinder(const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(Vector3(0,0,0),  Quaternion(1,0,0,0));
}

Cylinder::Cylinder(const Vector3& center, const Quaternion& orient, const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(center, orient);
}

Cylinder::Cylinder(const Vector3& center, const double orient[4][4], const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(center, orient);
}

Cylinder::Cylinder(const double *mat, const Vector3& len)
{
    quadratic = NULL;
    length = len;
    update(mat);
}

Cylinder::Cylinder(const Vector3& center, const Quaternion& orient, SReal len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(center, orient);
}
Cylinder::Cylinder(const Vector3& center, const double orient[4][4], SReal len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(center, orient);
}

Cylinder::Cylinder(const double *mat, SReal len)
{
    quadratic = NULL;
    length = Vector3(len,len,len);
    update(mat);
}

Cylinder::~Cylinder()
{
    if (quadratic != NULL)
        gluDeleteQuadric(quadratic);
}

Cylinder* Cylinder::get(const Vector3& len)
{
    Cylinder*& a = CylinderMap[std::make_pair(std::make_pair((float)len[0],(float)len[1]),(float)len[2])];
    if (a==NULL)
        a = new Cylinder(len);
    return a;
}

void Cylinder::draw(const Vector3& center, const Quaternion& orient, const Vector3& len)
{
    Cylinder* a = get(len);
    a->update(center, orient);
    a->draw();
}

void Cylinder::draw(const Vector3& center, const double orient[4][4], const Vector3& len)
{
    Cylinder* a = get(len);
    a->update(center, orient);
    a->draw();
}

void Cylinder::draw(const double *mat, const Vector3& len)
{
    Cylinder* a = get(len);
    a->update(mat);
    a->draw();
}

void Cylinder::draw(const Vector3& center, const Quaternion& orient, SReal len)
{
    Cylinder* a = get(Vector3(len,len,len));
    a->update(center, orient);
    a->draw();
}

void Cylinder::draw(const Vector3& center, const double orient[4][4], SReal len)
{
    Cylinder* a = get(Vector3(len,len,len));
    a->update(center, orient);
    a->draw();
}

void Cylinder::draw(const double *mat, SReal len)
{
    Cylinder* a = get(Vector3(len,len,len));
    a->update(mat);
    a->draw();
}

} // namespace gl

} // namespace helper

} // namespace sofa

