/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/core/ObjectFactory.h>

#include "OglSceneFrame.h"

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(OglSceneFrame)

int OglSceneFrameClass = core::RegisterObject("Display a frame at the corner of the scene view")
        .add< OglSceneFrame >()
        ;

using namespace sofa::defaulttype;

void OglSceneFrame::init()
{
    updateVisual();
}

void OglSceneFrame::reinit()
{
    updateVisual();
}


void OglSceneFrame::updateVisual()
{

}

void OglSceneFrame::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!drawFrame.getValue()) return;

    glPushAttrib( GL_ALL_ATTRIB_BITS);

    const Viewport& viewport = vparams->viewport();

    switch(alignment.getValue().getSelectedId())
    {
        case 0:
        default:
            glViewport(0,0,150,150);
            glScissor(0,0,150,150);
            break;
        case 1:
            glViewport(viewport[2]-150,0,150,150);
            glScissor(viewport[2]-150,0,150,150);
            break;
        case 2:
            glViewport(viewport[2]-150,viewport[3]-150,150,150);
            glScissor(viewport[2]-150,viewport[3]-150,150,150);
            break;
        case 3:
            glViewport(0,viewport[3]-150,150,150);
            glScissor(0,viewport[3]-150,150,150);
            break;
    }




    glEnable(GL_SCISSOR_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glClearColor (1.0f, 1.0f, 1.0f, 0.0f);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluPerspective(60.0, 1.0, 0.5, 10.0);

    GLdouble matrix[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, matrix);

    matrix[12] = 0;
    matrix[13] = 0;
    matrix[14] = -3;
    matrix[15] = 1;

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadMatrixd(matrix);

    if (!quadratic)
    {
        quadratic = gluNewQuadric();

        gluQuadricNormals(quadratic, GLU_SMOOTH);
        gluQuadricTexture(quadratic, GL_TRUE);
    }

    glDisable(GL_LIGHTING);

    if (quadratic)
    {

        switch (style.getValue().getSelectedId())
        {
            case 0:
            default:
                //X axis
                glColor4f( 1.0f, 0.0f, 0.0f, 1.0f );
                glRotatef(90,0,1,0);
                gluCylinder(quadratic,0.1f,0.1f,1.0f,32,32);
                glRotatef(-90,0,1,0);

                glTranslated(1.0f,0,0);
                glRotatef(90,0,1,0);
                gluDisk(quadratic,0,0.2f,32,32);
                gluCylinder(quadratic,0.2f,0,0.2f,32,32);
                glRotatef(-90,0,1,0);
                glTranslated(-1.0f,0,0);

                //Y axis
                glColor4f( 0.0f, 1.0f, 0.0f, 1.0f );
                glRotatef(-90,1,0,0);
                gluCylinder(quadratic,0.1f,0.1f,1.0f,32,32);
                glRotatef(90,1,0,0);

                glTranslated(0.0f, 1.0f, 0);
                glRotatef(-90,1,0,0);
                gluDisk(quadratic,0,0.2f,32,32);
                gluCylinder(quadratic,0.2f,0,0.2f,32,32);
                glRotatef(90,1,0,0);
                glTranslated(0.0f, -1.0f, 0.0f);

                //Z axis
                glColor4f( 0.0f, 0.0f, 1.0f, 1.0f );
                gluCylinder(quadratic,0.1f,0.1f,1.0f,32,32);

                glTranslated(0.0f, 0.0f, 1.0f);
                gluDisk(quadratic,0,0.2f,32,32);
                gluCylinder(quadratic,0.2f,0,0.2f,32,32);
                glTranslated(0.0f, 0.0f, -1.0f);

                break;

            case 1:
                //X axis
                glColor4f( 1.0f, 0.0f, 0.0f, 1.0f );
                glRotatef(90,0,1,0);
                gluCylinder(quadratic,0.05f,0.05f,1.0f,32,32);
                glRotatef(-90,0,1,0);

                //Y axis
                glColor4f( 0.0f, 1.0f, 0.0f, 1.0f );
                glRotatef(-90,1,0,0);
                gluCylinder(quadratic,0.05f,0.05f,1.0f,32,32);
                glRotatef(90,1,0,0);

                //Z axis
                glColor4f( 0.0f, 0.0f, 1.0f, 1.0f );
                gluCylinder(quadratic,0.05f,0.05f,1.0f,32,32);

                break;

            case 2:
                glColor4f(0.5f, 0.5f, 0.5f, 1.0f);

                GLfloat s = 0.25f;

                glBegin(GL_QUADS);
                    glVertex3f(-s, -s,  s);
                    glVertex3f( s, -s,  s);
                    glVertex3f( s,  s,  s);
                    glVertex3f(-s,  s,  s);

                    glVertex3f(-s, -s, -s);
                    glVertex3f(-s,  s, -s);
                    glVertex3f( s,  s, -s);
                    glVertex3f( s, -s, -s);

                    glVertex3f(-s,  s, -s);
                    glVertex3f(-s,  s,  s);
                    glVertex3f( s,  s,  s);
                    glVertex3f( s,  s, -s);

                    glVertex3f(-s, -s, -s);
                    glVertex3f( s, -s, -s);
                    glVertex3f( s, -s,  s);
                    glVertex3f(-s, -s,  s);

                    glVertex3f( s, -s, -s);
                    glVertex3f( s,  s, -s);
                    glVertex3f( s,  s,  s);
                    glVertex3f( s, -s,  s);

                    glVertex3f(-s, -s, -s);
                    glVertex3f(-s, -s,  s);
                    glVertex3f(-s,  s,  s);
                    glVertex3f(-s,  s, -s);
                glEnd();

                //X axis
                glColor4f( 1.0f, 0.0f, 0.0f, 1.0f );
                glTranslated(s,0,0);
                glRotatef(90,0,1,0);
                gluCylinder(quadratic,0,s,s*3.0f,32,32);
                glRotatef(-90,0,1,0);
                glTranslated(-s,0,0);

                glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
                glTranslated(-s,0,0);
                glRotatef(-90,0,1,0);
                gluCylinder(quadratic,0,s,s*3.0f,32,32);
                glRotatef(90,0,1,0);
                glTranslated(s,0,0);

                //Y axis
                glColor4f( 0.0f, 1.0f, 0.0f, 1.0f );
                glTranslated(0.0f, s, 0);
                glRotatef(-90,1,0,0);
                gluCylinder(quadratic,0.0f,s,s*3.0f,32,32);
                glRotatef(90,1,0,0);
                glTranslated(0.0f, -s, 0.0f);

                glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
                glTranslated(0.0f, -s, 0);
                glRotatef(90,1,0,0);
                gluCylinder(quadratic,0.0f,s,s*3.0f,32,32);
                glRotatef(-90,1,0,0);
                glTranslated(0.0f, s, 0.0f);

                //Z axis
                glColor4f( 0.0f, 0.0f, 1.0f, 1.0f );
                glTranslated(0.0f, 0.0f, s);
                gluCylinder(quadratic,0.0f,s,s*3.0f,32,32);
                glTranslated(0.0f, 0.0f, -s);

                glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
                glTranslated(0.0f, 0.0f, -s);
                glRotatef(-180,0,1,0);
                gluCylinder(quadratic,0.0f,s,s*3.0f,32,32);
                glRotatef(180,0,1,0);
                glTranslated(0.0f, 0.0f, s);


                break;
        }
    }

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glPopAttrib();
    glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
#endif

}

} // namespace visualmodel

} // namespace component

} // namespace sofa
