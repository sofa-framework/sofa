/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define FLEXIBLE_QuadraticTYPES_CPP

#include "../initFlexible.h"
#include "../types/QuadraticTypes.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/container/MechanicalObject.inl>

namespace sofa
{

namespace component
{

namespace container
{


// ==========================================================================
// Draw Specializations


template <>
void MechanicalObject<Quadratic3dTypes>::draw(const core::visual::VisualParams* vparams)
{
    Mat<4,4, GLfloat> modelviewM;
    Vec<3, SReal> sceneMinBBox, sceneMaxBBox;
    sofa::simulation::Node* context;
    if ( showIndices.getValue() )
    {
        context = dynamic_cast<sofa::simulation::Node*> ( this->getContext() );
        glColor3f ( 1.0,1.0,1.0 );
        glDisable ( GL_LIGHTING );
        sofa::simulation::getSimulation()->computeBBox ( ( sofa::simulation::Node* ) context, sceneMinBBox.ptr(), sceneMaxBBox.ptr() );
        float scale = ( sceneMaxBBox - sceneMinBBox ).norm() * showIndicesScale.getValue();

        for ( int i=0 ; i< vsize ; i++ )
        {
            std::ostringstream oss;
            oss << i;
            std::string tmp = oss.str();
            const char* s = tmp.c_str();
            //glVertex3f(getPX(i),getPY(i),getPZ(i) );
            glPushMatrix();

            glTranslatef ( getPX ( i ), getPY ( i ), getPZ ( i ) );
            glScalef ( scale,scale,scale );

            // Makes text always face the viewer by removing the scene rotation
            // get the current modelview matrix
            glGetFloatv ( GL_MODELVIEW_MATRIX , modelviewM.ptr() );
            modelviewM.transpose();

            Vec3d temp ( getPX ( i ), getPY ( i ), getPZ ( i ) );
            temp = modelviewM.transform ( temp );

            //glLoadMatrixf(modelview);
            glLoadIdentity();

            glTranslatef ( temp[0], temp[1], temp[2] );
            glScalef ( scale,scale,scale );

            while ( *s )
            {
                glutStrokeCharacter ( GLUT_STROKE_ROMAN, *s );
                s++;
            }

            glPopMatrix();
        }
    }

    if ( showObject.getValue() )
    {
        glPushAttrib ( GL_LIGHTING_BIT );
        glDisable ( GL_LIGHTING );
        const Quadratic3dTypes::VecCoord& x = ( *getX() );
        const float& scale = showObjectScale.getValue();
        for ( int i=0; i<this->getSize(); i++ )
        {
            vparams->drawTool()->pushMatrix();
            float glTransform[16];
            x[i].writeOpenGlMatrix ( glTransform );
            vparams->drawTool()->multMatrix( glTransform );
            vparams->drawTool()->scale ( scale);
            vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ) );
            vparams->drawTool()->popMatrix();
        }
        glPopAttrib();
    }
}




// ==========================================================================
// Instanciation

SOFA_DECL_CLASS ( QuadraticMechanicalObject )

using namespace sofa::defaulttype;

int QuadraticMechanicalObjectClass = core::RegisterObject ( "mechanical state vectors" )
#ifndef SOFA_FLOAT
        .add< MechanicalObject<Quadratic3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<Quadratic3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_Flexible_API MechanicalObject<Quadratic3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API MechanicalObject<Quadratic3fTypes>;
#endif




} // namespace container

} // namespace component

} // namespace sofa
