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
#define FRAME_FRAMEMECHANICALOBJECT_CPP

#include "FrameMechanicalObject.h"
#include <SofaBaseMechanics/MechanicalObject.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/template.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace container
{


template <>
void MechanicalObject<Affine3dTypes>::draw(const core::visual::VisualParams* vparams)
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
        const Affine3dTypes::VecCoord& x = ( *getX() );
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

//            template<> class MechanicalObjectInternalData<DeformationGradient331dTypes>
//            {
//            public:
//                typedef MechanicalObject<DeformationGradient331dTypes> Main;
//
//                Data<bool> showDefTensors;
//                Data<bool> showDefTensorsValues;
//                Data<double> showDefTensorScale;
//
//                MechanicalObjectInternalData ( Main* mObj = NULL )
//                        : showDefTensors ( mObj->initData ( &showDefTensors, false, "showDefTensors","show computed deformation tensors." ) )
//                        , showDefTensorsValues ( mObj->initData ( &showDefTensorsValues, false, "showDefTensorsValues","Show Deformation Tensors Values." ) )
//                        , showDefTensorScale ( mObj->initData ( &showDefTensorScale, 1.0, "showDefTensorScale","deformation tensor scale." ) )
//                {}
//            };

template <>
void MechanicalObject<DeformationGradient331dTypes>::draw(const core::visual::VisualParams* vparams)
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
        const DeformationGradient331dTypes::VecCoord& x = ( *getX() );
        const float& scale = showObjectScale.getValue();
        for ( int i=0; i<this->getSize(); i++ )
        {
            vparams->drawTool()->pushMatrix();
            float glTransform[16];
            x[i].writeOpenGlMatrix ( glTransform );
            vparams->drawTool()->multMatrix( glTransform );
            vparams->drawTool()->scale ( scale);
            vparams->drawTool()->drawPlus ( 0.1, Vec<4,float> ( 1.0, 1.0, 0.0, 1.0 ) );
            vparams->drawTool()->popMatrix();
        }
        glPopAttrib();
    }
}

//            template<> class MechanicalObjectInternalData<DeformationGradient332dTypes>
//            {
//            public:
//                typedef MechanicalObject<DeformationGradient332dTypes> Main;
//
//                Data<bool> showDefTensors;
//                Data<bool> showDefTensorsValues;
//                Data<double> showDefTensorScale;
//
//                MechanicalObjectInternalData ( Main* mObj = NULL )
//                        : showDefTensors ( mObj->initData ( &showDefTensors, false, "showDefTensors","show computed deformation tensors." ) )
//                        , showDefTensorsValues ( mObj->initData ( &showDefTensorsValues, false, "showDefTensorsValues","Show Deformation Tensors Values." ) )
//                        , showDefTensorScale ( mObj->initData ( &showDefTensorScale, 1.0, "showDefTensorScale","deformation tensor scale." ) )
//                {}
//            };

template <>
void MechanicalObject<DeformationGradient332dTypes >::draw(const core::visual::VisualParams* vparams)
{
    Mat<4,4, GLfloat> modelviewM;
    Vec<3, SReal> sceneMinBBox, sceneMaxBBox;
    sofa::simulation::Node* context;

    context = dynamic_cast<sofa::simulation::Node*> ( this->getContext() );
    glColor3f ( 1.0,1.0,1.0 );
    glDisable ( GL_LIGHTING );
    sofa::simulation::getSimulation()->computeBBox ( ( sofa::simulation::Node* ) context, sceneMinBBox.ptr(), sceneMaxBBox.ptr() );
    float scale = ( sceneMaxBBox - sceneMinBBox ).norm() * showIndicesScale.getValue();

    if ( showIndices.getValue() )
    {

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
        glEnable ( GL_LIGHTING );
        const DeformationGradient332dTypes::VecCoord& x = ( *getX() );
        const float& scale = showObjectScale.getValue();
        for ( int i=0; i<this->getSize(); i++ )
        {
            vparams->drawTool()->pushMatrix();
            float glTransform[16];
            x[i].writeOpenGlMatrix ( glTransform );
            vparams->drawTool()->multMatrix( glTransform );
            vparams->drawTool()->scale ( scale);
            vparams->drawTool()->drawPlus ( 0.1, Vec<4,float> ( 1.0, 1.0, 0.0, 1.0 ) );
            vparams->drawTool()->popMatrix();
        }
        glPopAttrib();
    }
}



SOFA_DECL_CLASS ( FrameMechanicalObject )

using namespace sofa::defaulttype;

int MechanicalObjectClass = core::RegisterObject ( "mechanical state vectors" )
#ifndef SOFA_FLOAT
        .add< MechanicalObject<Affine3dTypes> >()
        .add< MechanicalObject<Quadratic3dTypes> >()
        .add< MechanicalObject<DeformationGradient331dTypes> >()
        .add< MechanicalObject<DeformationGradient332dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<Affine3fTypes> >()
        .add< MechanicalObject<Quadratic3fTypes> >()
        .add< MechanicalObject<DeformationGradient331fTypes> >()
        .add< MechanicalObject<DeformationGradient332fTypes> >()
#endif
        ;



#ifndef SOFA_FLOAT
template class SOFA_FRAME_API MechanicalObject<Affine3dTypes>;
template class SOFA_FRAME_API MechanicalObject<Quadratic3dTypes>;
template class SOFA_FRAME_API MechanicalObject<DeformationGradient331dTypes>;
template class SOFA_FRAME_API MechanicalObject<DeformationGradient332dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API MechanicalObject<Affine3fTypes>;
template class SOFA_FRAME_API MechanicalObject<Quadratic3fTypes>;
template class SOFA_FRAME_API MechanicalObject<DeformationGradient331fTypes>;
template class SOFA_FRAME_API MechanicalObject<DeformationGradient332fTypes>;
#endif
} // namespace behavior

} // namespace core

} // namespace sofa
