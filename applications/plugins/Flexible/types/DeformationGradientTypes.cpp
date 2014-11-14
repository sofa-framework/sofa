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
#define FLEXIBLE_DeformationGradientTYPES_CPP

#include "../initFlexible.h"
#include "../types/DeformationGradientTypes.h"
#include <sofa/core/ObjectFactory.h>

#include <SofaBaseMechanics/MechanicalObject.inl>

namespace sofa
{
namespace component
{
namespace container
{

// ==========================================================================
// Init Specializations (initialization from GaussPointSampler)
/*
template <>
void MechanicalObject<F331Types>::init()
{
    engine::BaseGaussPointSampler* sampler=NULL;
    this->getContext()->get(sampler,core::objectmodel::BaseContext::Local);
    if(sampler)
    {
        unsigned int nbp=sampler->getNbSamples();
        this->resize(nbp);

        Data<VecCoord>* x_wAData = this->write(VecCoordId::position());
        VecCoord& x_wA = *x_wAData->beginEdit();
        for(unsigned int i=0;i<nbp;i++) DataTypes::set(x_wA[i], sampler->getSample(i)[0], sampler->getSample(i)[1], sampler->getSample(i)[2]);

        VecCoord *x0_edit = x0.beginEdit();
        x0.setValue(x.getValue());
        if (restScale.getValue() != (Real)1) { Real s = (Real)restScale.getValue(); for (unsigned int i=0; i<x0_edit->size(); i++) (*x0_edit)[i] *= s;        }
        x0.endEdit();

        if(this->f_printLog.getValue())  std::cout<<this->getName()<<" : "<< nbp <<" gauss points imported"<<std::endl;
        reinit();
    }
}

template <>
void MechanicalObject<F332Types>::init()
{
    engine::BaseGaussPointSampler* sampler=NULL;
    this->getContext()->get(sampler,core::objectmodel::BaseContext::Local);
    if(sampler)
    {
        unsigned int nbp=sampler->getNbSamples();
        this->resize(nbp);

        Data<VecCoord>* x_wAData = this->write(VecCoordId::position());
        VecCoord& x_wA = *x_wAData->beginEdit();
        for(unsigned int i=0;i<nbp;i++) DataTypes::set(x_wA[i], sampler->getSample(i)[0], sampler->getSample(i)[1], sampler->getSample(i)[2]);

        VecCoord *x0_edit = x0.beginEdit();
        x0.setValue(x.getValue());
        if (restScale.getValue() != (Real)1) { Real s = (Real)restScale.getValue(); for (unsigned int i=0; i<x0_edit->size(); i++) (*x0_edit)[i] *= s;        }
        x0.endEdit();

        if(this->f_printLog.getValue())  std::cout<<this->getName()<<" : "<< nbp <<" gauss points imported"<<std::endl;
        reinit();
    }
}
*/
// ==========================================================================
// Draw Specializations
/*
template <>
void MechanicalObject<F331Types>::draw(const core::visual::VisualParams* vparams)
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
        const F331Types::VecCoord& x = ( *getX() );
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



template <>
void MechanicalObject<F332Types >::draw(const core::visual::VisualParams* vparams)
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
        const F332Types::VecCoord& x = ( *getX() );
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
*/

// ==========================================================================
// Instanciation

SOFA_DECL_CLASS ( DefGradientMechanicalObject )

using namespace sofa::defaulttype;

int DefGradientMechanicalObjectClass = core::RegisterObject ( "mechanical state vectors" )
#ifndef SOFA_FLOAT
        .add< MechanicalObject<F331dTypes> >()
        .add< MechanicalObject<F321dTypes> >()
        .add< MechanicalObject<F311dTypes> >()
        .add< MechanicalObject<F332dTypes> >()
        .add< MechanicalObject<F221dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<F331fTypes> >()
        .add< MechanicalObject<F321fTypes> >()
        .add< MechanicalObject<F311fTypes> >()
        .add< MechanicalObject<F332fTypes> >()
        .add< MechanicalObject<F221fTypes> >()
#endif
		;

#ifndef SOFA_FLOAT
template class SOFA_Flexible_API MechanicalObject<F331dTypes>;
template class SOFA_Flexible_API MechanicalObject<F321dTypes>;
template class SOFA_Flexible_API MechanicalObject<F311dTypes>;
template class SOFA_Flexible_API MechanicalObject<F332dTypes>;
template class SOFA_Flexible_API MechanicalObject<F221dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API MechanicalObject<F331fTypes>;
template class SOFA_Flexible_API MechanicalObject<F321fTypes>;
template class SOFA_Flexible_API MechanicalObject<F311fTypes>;
template class SOFA_Flexible_API MechanicalObject<F332fTypes>;
template class SOFA_Flexible_API MechanicalObject<F221fTypes>;
#endif

} // namespace container
} // namespace component
} // namespace sofa
