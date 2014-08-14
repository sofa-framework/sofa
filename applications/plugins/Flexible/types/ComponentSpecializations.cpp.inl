

#include "../initFlexible.h"


#include "ComponentSpecializationsDefines.h"

#ifdef Success
#undef Success // before including eigen stuff http://eigen.tuxfamily.org/bz/show_bug.cgi?id=253
#endif
#include <SofaBoundaryCondition/ProjectToPointConstraint.inl>
#include <SofaBoundaryCondition/ProjectToLineConstraint.inl>
#include <SofaBoundaryCondition/ProjectToPlaneConstraint.inl>
#include <SofaBoundaryCondition/ProjectDirectionConstraint.inl>

#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/common/Node.h>

#include <SofaBaseMechanics/MechanicalObject.inl>

#include <SofaBoundaryCondition/FixedConstraint.inl>
#include <SofaBoundaryCondition/PartialFixedConstraint.inl>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>

#include <SofaEngine/BoxROI.inl>


#include <SofaBaseMechanics/UniformMass.inl>

#include <SofaValidation/Monitor.inl>
#include <SofaValidation/ExtraMonitor.inl>

#include <SofaConstraint/UncoupledConstraintCorrection.inl>

#include <SofaBaseMechanics/IdentityMapping.inl>
#include <SofaMiscMapping/SubsetMultiMapping.inl>

#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/behavior/Mass.inl>
#include <SofaDeformable/RestShapeSpringsForceField.inl>
#include <SofaBoundaryCondition/ConstantForceField.inl>


#ifdef SOFA_HAVE_IMAGE
#include "../mass/ImageDensityMass.inl"
#endif




namespace sofa
{

namespace core
{

namespace behavior
{


#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API ForceField< defaulttype::TYPEABSTRACTNAME3dTypes >;
    template class SOFA_Flexible_API Mass< defaulttype::TYPEABSTRACTNAME3dTypes >;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API ForceField< defaulttype::TYPEABSTRACTNAME3fTypes >;
    template class SOFA_Flexible_API Mass< defaulttype::TYPEABSTRACTNAME3fTypes >;
#endif


} // namespace behavior

} // namespace core



namespace component
{
namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;



// ==========================================================================
// FixedConstraint

#ifndef SOFA_FLOAT
template<>
void FixedConstraint< TYPEABSTRACTNAME3dTypes >::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    const SetIndexArray & indices = f_indices.getValue();
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();

    if( f_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< Vector3 > points;

        if( f_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
                points.push_back(x[i].getCenter());
        else
        {
            if( x.size() < indices.size() ) for (unsigned i=0; i<x.size(); i++ ) points.push_back(x[indices[i]].getCenter());
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it) points.push_back(x[*it].getCenter());
        }

        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    }
    else
//        vparams->drawTool()->drawSpheres(points, (float)f_drawSize.getValue(), Vec<4,float>(0.2f,0.1f,0.9f,1.0f));
    {
        if( f_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[i].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( f_drawSize.getValue() );
                vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                vparams->drawTool()->popMatrix();
            }
        else
        {
            if( x.size() < indices.size() )
                for (unsigned i=0; i<x.size(); i++ )
                {
                    vparams->drawTool()->pushMatrix();
                    float glTransform[16];
                    x[indices[i]].writeOpenGlMatrix ( glTransform );
                    vparams->drawTool()->multMatrix( glTransform );
                    vparams->drawTool()->scale ( f_drawSize.getValue() );
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    vparams->drawTool()->popMatrix();
                }
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[*it].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( f_drawSize.getValue() );
                vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                vparams->drawTool()->popMatrix();
            }
        }
    }
}
#endif
#ifndef SOFA_DOUBLE
template<>
void FixedConstraint< TYPEABSTRACTNAME3fTypes >::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    const SetIndexArray & indices = f_indices.getValue();
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();

    if( f_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< Vector3 > points;

        if( f_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
                points.push_back(x[i].getCenter());
        else
        {
            if( x.size() < indices.size() ) for (unsigned i=0; i<x.size(); i++ ) points.push_back(x[indices[i]].getCenter());
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it) points.push_back(x[*it].getCenter());
        }

        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    }
    else
//        vparams->drawTool()->drawSpheres(points, (float)f_drawSize.getValue(), Vec<4,float>(0.2f,0.1f,0.9f,1.0f));
    {
        if( f_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[i].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( f_drawSize.getValue() );
                vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                vparams->drawTool()->popMatrix();
            }
        else
        {
            if( x.size() < indices.size() )
                for (unsigned i=0; i<x.size(); i++ )
                {
                    vparams->drawTool()->pushMatrix();
                    float glTransform[16];
                    x[indices[i]].writeOpenGlMatrix ( glTransform );
                    vparams->drawTool()->multMatrix( glTransform );
                    vparams->drawTool()->scale ( f_drawSize.getValue() );
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    vparams->drawTool()->popMatrix();
                }
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[*it].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( f_drawSize.getValue() );
                vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                vparams->drawTool()->popMatrix();
            }
        }
    }
}
#endif


SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,FixedConstraint) )
int EVALUATOR(TYPEABSTRACTNAME,FixedConstraintClass) = core::RegisterObject ( "Attach given dofs to their initial positions" )
        #ifndef SOFA_FLOAT
                .add< FixedConstraint<defaulttype::TYPEABSTRACTNAME3dTypes> >()
    #endif
    #ifndef SOFA_DOUBLE
                .add< FixedConstraint<defaulttype::TYPEABSTRACTNAME3fTypes> >()
    #endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_Flexible_API FixedConstraint<TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API FixedConstraint<TYPEABSTRACTNAME3fTypes>;
#endif







// ==========================================================================
// PartialFixedConstraint


#ifndef SOFA_FLOAT
template <>
void PartialFixedConstraint<TYPEABSTRACTNAME3dTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    const SetIndexArray & indices = f_indices.getValue();
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();

    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< Vector3 > points;

        if( f_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
                points.push_back(x[i].getCenter());
        else
        {
            if( x.size() < indices.size() ) for (unsigned i=0; i<x.size(); i++ ) points.push_back(x[indices[i]].getCenter());
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it) points.push_back(x[*it].getCenter());
        }

        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    }
    else
//        vparams->drawTool()->drawSpheres(points, (float)f_drawSize.getValue(), Vec<4,float>(0.2f,0.1f,0.9f,1.0f));
    {
        if( f_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[i].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( _drawSize.getValue() );
                vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                vparams->drawTool()->popMatrix();
            }
        else
        {
            if( x.size() < indices.size() )
                for (unsigned i=0; i<x.size(); i++ )
                {
                    vparams->drawTool()->pushMatrix();
                    float glTransform[16];
                    x[indices[i]].writeOpenGlMatrix ( glTransform );
                    vparams->drawTool()->multMatrix( glTransform );
                    vparams->drawTool()->scale ( _drawSize.getValue() );
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    vparams->drawTool()->popMatrix();
                }
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[*it].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( _drawSize.getValue() );
                vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                vparams->drawTool()->popMatrix();
            }
        }
    }
}
#endif
#ifndef SOFA_DOUBLE
template <>
void PartialFixedConstraint<TYPEABSTRACTNAME3fTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    const SetIndexArray & indices = f_indices.getValue();
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();

    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< Vector3 > points;

        if( f_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
                points.push_back(x[i].getCenter());
        else
        {
            if( x.size() < indices.size() ) for (unsigned i=0; i<x.size(); i++ ) points.push_back(x[indices[i]].getCenter());
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it) points.push_back(x[*it].getCenter());
        }

        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    }
    else
//        vparams->drawTool()->drawSpheres(points, (float)f_drawSize.getValue(), Vec<4,float>(0.2f,0.1f,0.9f,1.0f));
    {
        if( f_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[i].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( _drawSize.getValue() );
                vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                vparams->drawTool()->popMatrix();
            }
        else
        {
            if( x.size() < indices.size() )
                for (unsigned i=0; i<x.size(); i++ )
                {
                    vparams->drawTool()->pushMatrix();
                    float glTransform[16];
                    x[indices[i]].writeOpenGlMatrix ( glTransform );
                    vparams->drawTool()->multMatrix( glTransform );
                    vparams->drawTool()->scale ( _drawSize.getValue() );
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    vparams->drawTool()->popMatrix();
                }
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[*it].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( _drawSize.getValue() );
                vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                vparams->drawTool()->popMatrix();
            }
        }
    }
}
#endif

SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,PartialFixedConstraint) )
int EVALUATOR(TYPEABSTRACTNAME,PartialFixedConstraintClass) = core::RegisterObject ( "Attach given cinematic dofs to their initial positions" )
        #ifndef SOFA_FLOAT
        .add< PartialFixedConstraint<defaulttype::TYPEABSTRACTNAME3dTypes> >()
    #endif
    #ifndef SOFA_DOUBLE
        .add< PartialFixedConstraint<defaulttype::TYPEABSTRACTNAME3fTypes> >()
    #endif
;
#ifndef SOFA_FLOAT
template class SOFA_Flexible_API PartialFixedConstraint<TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API PartialFixedConstraint<TYPEABSTRACTNAME3fTypes>;
#endif




// ==========================================================================
// ProjectToPointConstraint
SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,ProjectToPointConstraint) )
int EVALUATOR(TYPEABSTRACTNAME,ProjectToPointConstraintClass) = core::RegisterObject ( "Project particles to a point" )
#ifndef SOFA_FLOAT
        .add< ProjectToPointConstraint<defaulttype::TYPEABSTRACTNAME3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
.add< ProjectToPointConstraint<defaulttype::TYPEABSTRACTNAME3fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_Flexible_API ProjectToPointConstraint<TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API ProjectToPointConstraint<TYPEABSTRACTNAME3fTypes>;
#endif

// ==========================================================================
// ProjectToLineConstraint
SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,ProjectToLineConstraint) )
int EVALUATOR(TYPEABSTRACTNAME,ProjectToLineConstraintClass) = core::RegisterObject ( "Project particles to a line" )
#ifndef SOFA_FLOAT
.add< ProjectToLineConstraint<defaulttype::TYPEABSTRACTNAME3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
.add< ProjectToLineConstraint<defaulttype::TYPEABSTRACTNAME3fTypes> >()
#endif
;
#ifndef SOFA_FLOAT
template class SOFA_Flexible_API ProjectToLineConstraint<TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API ProjectToLineConstraint<TYPEABSTRACTNAME3fTypes>;
#endif

// ==========================================================================
// ProjectToPlaneConstraint
SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,ProjectToPlaneConstraint) )
int EVALUATOR(TYPEABSTRACTNAME,ProjectToPlaneConstraintClass) = core::RegisterObject ( "Project particles to a plane" )
#ifndef SOFA_FLOAT
.add< ProjectToPlaneConstraint<defaulttype::TYPEABSTRACTNAME3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
.add< ProjectToPlaneConstraint<defaulttype::TYPEABSTRACTNAME3fTypes> >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Flexible_API ProjectToPlaneConstraint<TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API ProjectToPlaneConstraint<TYPEABSTRACTNAME3fTypes>;
#endif

// ==========================================================================
// ProjectDirectionConstraint
SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,ProjectDirectionConstraint) )
int EVALUATOR(TYPEABSTRACTNAME,ProjectDirectionConstraintClass) = core::RegisterObject ( "Project particles to a line" )
#ifndef SOFA_FLOAT
.add< ProjectDirectionConstraint<defaulttype::TYPEABSTRACTNAME3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
.add< ProjectDirectionConstraint<defaulttype::TYPEABSTRACTNAME3fTypes> >()
#endif
    ;
#ifndef SOFA_FLOAT
template class SOFA_Flexible_API ProjectDirectionConstraint<TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API ProjectDirectionConstraint<TYPEABSTRACTNAME3fTypes>;
#endif


} // namespace projectiveconstraintset
} // namespace component
} // namespace sofa



#include <sofa/helper/gl/Axis.h>
namespace sofa
{
namespace component
{
namespace container
{

using defaulttype::Vector3;
using defaulttype::Quat;
using defaulttype::Vec4f;

// ==========================================================================
// Draw Specializations
#ifndef SOFA_FLOAT
template <> SOFA_Flexible_API
void MechanicalObject<defaulttype::TYPEABSTRACTNAME3dTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL

    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    if ( showIndices.getValue() )
    {
        glColor3f ( 1.0,1.0,1.0 );
        glPushAttrib(GL_LIGHTING_BIT);
        glDisable ( GL_LIGHTING );
        float scale = ( vparams->sceneBBox().maxBBox() - vparams->sceneBBox().minBBox() ).norm() * showIndicesScale.getValue();

        defaulttype::Mat<4,4, GLfloat> modelviewM;

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

            defaulttype::Vec3d temp ( getPX ( i ), getPY ( i ), getPZ ( i ) );
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
        glPopAttrib();
    }


    if (showObject.getValue())
    {
        const float& scale = showObjectScale.getValue();
        const defaulttype::TYPEABSTRACTNAME3dTypes::VecCoord& x = ( read(core::ConstVecCoordId::position())->getValue() );
        
        for (int i = 0; i < this->getSize(); ++i)
        {
            vparams->drawTool()->pushMatrix();
            float glTransform[16];
            x[i].writeOpenGlMatrix ( glTransform );
            vparams->drawTool()->multMatrix( glTransform );
            vparams->drawTool()->scale ( scale);

            switch( drawMode.getValue() )
            {
                case 1:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,1,0,1) );
                    break;
                case 2:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(1,0,0,1) );
                    break;
                case 3:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    break;
                default:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ) );
            }

            vparams->drawTool()->popMatrix();
        }
    }
#endif /* SOFA_NO_OPENGL */
}
#endif
#ifndef SOFA_DOUBLE
template <> SOFA_Flexible_API
void MechanicalObject<defaulttype::TYPEABSTRACTNAME3fTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL

    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    if ( showIndices.getValue() )
    {
        glColor3f ( 1.0,1.0,1.0 );
        glPushAttrib(GL_LIGHTING_BIT);
        glDisable ( GL_LIGHTING );
        float scale = ( vparams->sceneBBox().maxBBox() - vparams->sceneBBox().minBBox() ).norm() * showIndicesScale.getValue();

        defaulttype::Mat<4,4, GLfloat> modelviewM;

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

            defaulttype::Vec3d temp ( getPX ( i ), getPY ( i ), getPZ ( i ) );
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
        glPopAttrib();
    }


    if (showObject.getValue())
    {
        const float& scale = showObjectScale.getValue();
        const defaulttype::TYPEABSTRACTNAME3fTypes::VecCoord& x = read(core::ConstVecCoordId::position())->getValue();

        for (int i = 0; i < this->getSize(); ++i)
        {
            vparams->drawTool()->pushMatrix();
            float glTransform[16];
            x[i].writeOpenGlMatrix ( glTransform );
            vparams->drawTool()->multMatrix( glTransform );
            vparams->drawTool()->scale ( scale);

            switch( drawMode.getValue() )
            {
                case 1:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,1,0,1) );
                    break;
                case 2:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(1,0,0,1) );
                    break;
                case 3:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    break;
                default:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ) );
            }

            vparams->drawTool()->popMatrix();
        }
    }
#endif /* SOFA_NO_OPENGL */
}
#endif
// ==========================================================================
// Instanciation

SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,MechanicalObject) )

using namespace sofa::defaulttype;

int EVALUATOR(TYPEABSTRACTNAME,MechanicalObjectClass) = core::RegisterObject ( "mechanical state vectors" )
    #ifndef SOFA_FLOAT
        .add< MechanicalObject<TYPEABSTRACTNAME3dTypes> >()
    #endif
    #ifndef SOFA_DOUBLE
        .add< MechanicalObject<TYPEABSTRACTNAME3fTypes> >()
    #endif
        ;



#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API MechanicalObject<TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API MechanicalObject<TYPEABSTRACTNAME3fTypes>;
#endif



} // namespace container

namespace mass
{

//#ifndef SOFA_FLOAT
//template<> SOFA_Flexible_API
//void UniformMass<TYPEABSTRACTNAME3dTypes, TYPEABSTRACTNAME3dMass>::reinit()
//{
//    if (this->totalMass.getValue()>0 && this->mstate!=NULL)
//    {
//        MassType* m = this->mass.beginEdit();
//        *m = ((Real)this->totalMass.getValue() / this->mstate->getX()->size());
//        this->mass.endEdit();
//    }
//    else
//    {
//        this->totalMass.setValue( this->mstate->getX()->size() * this->mass.getValue().getUniformValue() );
//    }
//}
//#endif
//#ifndef SOFA_DOUBLE
//template<> SOFA_Flexible_API
//void UniformMass<TYPEABSTRACTNAME3fTypes, TYPEABSTRACTNAME3fMass>::reinit()
//{
//    if (this->totalMass.getValue()>0 && this->mstate!=NULL)
//    {
//        MassType* m = this->mass.beginEdit();
//        *m = ((Real)this->totalMass.getValue() / this->mstate->getX()->size());
//        this->mass.endEdit();
//    }
//    else
//    {
//        this->totalMass.setValue( this->mstate->getX()->size() * this->mass.getValue().getUniformValue() );
//    }
//}
//#endif

#ifndef SOFA_FLOAT
template <> SOFA_Flexible_API
void UniformMass<defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dMass>::draw(const core::visual::VisualParams* /*vparams*/)
{
}
template <> SOFA_Flexible_API
double UniformMass<defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dMass>::getPotentialEnergy ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx  ) const
{
    helper::ReadAccessor<DataVecCoord> x = vx;

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    double e = 0;
    const MassType& m = mass.getValue();
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    theGravity[0]=g[0], theGravity[1]=g[1], theGravity[2]=g[2];

    Deriv mg = m * theGravity;

    for ( unsigned int i=ibegin; i<iend; i++ )
    {
        Deriv translation;
        translation[0]=(float)x[i].getCenter()[0],  translation[0]=(float)x[1].getCenter()[1], translation[2]=(float)x[i].getCenter()[2];
        e -= translation * mg;
    }
    return e;
}
#endif
#ifndef SOFA_DOUBLE
template <> SOFA_Flexible_API
void UniformMass<defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3fMass>::draw(const core::visual::VisualParams* /*vparams*/)
{
}
template <> SOFA_Flexible_API
double UniformMass<defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3fMass>::getPotentialEnergy ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx  ) const
{
    helper::ReadAccessor<DataVecCoord> x = vx;

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    double e = 0;
    const MassType& m = mass.getValue();
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    theGravity[0]=g[0], theGravity[1]=g[1], theGravity[2]=g[2];

    Deriv mg = m * theGravity;

    for ( unsigned int i=ibegin; i<iend; i++ )
    {
        Deriv translation;
        translation[0]=(float)x[i].getCenter()[0],  translation[0]=(float)x[1].getCenter()[1], translation[2]=(float)x[i].getCenter()[2];
        e -= translation * mg;
    }
    return e;
}
#endif

    // ==========================================================================
    // Instanciation

    using namespace sofa::defaulttype;

    SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,UniformMass) )

    int EVALUATOR(TYPEABSTRACTNAME,UniformMassClass) = core::RegisterObject ( "Define the same mass for all the particles" )
#ifndef SOFA_FLOAT
    .add< UniformMass<TYPEABSTRACTNAME3dTypes,TYPEABSTRACTNAME3dMass> >()
#endif
#ifndef SOFA_DOUBLE
    .add< UniformMass<TYPEABSTRACTNAME3fTypes,TYPEABSTRACTNAME3fMass> >()
#endif
            ;


#ifdef SOFA_HAVE_IMAGE

    SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,ImageDensityMass) )

    int EVALUATOR(TYPEABSTRACTNAME,ImageDensityMassClass) = core::RegisterObject ( "Define a global mass matrix including non diagonal terms" )
#ifndef SOFA_FLOAT
    .add< ImageDensityMass<TYPEABSTRACTNAME3dTypes,core::behavior::ShapeFunctiond,TYPEABSTRACTNAME3dMass> >()
#endif
#ifndef SOFA_DOUBLE
    .add< ImageDensityMass<TYPEABSTRACTNAME3fTypes,core::behavior::ShapeFunctionf,TYPEABSTRACTNAME3fMass> >()
#endif
            ;


#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API ImageDensityMass<TYPEABSTRACTNAME3dTypes,core::behavior::ShapeFunctiond,TYPEABSTRACTNAME3dMass>;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API ImageDensityMass<TYPEABSTRACTNAME3fTypes,core::behavior::ShapeFunctionf,TYPEABSTRACTNAME3fMass>;
#endif


#endif


#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API UniformMass<TYPEABSTRACTNAME3dTypes,TYPEABSTRACTNAME3dMass>;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API UniformMass<TYPEABSTRACTNAME3fTypes,TYPEABSTRACTNAME3fMass>;
#endif


} // namespace mass

namespace misc
{


SOFA_DECL_CLASS( EVALUATOR(TYPEABSTRACTNAME,Monitor) )
// Register in the Factory
int EVALUATOR(TYPEABSTRACTNAME,MonitorClass) = core::RegisterObject("Monitoring of particles")
#ifndef SOFA_FLOAT
        .add< Monitor<defaulttype::TYPEABSTRACTNAME3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< Monitor<defaulttype::TYPEABSTRACTNAME3fTypes> >()
#endif
    ;

#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API Monitor<defaulttype::TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API Monitor<defaulttype::TYPEABSTRACTNAME3fTypes>;
#endif





SOFA_DECL_CLASS( EVALUATOR(TYPEABSTRACTNAME,ExtraMonitor) )
// Register in the Factory
int EVALUATOR(TYPEABSTRACTNAME,ExtraMonitorClass) = core::RegisterObject("Monitoring of particles")
#ifndef SOFA_FLOAT
    .add< ExtraMonitor<defaulttype::TYPEABSTRACTNAME3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
    .add< ExtraMonitor<defaulttype::TYPEABSTRACTNAME3fTypes> >()
#endif
;



#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API ExtraMonitor<defaulttype::TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API ExtraMonitor<defaulttype::TYPEABSTRACTNAME3fTypes>;
#endif



} // namespace misc

namespace constraintset
{
#ifndef SOFA_FLOAT
template<> SOFA_Flexible_API
void UncoupledConstraintCorrection< defaulttype::TYPEABSTRACTNAME3dTypes >::init()
{
    Inherit::init();

    const double dt = this->getContext()->getDt();

    const double dt2 = dt * dt;

    defaulttype::TYPEABSTRACTNAME3dMass massValue;
    VecReal usedComp;

    sofa::component::mass::UniformMass< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dMass >* uniformMass;

    this->getContext()->get( uniformMass, core::objectmodel::BaseContext::SearchUp );
    if( uniformMass )
    {
        massValue = uniformMass->getMass();

        Real H = dt2 / (Real)massValue;

        //for( int i=0 ; i<12 ; ++i )
            usedComp.push_back( H );
    }
    // todo add ImageDensityMass
    /*else
    {
        for( int i=0 ; i<1 ; ++i )
            usedComp.push_back( defaultCompliance.getValue() );
    }*/

    compliance.setValue(usedComp);
}
#endif
#ifndef SOFA_DOUBLE
template<> SOFA_Flexible_API
void UncoupledConstraintCorrection< defaulttype::TYPEABSTRACTNAME3fTypes >::init()
{
    Inherit::init();

    const double dt = this->getContext()->getDt();

    const double dt2 = dt * dt;

    defaulttype::TYPEABSTRACTNAME3fMass massValue;
    VecReal usedComp;

    sofa::component::mass::UniformMass< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3fMass >* uniformMass;

    this->getContext()->get( uniformMass, core::objectmodel::BaseContext::SearchUp );
    if( uniformMass )
    {
        massValue = uniformMass->getMass();

        Real H = dt2 / (Real)massValue;

        //for( int i=0 ; i<12 ; ++i )
            usedComp.push_back( H );
    }
    // todo add ImageDensityMass
    /*else
    {
        for( int i=0 ; i<1 ; ++i )
            usedComp.push_back( defaultCompliance.getValue() );
    }*/

    compliance.setValue(usedComp);
}
#endif

SOFA_DECL_CLASS( EVALUATOR(TYPEABSTRACTNAME,UncoupledConstraintCorrection) )
// Register in the Factory
int EVALUATOR(TYPEABSTRACTNAME,UncoupledConstraintCorrectionClass) = core::RegisterObject("Component computing contact forces within a simulated body using the compliance method.")
#ifndef SOFA_FLOAT
    .add< UncoupledConstraintCorrection<defaulttype::TYPEABSTRACTNAME3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
    .add< UncoupledConstraintCorrection<defaulttype::TYPEABSTRACTNAME3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API UncoupledConstraintCorrection<defaulttype::TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API UncoupledConstraintCorrection<defaulttype::TYPEABSTRACTNAME3fTypes>;
#endif


} // namespace constraintset

namespace mapping
{

SOFA_DECL_CLASS(EVALUATOR(TYPEABSTRACTNAME,IdentityMapping))

// Register in the Factory
int EVALUATOR(TYPEABSTRACTNAME,IdentityMappingClass) = core::RegisterObject("Special case of mapping where the child points are the same as the parent points")
#ifndef SOFA_FLOAT
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Vec3dTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::ExtVec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::Vec3fTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::ExtVec3fTypes > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::Vec3dTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::ExtVec3dTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Vec3fTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::ExtVec3fTypes > >()
#endif
#endif
        ;




#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Vec3dTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::ExtVec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::Vec3fTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::Vec3dTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::ExtVec3dTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Vec3fTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::ExtVec3fTypes >;
#endif
#endif



///////////////////////////////

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(EVALUATOR(TYPEABSTRACTNAME,SubsetMultiMapping))

int EVALUATOR(TYPEABSTRACTNAME,SubsetMultiMappingClass) = core::RegisterObject("Compute a subset of the input MechanicalObjects according to a dof index list")
#ifndef SOFA_FLOAT
    .add< SubsetMultiMapping< TYPEABSTRACTNAME3dTypes, TYPEABSTRACTNAME3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
    .add< SubsetMultiMapping< TYPEABSTRACTNAME3fTypes, TYPEABSTRACTNAME3fTypes > >()
#endif
        ;

#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API SubsetMultiMapping< TYPEABSTRACTNAME3dTypes, TYPEABSTRACTNAME3dTypes >;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API SubsetMultiMapping< TYPEABSTRACTNAME3fTypes, TYPEABSTRACTNAME3fTypes >;
#endif


} // namespace mapping


namespace engine
{
    SOFA_DECL_CLASS(EVALUATOR(TYPEABSTRACTNAME,BoxROI))

    // Register in the Factory
    int EVALUATOR(TYPEABSTRACTNAME,BoxROIClass) = core::RegisterObject("Find the primitives (vertex/edge/triangle/tetrahedron) inside a given box")
#ifndef SOFA_FLOAT
            .add< BoxROI< defaulttype::TYPEABSTRACTNAME3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
            .add< BoxROI< defaulttype::TYPEABSTRACTNAME3fTypes > >()
#endif
    ;


#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API BoxROI< defaulttype::TYPEABSTRACTNAME3dTypes >;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API BoxROI< defaulttype::TYPEABSTRACTNAME3fTypes >;
#endif

} // namespace engine

namespace forcefield
{

    SOFA_DECL_CLASS(EVALUATOR(TYPEABSTRACTNAME,RestShapeSpringsForceField))

    // Register in the Factory
    int EVALUATOR(TYPEABSTRACTNAME,RestShapeSpringsForceFieldClass) = core::RegisterObject("Spring attached to rest position")
    #ifndef SOFA_FLOAT
            .add< RestShapeSpringsForceField< defaulttype::TYPEABSTRACTNAME3dTypes > >()
    #endif
    #ifndef SOFA_DOUBLE
            .add< RestShapeSpringsForceField< defaulttype::TYPEABSTRACTNAME3fTypes > >()
    #endif
    ;

    #ifndef SOFA_FLOAT
        template class SOFA_Flexible_API RestShapeSpringsForceField< defaulttype::TYPEABSTRACTNAME3dTypes >;
    #endif
    #ifndef SOFA_DOUBLE
        template class SOFA_Flexible_API RestShapeSpringsForceField< defaulttype::TYPEABSTRACTNAME3fTypes >;
    #endif





    SOFA_DECL_CLASS(EVALUATOR(TYPEABSTRACTNAME,ConstantForceField))

    // Register in the Factory
    int EVALUATOR(TYPEABSTRACTNAME,ConstantForceFieldClass) = core::RegisterObject("Constant forces applied to given degrees of freedom")
    #ifndef SOFA_FLOAT
            .add< ConstantForceField< defaulttype::TYPEABSTRACTNAME3dTypes > >()
    #endif
    #ifndef SOFA_DOUBLE
            .add< ConstantForceField< defaulttype::TYPEABSTRACTNAME3fTypes > >()
    #endif
    ;

    #ifndef SOFA_FLOAT
        template class SOFA_Flexible_API ConstantForceField< defaulttype::TYPEABSTRACTNAME3dTypes >;
    #endif
    #ifndef SOFA_DOUBLE
        template class SOFA_Flexible_API ConstantForceField< defaulttype::TYPEABSTRACTNAME3fTypes >;
    #endif

} // namespace forcefield

} // namespace component



} // namespace sofa


#include "ComponentSpecializationsUndef.h"
