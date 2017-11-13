

#include <Flexible/config.h>


#include "ComponentSpecializationsDefines.h"

#include <SofaBoundaryCondition/ProjectToPointConstraint.inl>
#include <SofaBoundaryCondition/ProjectToLineConstraint.inl>
#include <SofaBoundaryCondition/ProjectToPlaneConstraint.inl>
#include <SofaBoundaryCondition/ProjectDirectionConstraint.inl>

#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Node.h>

#include <sofa/core/State.inl>
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
#include <sofa/core/behavior/ConstraintCorrection.inl>
#include <SofaDeformable/RestShapeSpringsForceField.inl>
#include <SofaBoundaryCondition/ConstantForceField.inl>
#include <SofaBoundaryCondition/UniformVelocityDampingForceField.inl>


#ifdef SOFA_HAVE_IMAGE
#include "../mass/ImageDensityMass.inl"
#endif


#include <sofa/core/Mapping.inl>
#include <sofa/core/MultiMapping.inl>




namespace sofa
{

namespace core
{


#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API State< defaulttype::TYPEABSTRACTNAME3dTypes >;
    template class SOFA_Flexible_API Mapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Vec3dTypes >;
    template class SOFA_Flexible_API Mapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Rigid3dTypes >;
    template class SOFA_Flexible_API MultiMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dTypes >;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API State< defaulttype::TYPEABSTRACTNAME3fTypes >;
    template class SOFA_Flexible_API Mapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::Vec3fTypes >;
    template class SOFA_Flexible_API Mapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::Rigid3fTypes >;
    template class SOFA_Flexible_API MultiMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3fTypes >;
#endif


namespace behavior
{


#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API ForceField< defaulttype::TYPEABSTRACTNAME3dTypes >;
    template class SOFA_Flexible_API Mass< defaulttype::TYPEABSTRACTNAME3dTypes >;
    template class SOFA_Flexible_API ConstraintCorrection< defaulttype::TYPEABSTRACTNAME3dTypes >;
    template class SOFA_Flexible_API ProjectiveConstraintSet< defaulttype::TYPEABSTRACTNAME3dTypes >;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API ForceField< defaulttype::TYPEABSTRACTNAME3fTypes >;
    template class SOFA_Flexible_API Mass< defaulttype::TYPEABSTRACTNAME3fTypes >;
    template class SOFA_Flexible_API ConstraintCorrection< defaulttype::TYPEABSTRACTNAME3fTypes >;
    template class SOFA_Flexible_API ProjectiveConstraintSet< defaulttype::TYPEABSTRACTNAME3fTypes >;
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

    const SetIndexArray & indices = d_indices.getValue();
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();

    if( d_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< Vector3 > points;

        if( d_fixAll.getValue()==true )
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
//        vparams->drawTool()->drawSpheres(points, (float)d_drawSize.getValue(), Vec<4,float>(0.2f,0.1f,0.9f,1.0f));
    {
        if( d_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[i].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( d_drawSize.getValue() );
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
                    vparams->drawTool()->scale ( d_drawSize.getValue() );
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    vparams->drawTool()->popMatrix();
                }
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[*it].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( d_drawSize.getValue() );
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

    const SetIndexArray & indices = d_indices.getValue();
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();

    if( d_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< Vector3 > points;

        if( d_fixAll.getValue()==true )
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
//        vparams->drawTool()->drawSpheres(points, (float)d_drawSize.getValue(), Vec<4,float>(0.2f,0.1f,0.9f,1.0f));
    {
        if( d_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[i].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( d_drawSize.getValue() );
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
                    vparams->drawTool()->scale ( d_drawSize.getValue() );
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    vparams->drawTool()->popMatrix();
                }
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[*it].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( d_drawSize.getValue() );
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

    const SetIndexArray & indices = d_indices.getValue();
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();

    if( d_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< Vector3 > points;

        if( d_fixAll.getValue()==true )
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
//        vparams->drawTool()->drawSpheres(points, (float)d_drawSize.getValue(), Vec<4,float>(0.2f,0.1f,0.9f,1.0f));
    {
        if( d_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[i].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( d_drawSize.getValue() );
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
                    vparams->drawTool()->scale ( d_drawSize.getValue() );
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    vparams->drawTool()->popMatrix();
                }
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[*it].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( d_drawSize.getValue() );
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

    const SetIndexArray & indices = d_indices.getValue();
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();

    if( d_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< Vector3 > points;

        if( d_fixAll.getValue()==true )
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
//        vparams->drawTool()->drawSpheres(points, (float)d_drawSize.getValue(), Vec<4,float>(0.2f,0.1f,0.9f,1.0f));
    {
        if( d_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[i].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( d_drawSize.getValue() );
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
                    vparams->drawTool()->scale ( d_drawSize.getValue() );
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ), Vec4f(0,0,1,1) );
                    vparams->drawTool()->popMatrix();
                }
            else for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                vparams->drawTool()->pushMatrix();
                float glTransform[16];
                x[*it].writeOpenGlMatrix ( glTransform );
                vparams->drawTool()->multMatrix( glTransform );
                vparams->drawTool()->scale ( d_drawSize.getValue() );
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


// TODO: jacobians need to be adjusted to complex types for following projective constaints

//// ==========================================================================
//// ProjectToLineConstraint
//SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,ProjectToLineConstraint) )
//int EVALUATOR(TYPEABSTRACTNAME,ProjectToLineConstraintClass) = core::RegisterObject ( "Project particles to a line" )
//#ifndef SOFA_FLOAT
//.add< ProjectToLineConstraint<defaulttype::TYPEABSTRACTNAME3dTypes> >()
//#endif
//#ifndef SOFA_DOUBLE
//.add< ProjectToLineConstraint<defaulttype::TYPEABSTRACTNAME3fTypes> >()
//#endif
//;
//#ifndef SOFA_FLOAT
//template class SOFA_Flexible_API ProjectToLineConstraint<TYPEABSTRACTNAME3dTypes>;
//#endif
//#ifndef SOFA_DOUBLE
//template class SOFA_Flexible_API ProjectToLineConstraint<TYPEABSTRACTNAME3fTypes>;
//#endif

//// ==========================================================================
//// ProjectToPlaneConstraint
//SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,ProjectToPlaneConstraint) )
//int EVALUATOR(TYPEABSTRACTNAME,ProjectToPlaneConstraintClass) = core::RegisterObject ( "Project particles to a plane" )
//#ifndef SOFA_FLOAT
//.add< ProjectToPlaneConstraint<defaulttype::TYPEABSTRACTNAME3dTypes> >()
//#endif
//#ifndef SOFA_DOUBLE
//.add< ProjectToPlaneConstraint<defaulttype::TYPEABSTRACTNAME3fTypes> >()
//#endif
//;

//#ifndef SOFA_FLOAT
//template class SOFA_Flexible_API ProjectToPlaneConstraint<TYPEABSTRACTNAME3dTypes>;
//#endif
//#ifndef SOFA_DOUBLE
//template class SOFA_Flexible_API ProjectToPlaneConstraint<TYPEABSTRACTNAME3fTypes>;
//#endif

//// ==========================================================================
//// ProjectDirectionConstraint
//SOFA_DECL_CLASS ( EVALUATOR(TYPEABSTRACTNAME,ProjectDirectionConstraint) )
//int EVALUATOR(TYPEABSTRACTNAME,ProjectDirectionConstraintClass) = core::RegisterObject ( "Project particles to a line" )
//#ifndef SOFA_FLOAT
//.add< ProjectDirectionConstraint<defaulttype::TYPEABSTRACTNAME3dTypes> >()
//#endif
//#ifndef SOFA_DOUBLE
//.add< ProjectDirectionConstraint<defaulttype::TYPEABSTRACTNAME3fTypes> >()
//#endif
//    ;
//#ifndef SOFA_FLOAT
//template class SOFA_Flexible_API ProjectDirectionConstraint<TYPEABSTRACTNAME3dTypes>;
//#endif
//#ifndef SOFA_DOUBLE
//template class SOFA_Flexible_API ProjectDirectionConstraint<TYPEABSTRACTNAME3fTypes>;
//#endif


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
        drawIndices(vparams);
    }


    if (showObject.getValue())
    {
        const float& scale = showObjectScale.getValue();
        const defaulttype::TYPEABSTRACTNAME3dTypes::VecCoord& x = ( read(core::ConstVecCoordId::position())->getValue() );

        for (size_t i = 0; i < this->getSize(); ++i)
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
        drawIndices(vparams);
    }


    if (showObject.getValue())
    {
        const float& scale = showObjectScale.getValue();
        const defaulttype::TYPEABSTRACTNAME3fTypes::VecCoord& x = read(core::ConstVecCoordId::position())->getValue();

        for (size_t i = 0; i < this->getSize(); ++i)
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
//        MassType* m = this->mass.beginWriteOnly();
//        *m = ((Real)this->totalMass.getValue() / this->mstate->getSize());
//        this->mass.endEdit();
//    }
//    else
//    {
//        this->totalMass.setValue( this->mstate->getSize() * this->mass.getValue().getUniformValue() );
//    }
//}
//#endif
//#ifndef SOFA_DOUBLE
//template<> SOFA_Flexible_API
//void UniformMass<TYPEABSTRACTNAME3fTypes, TYPEABSTRACTNAME3fMass>::reinit()
//{
//    if (this->totalMass.getValue()>0 && this->mstate!=NULL)
//    {
//        MassType* m = this->mass.beginWriteOnly();
//        *m = ((Real)this->totalMass.getValue() / this->mstate->getSize());
//        this->mass.endEdit();
//    }
//    else
//    {
//        this->totalMass.setValue( this->mstate->getSize() * this->mass.getValue().getUniformValue() );
//    }
//}
//#endif

#ifndef SOFA_FLOAT
template <> SOFA_Flexible_API
void UniformMass<defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dMass>::constructor_message()
{
    serr << "UniformMass on '" << this->templateName() << "' is for debug purpose only and should NOT be used for simulation" << sendl;
}
template <> SOFA_Flexible_API
void UniformMass<defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dMass>::draw(const core::visual::VisualParams* /*vparams*/)
{
}
template <> SOFA_Flexible_API
SReal UniformMass<defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dMass>::getPotentialEnergy ( const core::MechanicalParams*, const DataVecCoord& vx  ) const
{
    helper::ReadAccessor<DataVecCoord> x = vx;

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if ( d_localRange.getValue() [0] >= 0 )
        ibegin = d_localRange.getValue() [0];

    if ( d_localRange.getValue() [1] >= 0 && ( unsigned int ) d_localRange.getValue() [1]+1 < iend )
        iend = d_localRange.getValue() [1]+1;

    SReal e = 0;
    const MassType& m = d_mass.getValue();
    // gravity
    defaulttype::Vec3d g ( this->getContext()->getGravity() );
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
void UniformMass<defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3fMass>::constructor_message()
{
    serr << "UniformMass on '" << this->templateName() << "' is for debug purpose only and should NOT be used for simulation" << sendl;
}
template <> SOFA_Flexible_API
void UniformMass<defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3fMass>::draw(const core::visual::VisualParams* /*vparams*/)
{
}
template <> SOFA_Flexible_API
SReal UniformMass<defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3fMass>::getPotentialEnergy ( const core::MechanicalParams*, const DataVecCoord& vx  ) const
{
    helper::ReadAccessor<DataVecCoord> x = vx;

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if ( d_localRange.getValue() [0] >= 0 )
        ibegin = d_localRange.getValue() [0];

    if ( d_localRange.getValue() [1] >= 0 && ( unsigned int ) d_localRange.getValue() [1]+1 < iend )
        iend = d_localRange.getValue() [1]+1;

    SReal e = 0;
    const MassType& m = d_mass.getValue();
    // gravity
    defaulttype::Vec3d g ( this->getContext()->getGravity() );
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

    const SReal dt = this->getContext()->getDt();

    const SReal dt2 = dt * dt;

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

    const SReal dt = this->getContext()->getDt();

    const SReal dt2 = dt * dt;

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
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::Vec3fTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::ExtVec3fTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3fTypes > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::Vec3dTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::ExtVec3dTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Vec3fTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::ExtVec3fTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3dTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3fTypes > >()
#endif
#endif
        ;




#ifndef SOFA_FLOAT
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Vec3dTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::ExtVec3dTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dTypes >;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::Vec3fTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::ExtVec3fTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::Vec3dTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::ExtVec3dTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Vec3fTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::ExtVec3fTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3dTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3fTypes >;
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
    template class SOFA_Flexible_API boxroi::BoxROI< defaulttype::TYPEABSTRACTNAME3dTypes >;
#endif
#ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API boxroi::BoxROI< defaulttype::TYPEABSTRACTNAME3fTypes >;
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

    SOFA_DECL_CLASS(EVALUATOR(TYPEABSTRACTNAME,UniformVelocityDampingForceField))

    // Register in the Factory
    int EVALUATOR(TYPEABSTRACTNAME,UniformVelocityDampingForceFieldClass) = core::RegisterObject("Uniform velocity damping")
    #ifndef SOFA_FLOAT
            .add< UniformVelocityDampingForceField< defaulttype::TYPEABSTRACTNAME3dTypes > >()
    #endif
    #ifndef SOFA_DOUBLE
            .add< UniformVelocityDampingForceField< defaulttype::TYPEABSTRACTNAME3fTypes > >()
    #endif
    ;

    #ifndef SOFA_FLOAT
    template class SOFA_Flexible_API UniformVelocityDampingForceField<defaulttype::TYPEABSTRACTNAME3dTypes>;
    #endif
    #ifndef SOFA_DOUBLE
    template class SOFA_Flexible_API UniformVelocityDampingForceField<defaulttype::TYPEABSTRACTNAME3fTypes>;
    #endif

} // namespace forcefield

} // namespace component



} // namespace sofa


#include "ComponentSpecializationsUndef.h"
