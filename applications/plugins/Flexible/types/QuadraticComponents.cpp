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
#define FLEXIBLE_QuadraticComponents_CPP

#include "../initFlexible.h"
#include "QuadraticComponents.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/common/Node.h>

#include <sofa/component/container/MechanicalObject.inl>

#include <sofa/component/projectiveconstraintset/FixedConstraint.inl>
#include <sofa/component/projectiveconstraintset/PartialFixedConstraint.inl>
#include <sofa/component/projectiveconstraintset/ProjectToPointConstraint.inl>
#include <sofa/component/projectiveconstraintset/ProjectToLineConstraint.inl>
#include <sofa/component/projectiveconstraintset/ProjectToPlaneConstraint.inl>
#include <sofa/component/projectiveconstraintset/ProjectDirectionConstraint.inl>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>

#include <sofa/component/engine/BoxROI.inl>


#include <sofa/component/mass/UniformMass.inl>

#include <sofa/component/misc/Monitor.inl>
#include <sofa/component/misc/ExtraMonitor.inl>

#include <sofa/component/constraintset/UncoupledConstraintCorrection.inl>

#include <sofa/component/mapping/IdentityMapping.inl>

#include <sofa/core/behavior/ForceField.inl>


namespace sofa
{
namespace component
{
namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;



// ==========================================================================
// FixedConstraint


template <>
void FixedConstraint<Quadratic3Types>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    const SetIndexArray & indices = f_indices.getValue();
    const VecCoord& x = *mstate->getX();

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



SOFA_DECL_CLASS ( QuadraticFixedConstraint )
int QuadraticFixedConstraintClass = core::RegisterObject ( "Attach given dofs to their initial positions" )
        .add< FixedConstraint<defaulttype::Quadratic3Types> >()
        ;
template class SOFA_Flexible_API FixedConstraint<Quadratic3Types>;







// ==========================================================================
// PartialFixedConstraint



template <>
void PartialFixedConstraint<Quadratic3Types>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    const SetIndexArray & indices = f_indices.getValue();
    const VecCoord& x = *mstate->getX();

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


SOFA_DECL_CLASS ( QuadraticPartialFixedConstraint )
int QuadraticPartialFixedConstraintClass = core::RegisterObject ( "Attach given cinematic dofs to their initial positions" )
.add< PartialFixedConstraint<defaulttype::Quadratic3Types> >()
;
template class SOFA_Flexible_API PartialFixedConstraint<Quadratic3Types>;




// ==========================================================================
// ProjectToPointConstraint
SOFA_DECL_CLASS ( QuadraticProjectToPointConstraint )
int QuadraticProjectToPointConstraintClass = core::RegisterObject ( "Project particles to a point" )
.add< ProjectToPointConstraint<defaulttype::Quadratic3Types> >();
template class SOFA_Flexible_API ProjectToPointConstraint<Quadratic3Types>;

// ==========================================================================
// ProjectToLineConstraint
SOFA_DECL_CLASS ( QuadraticProjectToLineConstraint )
int QuadraticProjectToLineConstraintClass = core::RegisterObject ( "Project particles to a line" )
.add< ProjectToLineConstraint<defaulttype::Quadratic3Types> >();
template class SOFA_Flexible_API ProjectToLineConstraint<Quadratic3Types>;

// ==========================================================================
// ProjectToPlaneConstraint
SOFA_DECL_CLASS ( QuadraticProjectToPlaneConstraint )
int QuadraticProjectToPlaneConstraintClass = core::RegisterObject ( "Project particles to a plane" )
.add< ProjectToPlaneConstraint<defaulttype::Quadratic3Types> >();
template class SOFA_Flexible_API ProjectToPlaneConstraint<Quadratic3Types>;

// ==========================================================================
// ProjectDirectionConstraint
SOFA_DECL_CLASS ( QuadraticProjectDirectionConstraint )
int QuadraticProjectDirectionConstraintClass = core::RegisterObject ( "Project particles to a line" )
.add< ProjectDirectionConstraint<defaulttype::Quadratic3Types> >();
template class SOFA_Flexible_API ProjectDirectionConstraint<Quadratic3Types>;


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

// ==========================================================================
// Draw Specializations


template <> SOFA_Flexible_API
void MechanicalObject<Quadratic3Types>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    if ( showIndices.getValue() )
    {
        glColor3f ( 1.0,1.0,1.0 );
        glPushAttrib(GL_LIGHTING_BIT);
        glDisable ( GL_LIGHTING );
        float scale = ( vparams->sceneBBox().maxBBox() - vparams->sceneBBox().minBBox() ).norm() * showIndicesScale.getValue();

        Mat<4,4, GLfloat> modelviewM;

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
        glPopAttrib();
    }


    if (showObject.getValue())
    {
        const float& scale = showObjectScale.getValue();
        const Quadratic3Types::VecCoord& x = ( *getX() );

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
                default:
                    vparams->drawTool()->drawFrame ( Vector3(), Quat(), Vector3 ( 1,1,1 ) );
            }

            vparams->drawTool()->popMatrix();
        }
    }
}




// ==========================================================================
// Instanciation

SOFA_DECL_CLASS ( QuadraticMechanicalObject )

using namespace sofa::defaulttype;

int QuadraticMechanicalObjectClass = core::RegisterObject ( "mechanical state vectors" )
        .add< MechanicalObject<Quadratic3Types> >()
        ;

template class SOFA_Flexible_API MechanicalObject<Quadratic3Types>;


} // namespace container


namespace mass
{

//#ifndef SOFA_FLOAT
//template<> SOFA_Flexible_API
//void UniformMass<Quadratic3dTypes, Quadratic3dMass>::reinit()
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
//void UniformMass<Quadratic3fTypes, Quadratic3fMass>::reinit()
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
void UniformMass<defaulttype::Quadratic3dTypes, defaulttype::Quadratic3dMass>::draw(const core::visual::VisualParams* /*vparams*/)
{
}
template <> SOFA_Flexible_API
double UniformMass<defaulttype::Quadratic3dTypes, defaulttype::Quadratic3dMass>::getPotentialEnergy ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx  ) const
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
void UniformMass<defaulttype::Quadratic3fTypes, defaulttype::Quadratic3fMass>::draw(const core::visual::VisualParams* /*vparams*/)
{
}
template <> SOFA_Flexible_API
double UniformMass<defaulttype::Quadratic3fTypes, defaulttype::Quadratic3fMass>::getPotentialEnergy ( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx  ) const
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

    SOFA_DECL_CLASS ( QuadraticUniformMass )

    using namespace sofa::defaulttype;

    int QuadraticUniformMassClass = core::RegisterObject ( "Define the same mass for all the particles" )
            .add< UniformMass<Quadratic3Types,Quadratic3Mass> >()
            ;

    template class SOFA_Flexible_API UniformMass<Quadratic3Types,Quadratic3Mass>;

} // namespace mass






namespace misc
{


SOFA_DECL_CLASS( QuadraticMonitor )
// Register in the Factory
int QuadraticMonitorClass = core::RegisterObject("Monitoring of particles")
        .add< Monitor<defaulttype::Quadratic3Types> >();
template class SOFA_Flexible_API Monitor<defaulttype::Quadratic3Types>;


SOFA_DECL_CLASS( QuadraticExtraMonitor )
// Register in the Factory
int QuadraticExtraMonitorClass = core::RegisterObject("Monitoring of particles")
        .add< ExtraMonitor<defaulttype::Quadratic3Types> >();
template class SOFA_Flexible_API ExtraMonitor<defaulttype::Quadratic3Types>;



} // namespace misc




namespace constraintset
{


template<> SOFA_Flexible_API
void UncoupledConstraintCorrection< defaulttype::Quadratic3Types >::init()
{
    Inherit::init();

    const double dt = this->getContext()->getDt();

    const double dt2 = dt * dt;

    Quadratic3Mass massValue;
    VecReal usedComp;

    sofa::component::mass::UniformMass< Quadratic3Types, Quadratic3Mass >* uniformMass;

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




SOFA_DECL_CLASS( QuadraticUncoupledConstraintCorrection )
// Register in the Factory
int QuadraticUncoupledConstraintCorrectionClass = core::RegisterObject("Component computing contact forces within a simulated body using the compliance method.")
        .add< UncoupledConstraintCorrection<defaulttype::Quadratic3Types> >();
template class SOFA_Flexible_API UncoupledConstraintCorrection<defaulttype::Quadratic3Types>;


} // namespace constraintset



namespace mapping
{


SOFA_DECL_CLASS(QuadraticIdentityMapping)

// Register in the Factory
int QuadraticIdentityMappingClass = core::RegisterObject("Special case of mapping where the child points are the same as the parent points")
        .add< IdentityMapping< defaulttype::Quadratic3Types, defaulttype::Vec3Types > >()
        .add< IdentityMapping< defaulttype::Quadratic3Types, defaulttype::ExtVec3fTypes > >()
        ;


template class SOFA_Flexible_API IdentityMapping< defaulttype::Quadratic3Types, defaulttype::Vec3Types >;
template class SOFA_Flexible_API IdentityMapping< defaulttype::Quadratic3Types, defaulttype::ExtVec3fTypes >;


} // namespace mapping


namespace engine
{
    SOFA_DECL_CLASS(QuadraticBoxROI)

    // Register in the Factory
    int QuadraticBoxROIClass = core::RegisterObject("Find the primitives (vertex/edge/triangle/tetrahedron) inside a given box")
            .add< BoxROI< defaulttype::Quadratic3Types > >();

    template class SOFA_Flexible_API BoxROI< defaulttype::Quadratic3Types >;

} // namespace engine

} // namespace component

} // namespace sofa
