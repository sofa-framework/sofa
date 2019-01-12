

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


    template class SOFA_Flexible_API State< defaulttype::TYPEABSTRACTNAME3dTypes >;
    template class SOFA_Flexible_API Mapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Vec3Types >;
    template class SOFA_Flexible_API Mapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Rigid3Types >;
    template class SOFA_Flexible_API MultiMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dTypes >;



namespace behavior
{


    template class SOFA_Flexible_API ForceField< defaulttype::TYPEABSTRACTNAME3dTypes >;
    template class SOFA_Flexible_API Mass< defaulttype::TYPEABSTRACTNAME3dTypes >;
    template class SOFA_Flexible_API ConstraintCorrection< defaulttype::TYPEABSTRACTNAME3dTypes >;
    template class SOFA_Flexible_API ProjectiveConstraintSet< defaulttype::TYPEABSTRACTNAME3dTypes >;



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



int EVALUATOR(TYPEABSTRACTNAME,FixedConstraintClass) = core::RegisterObject ( "Attach given dofs to their initial positions" )
                        .add< FixedConstraint<defaulttype::TYPEABSTRACTNAME3dTypes> >()
    
        ;
template class SOFA_Flexible_API FixedConstraint<TYPEABSTRACTNAME3dTypes>;




// ==========================================================================
// ProjectToPointConstraint
int EVALUATOR(TYPEABSTRACTNAME,ProjectToPointConstraintClass) = core::RegisterObject ( "Project particles to a point" )
        .add< ProjectToPointConstraint<defaulttype::TYPEABSTRACTNAME3dTypes> >()

        ;
template class SOFA_Flexible_API ProjectToPointConstraint<TYPEABSTRACTNAME3dTypes>;

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
 /* SOFA_NO_OPENGL */
}
#endif
// ==========================================================================
// Instanciation

using namespace sofa::defaulttype;

int EVALUATOR(TYPEABSTRACTNAME,MechanicalObjectClass) = core::RegisterObject ( "mechanical state vectors" )
            .add< MechanicalObject<TYPEABSTRACTNAME3dTypes> >()
    
        ;



    template class SOFA_Flexible_API MechanicalObject<TYPEABSTRACTNAME3dTypes>;




} // namespace container

namespace mass
{


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
    const MassType& m = d_vertexMass.getValue();
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


    // ==========================================================================
    // Instanciation

    using namespace sofa::defaulttype;

    int EVALUATOR(TYPEABSTRACTNAME,UniformMassClass) = core::RegisterObject ( "Define the same mass for all the particles" )
    .add< UniformMass<TYPEABSTRACTNAME3dTypes,TYPEABSTRACTNAME3dMass> >()

            ;


#ifdef SOFA_HAVE_IMAGE

    int EVALUATOR(TYPEABSTRACTNAME,ImageDensityMassClass) = core::RegisterObject ( "Define a global mass matrix including non diagonal terms" )
    .add< ImageDensityMass<TYPEABSTRACTNAME3dTypes,core::behavior::ShapeFunction3d,TYPEABSTRACTNAME3dMass> >()

            ;


    template class SOFA_Flexible_API ImageDensityMass<TYPEABSTRACTNAME3dTypes,core::behavior::ShapeFunction3d,TYPEABSTRACTNAME3dMass>;



#endif


    template class SOFA_Flexible_API UniformMass<TYPEABSTRACTNAME3dTypes,TYPEABSTRACTNAME3dMass>;



} // namespace mass

namespace misc
{


// Register in the Factory
int EVALUATOR(TYPEABSTRACTNAME,MonitorClass) = core::RegisterObject("Monitoring of particles")
        .add< Monitor<defaulttype::TYPEABSTRACTNAME3dTypes> >()

    ;

    template class SOFA_Flexible_API Monitor<defaulttype::TYPEABSTRACTNAME3dTypes>;






// Register in the Factory
int EVALUATOR(TYPEABSTRACTNAME,ExtraMonitorClass) = core::RegisterObject("Monitoring of particles")
    .add< ExtraMonitor<defaulttype::TYPEABSTRACTNAME3dTypes> >()

;



    template class SOFA_Flexible_API ExtraMonitor<defaulttype::TYPEABSTRACTNAME3dTypes>;




} // namespace misc

namespace constraintset
{
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


// Register in the Factory
int EVALUATOR(TYPEABSTRACTNAME,UncoupledConstraintCorrectionClass) = core::RegisterObject("Component computing contact forces within a simulated body using the compliance method.")
    .add< UncoupledConstraintCorrection<defaulttype::TYPEABSTRACTNAME3dTypes> >()

        ;

    template class SOFA_Flexible_API UncoupledConstraintCorrection<defaulttype::TYPEABSTRACTNAME3dTypes>;



} // namespace constraintset

namespace mapping
{


// Register in the Factory
int EVALUATOR(TYPEABSTRACTNAME,IdentityMappingClass) = core::RegisterObject("Special case of mapping where the child points are the same as the parent points")
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Vec3dTypes > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::ExtVec3Types > >()
        .add< IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dTypes > >()


        ;




    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::Vec3dTypes >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::ExtVec3Types >;
    template class SOFA_Flexible_API IdentityMapping< defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dTypes >;





///////////////////////////////

using namespace sofa::defaulttype;

int EVALUATOR(TYPEABSTRACTNAME,SubsetMultiMappingClass) = core::RegisterObject("Compute a subset of the input MechanicalObjects according to a dof index list")
    .add< SubsetMultiMapping< TYPEABSTRACTNAME3dTypes, TYPEABSTRACTNAME3dTypes > >()

        ;

    template class SOFA_Flexible_API SubsetMultiMapping< TYPEABSTRACTNAME3dTypes, TYPEABSTRACTNAME3dTypes >;



} // namespace mapping


namespace engine
{
    // Register in the Factory
    int EVALUATOR(TYPEABSTRACTNAME,BoxROIClass) = core::RegisterObject("Find the primitives (vertex/edge/triangle/tetrahedron) inside a given box")
            .add< BoxROI< defaulttype::TYPEABSTRACTNAME3dTypes > >()

    ;


    template class SOFA_Flexible_API boxroi::BoxROI< defaulttype::TYPEABSTRACTNAME3dTypes >;


} // namespace engine

namespace forcefield
{

    // Register in the Factory
    int EVALUATOR(TYPEABSTRACTNAME,RestShapeSpringsForceFieldClass) = core::RegisterObject("Spring attached to rest position")
                .add< RestShapeSpringsForceField< defaulttype::TYPEABSTRACTNAME3dTypes > >()
    
    ;

            template class SOFA_Flexible_API RestShapeSpringsForceField< defaulttype::TYPEABSTRACTNAME3dTypes >;
    





    // Register in the Factory
    int EVALUATOR(TYPEABSTRACTNAME,ConstantForceFieldClass) = core::RegisterObject("Constant forces applied to given degrees of freedom")
                .add< ConstantForceField< defaulttype::TYPEABSTRACTNAME3dTypes > >()
    
    ;

            template class SOFA_Flexible_API ConstantForceField< defaulttype::TYPEABSTRACTNAME3dTypes >;
    

    // Register in the Factory
    int EVALUATOR(TYPEABSTRACTNAME,UniformVelocityDampingForceFieldClass) = core::RegisterObject("Uniform velocity damping")
                .add< UniformVelocityDampingForceField< defaulttype::TYPEABSTRACTNAME3dTypes > >()
    
    ;

        template class SOFA_Flexible_API UniformVelocityDampingForceField<defaulttype::TYPEABSTRACTNAME3dTypes>;
    

} // namespace forcefield

} // namespace component



} // namespace sofa


#include "ComponentSpecializationsUndef.h"
