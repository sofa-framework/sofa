#ifndef SOFA_COMPONENT_COLLISION_COMPLIANTCONTACT_H
#define SOFA_COMPONENT_COLLISION_COMPLIANTCONTACT_H

#include "BaseContact.h"

#include <Compliant/config.h>


#include "../constraint/UnilateralConstraint.h"
#include "../compliance/UniformCompliance.h"

// for constraint version
//#include "../constraint/DampingValue.h"
//#include "../compliance/DampingCompliance.h"

// for forcefield version
#include <SofaBoundaryCondition/UniformVelocityDampingForceField.h>

#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMiscCollision/TetrahedronModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseCollision/OBBModel.h>
#include <SofaBaseCollision/CylinderModel.h>
#include <sofa/core/collision/Contact.h>

#include <Compliant/utils/edit.h>
//#include <sofa/simulation/common/MechanicalVisitor.h>
//#include <sofa/core/VecId.h>
//#include <sofa/core/MultiVecId.h>

namespace sofa
{

namespace component
{

namespace collision
{

template <class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types>
class CompliantContact : public BaseCompliantConstraintContact<TCollisionModel1, TCollisionModel2, ResponseDataTypes>
{

public:

    SOFA_CLASS(SOFA_TEMPLATE3(CompliantContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes), SOFA_TEMPLATE3(BaseCompliantConstraintContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes) );

    typedef BaseCompliantConstraintContact<TCollisionModel1, TCollisionModel2, ResponseDataTypes> Inherit;
    typedef typename Inherit::node_type node_type;
    typedef typename Inherit::CollisionModel1 CollisionModel1;
    typedef typename Inherit::CollisionModel2 CollisionModel2;
    typedef typename Inherit::Intersection Intersection;

    Data< SReal > viscousFriction;


protected:


    typedef container::MechanicalObject<defaulttype::Vec1Types> contact_dofs_type;
    typename contact_dofs_type::SPtr contact_dofs;
    core::BaseMapping::SPtr contact_map;

    typedef linearsolver::UnilateralConstraint projector_type;
    projector_type::SPtr projector;

    typedef forcefield::UniformCompliance<defaulttype::Vec1Types> compliance_type;
    compliance_type::SPtr compliance;

    typename node_type::SPtr friction_node;
    typedef container::MechanicalObject<defaulttype::Vec2Types> friction_dofs_type;
    typename friction_dofs_type::SPtr friction_dofs;
    core::BaseMapping::SPtr friction_map;
    typedef forcefield::UniformVelocityDampingForceField<defaulttype::Vec2Types> damping_type;
    damping_type::SPtr damping;


    CompliantContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : Inherit(model1, model2, intersectionMethod)
        , viscousFriction( initData(&viscousFriction, SReal(0), "viscousFriction", "0 <= viscousFriction <= 1") )
        , friction_node(NULL)
    {}

    virtual void setActiveContact(bool active)
    {
        this->contact_node->setActive(active);
        if(friction_node)
            friction_node->setActive(active);
    }

    virtual void cleanup() {
        // should be called only when !keep
        
        if( this->contact_node ) {
            this->mapper1.cleanup();
            if (!this->selfCollision) this->mapper2.cleanup();
            this->contact_node->detachFromGraph();
            this->contact_node.reset();
        }

        if(friction_node){
            friction_node->detachFromGraph();
            friction_node.reset();
        }

        this->mappedContacts.clear();
    }

    void create_node()
    {
        const unsigned size = this->mappedContacts.size();

        this->contact_node = node_type::create( this->getName() + "_contact_frame" );
        down_cast< node_type >(this->mstate1->getContext())->addChild( this->contact_node.get() );

        // ensure all graph context parameters (e.g. dt are well copied)
        this->contact_node->updateSimulationContext();


        // 1d contact dofs
        contact_dofs = sofa::core::objectmodel::New<contact_dofs_type>();
        contact_dofs->resize( size );
        contact_dofs->setName( this->getName() + "_contact_dofs" );
        this->contact_node->addObject( contact_dofs.get() );

        // mapping
        contact_map = this->template createContactMapping<defaulttype::Vec1Types>(this->contact_node, contact_dofs);

        // compliance
        compliance = sofa::core::objectmodel::New<compliance_type>( contact_dofs.get() );
        this->contact_node->addObject( compliance.get() );
        compliance->compliance.setValue( this->compliance_value.getValue() );
        compliance->damping.setValue( this->damping_ratio.getValue() );
        compliance->init();

        // approximate restitution coefficient between the 2 objects as the product of both coefficients
        const SReal restitutionCoefficient = this->restitution_coef.getValue() ? this->restitution_coef.getValue() : this->model1->getContactRestitution(0) * this->model2->getContactRestitution(0);

        // constraint value + keep an eye on violated contacts
        vector<bool>* cvmask = this->addConstraintValue( this->contact_node.get(), contact_dofs.get(), restitutionCoefficient );

        // projector
        projector = sofa::core::objectmodel::New<projector_type>();
        this->contact_node->addObject( projector.get() );
        if( restitutionCoefficient ) projector->mask = cvmask; // for restitution, only activate violated constraints

        // approximate current mu between the 2 objects as the product of both friction coefficients
        const SReal frictionCoefficient = viscousFriction.getValue() ? viscousFriction.getValue() : this->model1->getContactFriction(0)*this->model2->getContactFriction(0);
        if( frictionCoefficient )
        {
            //frictionCoefficient = 1.0 - frictionCoefficient;

            // counting violated contacts to create only these ones
            int nout = !cvmask ? size : std::count( cvmask->begin(), cvmask->end(), true );
            if( nout ) create_friction_node( frictionCoefficient, nout, cvmask );
        }
    }


    // viscous friction
    void create_friction_node( SReal frictionCoefficient, size_t size, vector<bool>* cvmask )
    {
        friction_node = node_type::create( this->getName() + "_contact_tangents" );

        // 2d friction dofs
        friction_dofs = sofa::core::objectmodel::New<friction_dofs_type>();
        friction_dofs->resize( size );
        friction_dofs->setName( this->getName() + "_friction_dofs" );
        friction_node->addObject( friction_dofs.get() );
        down_cast< node_type >(this->mstate1->getContext())->addChild( this->friction_node.get() );

        // mapping
        friction_map = this->template createContactMapping<defaulttype::Vec2Types>(this->friction_node, friction_dofs, cvmask);       

        // ensure all graph context parameters (e.g. dt are well copied)
        friction_node->updateSimulationContext();
        friction_map->init();

        // cheap forcefield version
        damping = sofa::core::objectmodel::New<damping_type>();
        friction_node->addObject( damping.get() );
        damping->dampingCoefficient.setValue( frictionCoefficient );
        damping->init();
    }


    void update_node() {

        const unsigned size = this->mappedContacts.size();
        contact_dofs->resize( size );
        if( this->selfCollision )
        {
            typedef sofa::component::mapping::ContactMapping<ResponseDataTypes, defaulttype::Vec1Types> contact_mapping_type;
            core::objectmodel::SPtr_dynamic_cast<contact_mapping_type>(contact_map)->setDetectionOutput(this->contacts);
        }
        else
        {
            typedef sofa::component::mapping::ContactMultiMapping<ResponseDataTypes, defaulttype::Vec1Types> contact_mapping_type;
            core::objectmodel::SPtr_dynamic_cast<contact_mapping_type>(contact_map)->setDetectionOutput(this->contacts);  
        }
        contact_map->reinit();

        if( compliance->compliance.getValue() != this->compliance_value.getValue() ||
                compliance->damping.getValue() != this->damping_ratio.getValue() )
        {
            compliance->compliance.setValue( this->compliance_value.getValue() );
            compliance->damping.setValue( this->damping_ratio.getValue() );
            compliance->reinit();
        }


        // approximate restitution coefficient between the 2 objects as the product of both coefficients
        const SReal restitutionCoefficient = this->restitution_coef.getValue() ? this->restitution_coef.getValue() : this->model1->getContactRestitution(0) * this->model2->getContactRestitution(0);

        // updating constraint value
        this->contact_node->removeObject( this->baseConstraintValue ) ;
        vector<bool>* cvmask = this->addConstraintValue( this->contact_node.get(), contact_dofs.get(), restitutionCoefficient );

        if( restitutionCoefficient ) projector->mask = cvmask; // for restitution, only activate violated constraints
        else projector->mask = NULL;


        // updating viscous friction
        const SReal frictionCoefficient = viscousFriction.getValue() ? viscousFriction.getValue() : this->model1->getContactFriction(0)*this->model2->getContactFriction(0);
        if( frictionCoefficient )
        {
            int nout = !cvmask ? size : std::count( cvmask->begin(), cvmask->end(), true );
            if( nout )
            {
                if( !friction_node )
                    create_friction_node( frictionCoefficient, nout, cvmask );
                else
                {
                    damping->dampingCoefficient.setValue( frictionCoefficient );
                    friction_dofs->resize( nout );
                    if( this->selfCollision )
                    {
                        typedef mapping::ContactMapping<ResponseDataTypes, defaulttype::Vec2Types> friction_mapping_type;
                        typename friction_mapping_type::SPtr mapping = core::objectmodel::SPtr_dynamic_cast<friction_mapping_type>(friction_map);
                        mapping->setDetectionOutput(this->contacts);
                        mapping->mask = *cvmask;
                    }
                    else
                    {
                        typedef mapping::ContactMultiMapping<ResponseDataTypes, defaulttype::Vec2Types> friction_mapping_type;
                        typename friction_mapping_type::SPtr mapping = core::objectmodel::SPtr_dynamic_cast<friction_mapping_type>(friction_map);
                        mapping->setDetectionOutput(this->contacts);
                        mapping->mask = *cvmask;
                    }
                    friction_map->reinit();
                    friction_node->setActive(true);
                }
            }
            else if( friction_node )
                friction_node->setActive(false);
        }
        else if( friction_node )
            friction_node->setActive(false);
    }
};

void registerContactClasses();

} // namespace collision
} // namespace component
} // namespace sofa

#endif  // SOFA_COMPONENT_COLLISION_COMPLIANTCONTACT_H
