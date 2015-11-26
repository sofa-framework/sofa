#ifndef FRICTIONCOMPLIANTCONTACT_H
#define FRICTIONCOMPLIANTCONTACT_H

#include "BaseContact.h"

#include <Compliant/config.h>

#include <Compliant/constraint/CoulombConstraint.h>
#include <Compliant/compliance/UniformCompliance.h>

#include <Compliant/utils/map.h>
#include <Compliant/utils/edit.h>

namespace sofa
{
namespace component
{
namespace collision
{

// TODO we should inherit from more basic classes, eventually
template <class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types>
class FrictionCompliantContact : public BaseCompliantConstraintContact<TCollisionModel1, TCollisionModel2, ResponseDataTypes> {
public:

    SOFA_CLASS(SOFA_TEMPLATE3(FrictionCompliantContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes), SOFA_TEMPLATE3(BaseCompliantConstraintContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes) );

    typedef BaseCompliantConstraintContact<TCollisionModel1, TCollisionModel2, ResponseDataTypes> Inherit;
    typedef typename Inherit::node_type node_type;
    typedef typename Inherit::CollisionModel1 CollisionModel1;
    typedef typename Inherit::CollisionModel2 CollisionModel2;
    typedef typename Inherit::Intersection Intersection;

    Data< SReal > mu; ///< friction coef
    Data< bool > horizontalConeProjection; ///< should the cone projection be horizontal (default)? Otherwise an orthogonal cone projection is performed.

protected:

    typedef defaulttype::Vec3Types contact_type;
    typedef container::MechanicalObject<contact_type> contact_dofs_type;
    typename contact_dofs_type::SPtr contact_dofs;

    core::BaseMapping::SPtr contact_map;

    typedef forcefield::UniformCompliance<contact_type> compliance_type;
    compliance_type::SPtr compliance;

    typedef linearsolver::CoulombConstraint<contact_type> proj_type;
    proj_type::SPtr projector;

//    FrictionCompliantContact()
//        : Inherit()
//        , mu( initData(&mu, SReal(0.0), "mu", "friction coefficient (0 for frictionless contacts)") )
//        , horizontalConeProjection( initData(&horizontalConeProjection, true, "horizontalConeProjection", "Should the Coulomb cone projection be horizontal (default)? Otherwise an orthogonal cone projection is performed.") )
//    {}

    FrictionCompliantContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : Inherit(model1, model2, intersectionMethod)
        , mu( initData(&mu, SReal(0.7), "mu", "friction coefficient (0 for frictionless contacts)") )
        , horizontalConeProjection(initData(&horizontalConeProjection, true, "horizontal", "horizontal cone projection, else orthogonal"))
    {
        
    }


    void create_node()
    {
        const unsigned size = this->mappedContacts.size();

        // node->addChild( delta.node.get() );

        this->contact_node = node_type::create( this->getName() + "_contact_frame" );
        down_cast< node_type >(this->mstate1->getContext())->addChild( this->contact_node.get() );

        // ensure all graph context parameters (e.g. dt are well copied)
        this->contact_node->updateSimulationContext();

        contact_dofs = sofa::core::objectmodel::New<contact_dofs_type>();
        contact_dofs->resize( size );
        this->contact_node->addObject( contact_dofs.get() );

        // mapping
        contact_map = this->template createContactMapping<defaulttype::Vec3Types>(this->contact_node, contact_dofs);


//        // every contact points must propagate constraint forces
//        for(unsigned i = 0; i < size; ++i)
//        {
//            this->mstate1->forceMask.insertEntry( this->mappedContacts[i].index1 );
//            if( !this->selfCollision ) this->mstate2->forceMask.insertEntry( this->mappedContacts[i].index2 );
//        }


        // TODO diagonal compliance, soft  and compliance_value for normal
        compliance = sofa::core::objectmodel::New<compliance_type>( contact_dofs.get() );
//        compliance->_restitution.setValue( restitution_coef.getValue() );
        this->contact_node->addObject( compliance.get() );
        compliance->compliance.setValue( this->compliance_value.getValue() );
        compliance->damping.setValue( this->damping_ratio.getValue() );
        compliance->init();


        // approximate current mu between the 2 objects as the product of both friction coefficients
        const SReal frictionCoefficient = mu.getValue() ? mu.getValue() : this->model1->getContactFriction(0)*this->model2->getContactFriction(0);

        // approximate restitution coefficient between the 2 objects as the product of both coefficients
        const SReal restitutionCoefficient = this->restitution_coef.getValue() ? this->restitution_coef.getValue() : this->model1->getContactRestitution(0) * this->model2->getContactRestitution(0);


        // constraint value
        vector<bool>* cvmask = this->addConstraintValue( this->contact_node.get(), contact_dofs.get(), restitutionCoefficient );

        // projector
        projector = sofa::core::objectmodel::New<proj_type>( frictionCoefficient );
        projector->horizontalProjection = horizontalConeProjection.getValue();
        this->contact_node->addObject( projector.get() );
        // for restitution, only activate violated constraints
        if( restitutionCoefficient ) projector->mask = cvmask;
    }


    void update_node() {
        const unsigned size = this->mappedContacts.size();

        if( this->selfCollision )
        {
            typedef mapping::ContactMapping<ResponseDataTypes, defaulttype::Vec3Types> contact_mapping_type;
            core::objectmodel::SPtr_dynamic_cast<contact_mapping_type>(contact_map)->setDetectionOutput(this->contacts);
        }
        else
        {
            typedef mapping::ContactMultiMapping<ResponseDataTypes, defaulttype::Vec3Types> contact_mapping_type;
            core::objectmodel::SPtr_dynamic_cast<contact_mapping_type>(contact_map)->setDetectionOutput(this->contacts);  
        }

        contact_dofs->resize( size );
        contact_map->reinit();

        if( compliance->compliance.getValue() != this->compliance_value.getValue() ||
                compliance->damping.getValue() != this->damping_ratio.getValue() )
        {
            compliance->compliance.setValue( this->compliance_value.getValue() );
            compliance->damping.setValue( this->damping_ratio.getValue() );
            compliance->reinit();
        }

        // approximate current mu between the 2 objects as the product of both friction coefficients
        const SReal frictionCoefficient = mu.getValue() ? mu.getValue() : this->model1->getContactFriction(0)*this->model2->getContactFriction(0);

        // approximate restitution coefficient between the 2 objects as the product of both coefficients
        const SReal restitutionCoefficient = this->restitution_coef.getValue() ? this->restitution_coef.getValue() : this->model1->getContactRestitution(0) * this->model2->getContactRestitution(0);
        // updating constraint value
        this->contact_node->removeObject( this->baseConstraintValue ) ;
        vector<bool>* cvmask = this->addConstraintValue( this->contact_node.get(), contact_dofs.get(), restitutionCoefficient );

        if( restitutionCoefficient ) projector->mask = cvmask; // for restitution, only activate violated constraints
        else projector->mask = NULL;
        projector->horizontalProjection = horizontalConeProjection.getValue();
        projector->mu = frictionCoefficient;


//        // every contact points must propagate constraint forces
//        for(unsigned i = 0; i < size; ++i)
//        {
//            this->mstate1->forceMask.insertEntry( this->mappedContacts[i].index1 );
//            if( !this->selfCollision ) this->mstate2->forceMask.insertEntry( this->mappedContacts[i].index2 );
//        }

    }



};


} // namespace collision
} // namespace component
} // namespace sofa

#endif // FRICTIONCOMPLIANTCONTACT_H
