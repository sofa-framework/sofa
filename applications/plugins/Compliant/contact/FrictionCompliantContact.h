#ifndef FRICTIONCOMPLIANTCONTACT_H
#define FRICTIONCOMPLIANTCONTACT_H

#include "BaseContact.h"

#include <Compliant/config.h>

#include <Compliant/constraint/CoulombConstraint.h>
#include <Compliant/mapping/ContactMapping.h>
#include <Compliant/compliance/UniformCompliance.h>

#include <sofa/helper/OptionsGroup.h>

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
    Data< helper::OptionsGroup > frictionCoefficientMixingMethod; ///< how to blend the friction coefficients from two collision models?

protected:

    typename node_type::SPtr contact_node;

    typedef defaulttype::Vec3Types contact_type;
    typedef container::MechanicalObject<contact_type> contact_dofs_type;
    typename contact_dofs_type::SPtr contact_dofs;

    typedef mapping::ContactMapping<ResponseDataTypes, contact_type> contact_map_type;
    typename contact_map_type::SPtr contact_map;

    typedef forcefield::UniformCompliance<contact_type> compliance_type;
    compliance_type::SPtr compliance;

    typedef linearsolver::CoulombConstraint<contact_type> proj_type;
    proj_type::SPtr projector;


    FrictionCompliantContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : Inherit(model1, model2, intersectionMethod)
        , mu( initData(&mu, SReal(-1), "mu", "global friction coefficient. Warning it overrides everything. (0 for frictionless contacts, <0 -default- to blend coefficients given in collision models)") )
        , horizontalConeProjection(initData(&horizontalConeProjection, true, "horizontal", "horizontal cone projection, else orthogonal"))
        , frictionCoefficientMixingMethod(initData(&frictionCoefficientMixingMethod,"frictionCoefficientMixingMethod","how to blend the friction coefficients from two collision models?" ) )
    {
        helper::OptionsGroup frictionCoefficientMixingMethodOptions(4
                                            ,"MINIMUM" // min(c0,c1)
                                            ,"AVERAGE" // 0.5*(c0+c1)
                                            ,"GEOMETRIC" // sqrt(c0+c1)
                                            ,"PRODUCT" // c0*c1
                                            );
        frictionCoefficientMixingMethodOptions.setSelectedItem(3);
        frictionCoefficientMixingMethod.setValue(frictionCoefficientMixingMethodOptions);
    }


    void create_node() override
    {
        const unsigned size = this->mappedContacts.size();

        this->make_delta();

        // node->addChild( delta.node.get() );

        // TODO maybe remove this mapping level
        contact_node = node_type::create( this->getName() + "_contact_frame" );

        this->delta_node->addChild( contact_node.get() );

        // ensure all graph context parameters (e.g. dt are well copied)
        contact_node->updateSimulationContext();


        contact_dofs = sofa::core::objectmodel::New<contact_dofs_type>();

        contact_dofs->resize( size );
        contact_node->addObject( contact_dofs.get() );

        contact_map = core::objectmodel::New<contact_map_type>();

        contact_map->setModels( this->delta_dofs.get(), contact_dofs.get() );
        contact_node->addObject( contact_map.get() );

        this->copyNormals( *editOnly(contact_map->normal) );
        this->copyPenetrations( *editOnly(*contact_dofs->write(core::VecCoordId::position())) );

//        // every contact points must propagate constraint forces
//        for(unsigned i = 0; i < size; ++i)
//        {
//            this->mstate1->forceMask.insertEntry( this->mappedContacts[i].index1 );
//            if( !this->selfCollision ) this->mstate2->forceMask.insertEntry( this->mappedContacts[i].index2 );
//        }

        contact_map->init();

        // TODO diagonal compliance, soft  and compliance_value for normal
        compliance = sofa::core::objectmodel::New<compliance_type>( contact_dofs.get() );
//        compliance->_restitution.setValue( restitution_coef.getValue() );
        contact_node->addObject( compliance.get() );
        compliance->compliance.setValue( this->compliance_value.getValue() );
        compliance->damping.setValue( this->damping_ratio.getValue() );
        compliance->init();


        // approximate current mu between the 2 objects as the product of both friction coefficients
        const SReal frictionCoefficient = mu.getValue() ? mu.getValue() : this->model1->getContactFriction(0)*this->model2->getContactFriction(0);

        // approximate restitution coefficient between the 2 objects as the product of both coefficients
        const SReal restitutionCoefficient = this->restitution_coef.getValue() ? this->restitution_coef.getValue() : this->model1->getContactRestitution(0) * this->model2->getContactRestitution(0);


        // constraint value
        helper::vector<bool>* cvmask = this->addConstraintValue( contact_node.get(), contact_dofs.get(), restitutionCoefficient );

        // projector
        projector = sofa::core::objectmodel::New<proj_type>( frictionCoefficient );
        projector->horizontalProjection = horizontalConeProjection.getValue();
        contact_node->addObject( projector.get() );
        // for restitution, only activate violated constraints
        if( restitutionCoefficient ) projector->mask = cvmask;
    }


    void update_node() override {
        const unsigned size = this->mappedContacts.size();

        if( this->selfCollision )
        {
            this->copyPairs( *this->deltaContactMap->pairs.beginEdit() );
            this->deltaContactMap->pairs.endEdit();
            this->deltaContactMap->update();
        }
        else
        {
            this->copyPairs( *this->deltaContactMultiMap->pairs.beginEdit() );
            this->deltaContactMultiMap->pairs.endEdit();
            this->deltaContactMultiMap->update();
        }

        contact_dofs->resize( size );

        this->copyNormals( *editOnly(contact_map->normal) );
        this->copyPenetrations( *editOnly(*contact_dofs->write(core::VecCoordId::position())) );
        contact_map->update();

        if( compliance->compliance.getValue() != this->compliance_value.getValue() ||
                compliance->damping.getValue() != this->damping_ratio.getValue() )
        {
            compliance->compliance.setValue( this->compliance_value.getValue() );
            compliance->damping.setValue( this->damping_ratio.getValue() );
        }

        // don't forget reinit as dofs size may have changed !
        compliance->reinit();


        // approximate restitution coefficient between the 2 objects as the product of both coefficients
        const SReal restitutionCoefficient = this->restitution_coef.getValue() ? this->restitution_coef.getValue() : this->model1->getContactRestitution(0) * this->model2->getContactRestitution(0);
        // updating constraint value
        contact_node->removeObject( this->baseConstraintValue ) ;
        helper::vector<bool>* cvmask = this->addConstraintValue( contact_node.get(), contact_dofs.get(), restitutionCoefficient );

        if( restitutionCoefficient ) projector->mask = cvmask; // for restitution, only activate violated constraints
        else projector->mask = NULL;
        projector->horizontalProjection = horizontalConeProjection.getValue();


        // approximate current mu between the 2 objects from both friction coefficients
        const SReal& frictionCoef = mu.getValue();
        if( frictionCoef >=0 ) // if a global mu is defined, use it
        {
            projector->mu = frictionCoef;
        }
        else // blending friction coefficients given in collision models
        {
            switch(frictionCoefficientMixingMethod.getValue().getSelectedId())
            {
            case 0: // MINIMUM
                projector->mu = std::min( this->model1->getContactFriction(0), this->model2->getContactFriction(0) );
                break;
            case 1: // AVERAGE
                projector->mu = 0.5 * ( this->model1->getContactFriction(0) + this->model2->getContactFriction(0) );
                break;
            case 2: // GEOMETRIC
                projector->mu = std::sqrt( this->model1->getContactFriction(0)+this->model2->getContactFriction(0) );
                break;
            case 3: // PRODUCT
                projector->mu = this->model1->getContactFriction(0) * this->model2->getContactFriction(0);
                break;
            }
        }


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
