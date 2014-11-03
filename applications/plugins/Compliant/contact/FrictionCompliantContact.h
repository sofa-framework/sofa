#ifndef FRICTIONCOMPLIANTCONTACT_H
#define FRICTIONCOMPLIANTCONTACT_H

#include "BaseContact.h"

#include "../initCompliant.h"

#include "../constraint/CoulombConstraint.h"
#include "../mapping/ContactMapping.h"

#include "../compliance/UniformCompliance.h"

#include "utils/map.h"

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
    typedef typename Inherit::delta_type delta_type;
    typedef typename Inherit::CollisionModel1 CollisionModel1;
    typedef typename Inherit::CollisionModel2 CollisionModel2;
    typedef typename Inherit::Intersection Intersection;

    Data< SReal > mu; ///< friction coef
    Data< bool > horizontalConeProjection; ///< should the cone projection be horizontal? By default a regular orthogonal cone projection is performed.

protected:

    FrictionCompliantContact()
        : Inherit()
        , mu( initData(&mu, SReal(0.0), "mu", "friction coefficient (0 for frictionless contacts)") )
        , horizontalConeProjection( initData(&horizontalConeProjection, true, "horizontalConeProjection", "Should the Coulomb cone projection be horizontal? By default a regular orthogonal cone projection is performed.") )
    {}

    FrictionCompliantContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : Inherit(model1, model2, intersectionMethod)
        , mu( initData(&mu, SReal(0.0), "mu", "friction coefficient (0 for frictionless contacts)") )
        , horizontalConeProjection( initData(&horizontalConeProjection, true, "horizontalConeProjection", "Should the Coulomb cone projection be horizontal? By default a regular orthogonal cone projection is performed.") )
    {}


    typename node_type::SPtr create_node()
    {

        const unsigned size = this->mappedContacts.size();

        delta_type delta = this->make_delta();

        // node->addChild( delta.node.get() );

        // TODO maybe remove this mapping level
        typename node_type::SPtr contact_node = node_type::create( this->getName() + " contact frame" );

        delta.node->addChild( contact_node.get() );

        // ensure all graph context parameters (e.g. dt are well copied)
        contact_node->updateSimulationContext();

        typedef defaulttype::Vec3Types contact_type;

        typedef container::MechanicalObject<contact_type> contact_dofs_type;
        typename contact_dofs_type::SPtr contact_dofs = sofa::core::objectmodel::New<contact_dofs_type>();

        contact_dofs->resize( size );
        contact_node->addObject( contact_dofs.get() );

        typedef mapping::ContactMapping<ResponseDataTypes, contact_type> contact_map_type;
        typename contact_map_type::SPtr contact_map = core::objectmodel::New<contact_map_type>();

        contact_map->setModels( delta.dofs.get(), contact_dofs.get() );
        contact_node->addObject( contact_map.get() );

        this->copyNormals( *edit(contact_map->normal) );
        this->copyPenetrations( *edit(contact_map->penetrations) );

        // every contact points must propagate constraint forces
        for(unsigned i = 0; i < size; ++i)
        {
            this->mstate1->forceMask.insertEntry( this->mappedContacts[i].index1 );
            if( !this->selfCollision ) this->mstate2->forceMask.insertEntry( this->mappedContacts[i].index2 );
        }

        contact_map->init();

        // TODO diagonal compliance, soft  and compliance_value for normal
        typedef forcefield::UniformCompliance<contact_type> compliance_type;
        compliance_type::SPtr compliance = sofa::core::objectmodel::New<compliance_type>( contact_dofs.get() );
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
        this->addConstraintValue( contact_node.get(), contact_dofs.get(), restitutionCoefficient, 3 );

        // projector
        typedef linearsolver::CoulombConstraint proj_type;
        proj_type::SPtr projector = sofa::core::objectmodel::New<proj_type>( frictionCoefficient );
        projector->horizontalProjection = horizontalConeProjection.getValue();
        contact_node->addObject( projector.get() );
        if( restitutionCoefficient )
        {
            // for restitution, only activate violated constraints
            // todo, mutualize code with mask in addConstraintValue
            projector->mask.resize( this->mappedContacts.size() );
            for(unsigned i = 0; i < this->mappedContacts.size(); ++i)
                projector->mask[i] = ( (*this->contacts)[i].value <= 0 );
        }

        return delta.node;
    }


    void update_node( typename node_type::SPtr ) {
        // TODO
    }



};


} // namespace collision
} // namespace component
} // namespace sofa

#endif // FRICTIONCOMPLIANTCONTACT_H
