#ifndef SOFA_COMPONENT_COLLISION_PenalityCompliantContact_H
#define SOFA_COMPONENT_COLLISION_PenalityCompliantContact_H

#include "BaseContact.h"

#include "../initCompliant.h"

#include "../compliance/DiagonalCompliance.h"

#include "../mapping/ContactMapping.h"


namespace sofa
{

namespace component
{

namespace collision
{

template <class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types>
class PenalityCompliantContact : public BaseContact<TCollisionModel1, TCollisionModel2, ResponseDataTypes>
{

public:

    SOFA_CLASS(SOFA_TEMPLATE3(PenalityCompliantContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes), SOFA_TEMPLATE3(BaseContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes) );

    typedef BaseContact<TCollisionModel1, TCollisionModel2, ResponseDataTypes> Inherit;
    typedef typename Inherit::node_type node_type;
    typedef typename Inherit::delta_type delta_type;
    typedef typename Inherit::CollisionModel1 CollisionModel1;
    typedef typename Inherit::CollisionModel2 CollisionModel2;
    typedef typename Inherit::Intersection Intersection;


    Data< SReal > stiffness;


protected:

    PenalityCompliantContact()
        : Inherit()
        , stiffness( initData(&stiffness, SReal(10), "stiffness", "Contact Stiffness") )
    {}

    PenalityCompliantContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : Inherit(model1, model2, intersectionMethod)
        , stiffness( initData(&stiffness, SReal(10), "stiffness", "Contact Stiffness") )
    {}



    typename node_type::SPtr create_node()
    {
        const unsigned size = this->mappedContacts.size();

        delta_type delta = this->make_delta();

        typename node_type::SPtr contact_node = node_type::create( this->getName() + " contact frame" );

        delta.node->addChild( contact_node.get() );

        // ensure all graph context parameters (e.g. dt are well copied)
        contact_node->updateSimulationContext();

        // 1d contact dofs
        typedef container::MechanicalObject<defaulttype::Vec1Types> contact_dofs_type;
        typename contact_dofs_type::SPtr contact_dofs = sofa::core::objectmodel::New<contact_dofs_type>();

        contact_dofs->resize( size );
        contact_dofs->setName( this->getName() + " contact dofs" );
        contact_node->addObject( contact_dofs.get() );

        // contact mapping
        typedef mapping::ContactMapping<ResponseDataTypes, defaulttype::Vec1Types> contact_map_type;
        typename contact_map_type::SPtr contact_map = core::objectmodel::New<contact_map_type>();

        contact_map->setModels( delta.dofs.get(), contact_dofs.get() );
        contact_map->setName( this->getName() + " contact mapping" );
        contact_node->addObject( contact_map.get() );

        this->copyNormals( *edit(contact_map->normal) );
        this->copyPenetrations( *edit(contact_map->penetrations) );

        contact_map->init();


        // compliance
        typedef forcefield::DiagonalCompliance<defaulttype::Vec1Types> compliance_type;
        compliance_type::SPtr compliance = sofa::core::objectmodel::New<compliance_type>( contact_dofs.get() );
        contact_node->addObject( compliance.get() );
        edit(compliance->damping)->assign(1, this->damping_ratio.getValue() );
        compliance->isCompliance.setValue( false );

        typename compliance_type::VecDeriv complianceValues( size );
        for( unsigned i = 0 ; i < size ; ++i )
        {
            // no stiffness for non-violated penetration (alarm distance)
            // TODO add a kind of projector to perform an unilateral stiffness (to add stiffness only one way).
            // This would prevent sticky contacts and would allow to add small stiffness in alarm distance to slow-down object trying to go closer.

            if( (*this->contacts)[i].value < 0 )
            {
                complianceValues[i][0] = 1.0/this->stiffness.getValue();

                // only violated penetrations will propagate forces
                this->mstate1->forceMask.insertEntry( this->mappedContacts[i].index1 );
                if( !this->selfCollision ) this->mstate2->forceMask.insertEntry( this->mappedContacts[i].index2 );
            }
            else
            {
                complianceValues[i][0] = std::numeric_limits<int>::max();
            }
        }
        compliance->diagonal.setValue( complianceValues );

        compliance->init();

        return delta.node;
    }



    void update_node(typename node_type::SPtr ) { }



};

} // namespace collision
} // namespace component
} // namespace sofa

#endif  // SOFA_COMPONENT_COLLISION_PenalityCompliantContact_H
