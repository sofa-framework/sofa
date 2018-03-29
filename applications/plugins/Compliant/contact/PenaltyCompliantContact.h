#ifndef SOFA_COMPONENT_COLLISION_PenaltyCompliantContact_H
#define SOFA_COMPONENT_COLLISION_PenaltyCompliantContact_H

#include "BaseContact.h"

#include <Compliant/config.h>

#include "../forcefield/CompliantPenaltyForceField.h"

#include "../mapping/ContactMapping.h"

#include "../utils/edit.h"


namespace sofa
{

namespace component
{

namespace collision
{

template <class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types>
class PenaltyCompliantContact : public BaseContact<TCollisionModel1, TCollisionModel2, ResponseDataTypes>
{

public:

    SOFA_CLASS(SOFA_TEMPLATE3(PenaltyCompliantContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes), SOFA_TEMPLATE3(BaseContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes) );

    typedef BaseContact<TCollisionModel1, TCollisionModel2, ResponseDataTypes> Inherit;
    typedef typename Inherit::node_type node_type;
    typedef typename Inherit::CollisionModel1 CollisionModel1;
    typedef typename Inherit::CollisionModel2 CollisionModel2;
    typedef typename Inherit::Intersection Intersection;


    Data< SReal > stiffness; ///< Contact Stiffness


protected:

    typedef container::MechanicalObject<defaulttype::Vec1Types> contact_dofs_type;
    typename contact_dofs_type::SPtr contact_dofs;

    typedef mapping::ContactMapping<ResponseDataTypes, defaulttype::Vec1Types> contact_map_type;
    typename contact_map_type::SPtr contact_map;

    typedef forcefield::CompliantPenaltyForceField<defaulttype::Vec1Types> forcefield_type;
    forcefield_type::SPtr forcefield;

    PenaltyCompliantContact()
        : Inherit()
        , stiffness( initData(&stiffness, SReal(10), "stiffness", "Contact Stiffness") )
    {}

    PenaltyCompliantContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : Inherit(model1, model2, intersectionMethod)
        , stiffness( initData(&stiffness, SReal(10), "stiffness", "Contact Stiffness") )
    {}



    void create_node()
    {
        const unsigned size = this->mappedContacts.size();

        this->make_delta();

        typename node_type::SPtr contact_node = node_type::create( this->getName() + "_contact_frame" );

        this->delta_node->addChild( contact_node.get() );

        // ensure all graph context parameters (e.g. dt are well copied)
        contact_node->updateSimulationContext();

        // 1d contact dofs
        contact_dofs = sofa::core::objectmodel::New<contact_dofs_type>();

        contact_dofs->resize( size );
        contact_dofs->setName( this->getName() + "_contact_dofs" );
        contact_node->addObject( contact_dofs.get() );

        // contact mapping
        contact_map = core::objectmodel::New<contact_map_type>();

        contact_map->setModels( this->delta_dofs.get(), contact_dofs.get() );
        contact_map->setName( this->getName() + "_contact_mapping" );
        contact_node->addObject( contact_map.get() );

        this->copyNormals( *editOnly(contact_map->normal) );
        this->copyPenetrations( *editOnly(*contact_dofs->write(core::VecCoordId::position())) );

        contact_map->init();


        // compliance
        forcefield = sofa::core::objectmodel::New<forcefield_type>( contact_dofs.get() );
        contact_node->addObject( forcefield.get() );
        forcefield->d_damping.setValue( this->damping_ratio.getValue() );
        forcefield->d_stiffness.setValue( this->stiffness.getValue() );

        for( unsigned i = 0 ; i < size ; ++i )
        {
            // no stiffness for non-violated penetration (alarm distance)
            // TODO add a kind of projector to perform an unilateral stiffness (to add stiffness only one way).
            // This would prevent sticky contacts and would allow to add small stiffness in alarm distance to slow-down object trying to go closer.

            if( (*this->contacts)[i].value < 0 )
            {
                // only violated penetrations will propagate forces
                this->mstate1->forceMask.insertEntry( this->mappedContacts[i].index1 );
                if( !this->selfCollision ) this->mstate2->forceMask.insertEntry( this->mappedContacts[i].index2 );
            }
        }

    }



    void update_node() 
    {
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

        forcefield->d_damping.setValue( this->damping_ratio.getValue() );
        forcefield->d_stiffness.setValue( this->stiffness.getValue() );

        // every violated contact points must propagate constraint forces
        for( unsigned i = 0 ; i < size ; ++i )
        {
            if( (*this->contacts)[i].value < 0 )
            {
                // only violated penetrations will propagate forces
                this->mstate1->forceMask.insertEntry( this->mappedContacts[i].index1 );
                if( !this->selfCollision ) this->mstate2->forceMask.insertEntry( this->mappedContacts[i].index2 );
            }
        }
    }



};

} // namespace collision
} // namespace component
} // namespace sofa

#endif  // SOFA_COMPONENT_COLLISION_PenaltyCompliantContact_H
