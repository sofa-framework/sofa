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



#include "../mapping/ContactMapping.h"

#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMiscCollision/TetrahedronModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseCollision/OBBModel.h>
#include <SofaBaseCollision/CylinderModel.h>
#include <sofa/core/collision/Contact.h>

#include <Compliant/utils/edit.h>
//#include <sofa/simulation/MechanicalVisitor.h>
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

    Data< SReal > viscousFriction; ///< 0 <= viscousFriction <= 1


protected:

    typename node_type::SPtr contact_node;

    typedef container::MechanicalObject<defaulttype::Vec1Types> contact_dofs_type;
    typename contact_dofs_type::SPtr contact_dofs;

    typedef mapping::ContactMapping<ResponseDataTypes, defaulttype::Vec1Types> contact_map_type;
    typename contact_map_type::SPtr contact_map;

    typedef linearsolver::UnilateralConstraint projector_type;
    projector_type::SPtr projector;

    typedef forcefield::UniformCompliance<defaulttype::Vec1Types> compliance_type;
    compliance_type::SPtr compliance;

    typename node_type::SPtr friction_node;
    typedef container::MechanicalObject<defaulttype::Vec2Types> friction_dofs_type;
    typename friction_dofs_type::SPtr friction_dofs;
    typedef mapping::ContactMapping<ResponseDataTypes, defaulttype::Vec2Types> friction_map_type;
    typename friction_map_type::SPtr friction_map;
    typedef forcefield::UniformVelocityDampingForceField<defaulttype::Vec2Types> damping_type;
    damping_type::SPtr damping;


    CompliantContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : Inherit(model1, model2, intersectionMethod)
        , viscousFriction( initData(&viscousFriction, SReal(0), "viscousFriction", "0 <= viscousFriction <= 1") )
    {}

    void create_node() override
    {

//        simulation::MechanicalPropagateOnlyPositionAndVelocityVisitor bob( sofa::core::MechanicalParams::defaultInstance() );
//        this->mstate1->getContext()->getRootContext()->executeVisitor( &bob );
//        this->mstate2->getContext()->getRootContext()->executeVisitor( &bob );
//        this->mstate1->getContext()->executeVisitor( &bob );
//        this->mstate2->getContext()->executeVisitor( &bob );


//        typedef sofa::core::TMultiVecId<core::V_DERIV,core::V_READ> DestMultiVecId;
//        typedef sofa::core::TVecId<core::V_DERIV,core::V_READ> MyVecId;

//        DestMultiVecId v(core::VecDerivId::velocity());
//        MyVecId vid = v.getId(this->mstate1.get());

//        std::cerr<<SOFA_CLASS_METHOD<<"dof1 "<<this->mstate1->getName()<<"  ";this->mstate1->writeVec(core::VecId::velocity(),std::cerr);std::cerr<<std::endl;

//        MyVecId vid2 = v.getId(this->mstate2.get());
//        std::cerr<<SOFA_CLASS_METHOD<<"dof2 "<<this->mstate2->getName()<<"  ";this->mstate2->writeVec(core::VecId::velocity(),std::cerr);std::cerr<<std::endl;



        const unsigned size = this->mappedContacts.size();

        this->make_delta();

        contact_node = node_type::create( this->getName() + "_contact_frame" );

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

//        // every contact points must propagate constraint forces
//        for(unsigned i = 0; i < size; ++i)
//        {
//            this->mstate1->forceMask.insertEntry( this->mappedContacts[i].index1 );
//            if( !this->selfCollision ) this->mstate2->forceMask.insertEntry( this->mappedContacts[i].index2 );
//        }

        contact_map->init();


        // compliance
        compliance = sofa::core::objectmodel::New<compliance_type>( contact_dofs.get() );
        contact_node->addObject( compliance.get() );
        compliance->compliance.setValue( this->compliance_value.getValue() );
        compliance->damping.setValue( this->damping_ratio.getValue() );
        compliance->init();


        // approximate restitution coefficient between the 2 objects as the product of both coefficients
        const SReal restitutionCoefficient = this->restitution_coef.getValue() ? this->restitution_coef.getValue() : this->model1->getContactRestitution(0) * this->model2->getContactRestitution(0);

        // constraint value + keep an eye on violated contacts
        helper::vector<bool>* cvmask = this->addConstraintValue( contact_node.get(), contact_dofs.get(), restitutionCoefficient );

        // projector
        projector = sofa::core::objectmodel::New<projector_type>();
        contact_node->addObject( projector.get() );
        if( restitutionCoefficient ) projector->mask = cvmask; // for restitution, only activate violated constraints

        // approximate current mu between the 2 objects as the product of both friction coefficients
        const SReal frictionCoefficient = viscousFriction.getValue() ? viscousFriction.getValue() : this->model1->getContactFriction(0)*this->model2->getContactFriction(0);
        if( frictionCoefficient )
        {
//            frictionCoefficient = 1.0 - frictionCoefficient;

            // counting violated contacts to create only these ones
            int nout = !cvmask ? size : std::count( cvmask->begin(), cvmask->end(), true );
            if( nout ) create_friction_node( frictionCoefficient, nout, cvmask );
        }
    }


    // viscous friction
    void create_friction_node( SReal frictionCoefficient, size_t size, helper::vector<bool>* cvmask )
    {
        friction_node = node_type::create( this->getName() + "_contact_tangents" );

        this->delta_node->addChild( friction_node.get() );

        // ensure all graph context parameters (e.g. dt are well copied)
        friction_node->updateSimulationContext();

        // 2d friction dofs
        friction_dofs = sofa::core::objectmodel::New<friction_dofs_type>();
        friction_dofs->resize( size );
        friction_dofs->setName( this->getName() + "_friction_dofs" );
        friction_node->addObject( friction_dofs.get() );

        // mapping
        friction_map = core::objectmodel::New<friction_map_type>();
        friction_map->setModels( this->delta_dofs.get(), friction_dofs.get() );
        friction_map->setName( this->getName() + "_friction_mapping" );
        friction_map->mask = *cvmask; // by pointer copy, the vector was deleted before being used in contact mapping...  // TODO improve this by avoiding a copy
        friction_node->addObject( friction_map.get() );

        this->copyNormals( *editOnly(friction_map->normal) );

//                // every contact points must propagate constraint forces
//                for(unsigned i = 0; i < size; ++i)
//                {
//                    this->mstate1->forceMask.insertEntry( this->mappedContacts[i].index1 );
//                    if( !this->selfCollision ) this->mstate2->forceMask.insertEntry( this->mappedContacts[i].index2 );
//                }

        friction_map->init();

        // cheap forcefield version
        damping = sofa::core::objectmodel::New<damping_type>();
        friction_node->addObject( damping.get() );
        damping->dampingCoefficient.setValue( frictionCoefficient );
        damping->init();
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
            compliance->reinit();
        }

//        // every contact points must propagate constraint forces
//        for(unsigned i = 0; i < size; ++i)
//        {
//            this->mstate1->forceMask.insertEntry( this->mappedContacts[i].index1 );
//            if( !this->selfCollision ) this->mstate2->forceMask.insertEntry( this->mappedContacts[i].index2 );
//        }



        // approximate restitution coefficient between the 2 objects as the product of both coefficients
        const SReal restitutionCoefficient = this->restitution_coef.getValue() ? this->restitution_coef.getValue() : this->model1->getContactRestitution(0) * this->model2->getContactRestitution(0);

        // updating constraint value
        contact_node->removeObject( this->baseConstraintValue ) ;
        helper::vector<bool>* cvmask = this->addConstraintValue( contact_node.get(), contact_dofs.get(), restitutionCoefficient );

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
                    this->copyNormals( *editOnly(friction_map->normal) );
                    friction_map->mask = *cvmask; // by pointer copy, the vector was deleted before being used in contact mapping...  // TODO improve this by avoiding a copy
                    friction_map->update();
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
