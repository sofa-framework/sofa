#ifndef SOFA_COMPONENT_COLLISION_COMPLIANTCONTACT_H
#define SOFA_COMPONENT_COLLISION_COMPLIANTCONTACT_H

#include "BaseContact.h"
#include "../constraint/UnilateralConstraint.h"

#include "../initCompliant.h"

#include "../compliance/UniformCompliance.h"

#include "../mapping/ContactMapping.h" 		// should be normal mapping

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
    typedef typename Inherit::delta_type delta_type;
    typedef typename Inherit::CollisionModel1 CollisionModel1;
    typedef typename Inherit::CollisionModel2 CollisionModel2;
    typedef typename Inherit::Intersection Intersection;


protected:


    CompliantContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : Inherit(model1, model2, intersectionMethod)
    {}

    typename node_type::SPtr create_node()
    {

//        simulation::MechanicalPropagatePositionAndVelocityVisitor bob( sofa::core::MechanicalParams::defaultInstance() );
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

        // every contact points must propagate constraint forces
        for(unsigned i = 0; i < size; ++i)
        {
            this->mstate1->forceMask.insertEntry( this->mappedContacts[i].index1 );
            if( !this->selfCollision ) this->mstate2->forceMask.insertEntry( this->mappedContacts[i].index2 );
        }

        contact_map->init();


        // compliance
        typedef forcefield::UniformCompliance<defaulttype::Vec1Types> compliance_type;
        compliance_type::SPtr compliance = sofa::core::objectmodel::New<compliance_type>( contact_dofs.get() );
        contact_node->addObject( compliance.get() );
        compliance->compliance.setValue( this->compliance_value.getValue() );
        compliance->damping.setValue( this->damping_ratio.getValue() );
        compliance->init();


        // approximate restitution coefficient between the 2 objects as the product of both coefficients
        const SReal restitutionCoefficient = this->restitution_coef.getValue() ? this->restitution_coef.getValue() : this->model1->getContactRestitution(0) * this->model2->getContactRestitution(0);

        // constraint value
        this->addConstraintValue( contact_node.get(), contact_dofs.get(), restitutionCoefficient );

        // projector
        typedef linearsolver::UnilateralConstraint projector_type;
        projector_type::SPtr projector = sofa::core::objectmodel::New<projector_type>();
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

    void update_node(typename node_type::SPtr ) { }



};

void registerContactClasses();

} // namespace collision
} // namespace component
} // namespace sofa

#endif  // SOFA_COMPONENT_COLLISION_COMPLIANTCONTACT_H
