#ifndef BASECONTACT_H
#define BASECONTACT_H

#include <SofaBaseCollision/DefaultContactManager.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <SofaBaseCollision/BaseContactMapper.h>

#include <Compliant/config.h>

#include "../mapping/DifferenceMapping.h"

#include "../constraint/Restitution.h"
#include "../constraint/HolonomicConstraintValue.h"

//#include <sofa/simulation/DeactivatedNodeVisitor.h>

#include <sofa/helper/cast.h>


namespace sofa
{
namespace component
{
namespace collision
{

/// This is essentially a helper class to factor out all the painful
/// logic of contact classes in order to leave only interesting methods
/// for derived classes to implement.
template <class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types>
class BaseContact : public core::collision::Contact {
public:


    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE3(BaseContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes), core::collision::Contact );



    typedef core::collision::Contact Inherit;
    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::collision::Intersection Intersection;
    typedef ResponseDataTypes DataTypes1;
    typedef ResponseDataTypes DataTypes2;
    typedef core::behavior::MechanicalState<DataTypes1> MechanicalState1;
    typedef core::behavior::MechanicalState<DataTypes2> MechanicalState2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
    typedef core::collision::TDetectionOutputVector<CollisionModel1,CollisionModel2> DetectionOutputVector;



    Data< SReal > damping_ratio; ///< contact damping (used for stabilization)
    Data<bool> holonomic; ///< only enforce null relative velocity, do not try to remove penetration during the dynamics pass
    Data<bool> keep; ///< always keep contact nodes (deactivated when not colliding

protected:

    CollisionModel1* model1;
    CollisionModel2* model2;
    Intersection* intersectionMethod;
    bool selfCollision; ///< true if model1==model2 (in this case, only mapper1 is used)
    ContactMapper<CollisionModel1,DataTypes1> mapper1;
    ContactMapper<CollisionModel2,DataTypes2> mapper2;

    core::objectmodel::BaseContext* parent;


    DetectionOutputVector* contacts;



    /// At each contact, 2 contact points are mapped (stored in mstate1/mstate2)
    struct MappedContact
    {
        int index1; ///< index of the contact points in mstate1
        int index2; ///< index of the contact points in mstate2
//        SReal distance; ///< goal distance between the 2 mapped points // TODO: no longer necessary for CompliantContact & FrictionCompliantContact, do we still want to store this?
    };

    std::vector< MappedContact > mappedContacts;




    BaseContact()
        : damping_ratio( this->initData(&damping_ratio, SReal(0.0), "damping", "contact damping (used for stabilization)") )
        , holonomic(this->initData(&holonomic, false, "holonomic", "only enforce null relative velocity, do not try to remove penetration during the dynamics pass"))
        , keep(this->initData(&keep, false, "keepAlive", "always keep contact nodes (deactivated when not colliding"))
    { }

    BaseContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : damping_ratio( this->initData(&damping_ratio, SReal(0.0), "damping", "contact damping (used for stabilization)") )
        , holonomic(this->initData(&holonomic, false, "holonomic", "only enforce null relative velocity, do not try to remove penetration during the dynamics pass"))
        , keep(this->initData(&keep, false, "keepAlive", "always keep contact nodes (deactivated when not colliding"))
        , model1(model1)
        , model2(model2)
        , intersectionMethod(intersectionMethod)
        , parent(NULL)
    {
        selfCollision = ((core::CollisionModel*)model1 == (core::CollisionModel*)model2);
        mapper1.setCollisionModel(model1);
        if (!selfCollision) mapper2.setCollisionModel(model2);
        mappedContacts.clear();
    }

    virtual ~BaseContact() {}



public:


    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }

    void setDetectionOutputs(core::collision::DetectionOutputVector* o)
    {
        contacts = static_cast<DetectionOutputVector*>(o);

        // POSSIBILITY to have a filter on detected contacts, like removing duplicated/too close contacts
    }



    void createResponse(core::objectmodel::BaseContext* /*group*/ )
    {
        if( !contacts )
        {
            // should only be called when keepAlive
            delta_node->setActive( false );
//            simulation::DeactivationVisitor v(sofa::core::ExecParams::defaultInstance(), false);
//            node->executeVisitor(&v);
            return; // keeping contact alive imposes a call with a null DetectionOutput
        }

        if( !delta_node )
        {
            //fancy names
            std::string name1 = this->model1->getClassName() + "_contact_points";
            std::string name2 = this->model2->getClassName() + "_contact_points";

            // obtain point mechanical models from mappers
            mstate1 = mapper1.createMapping( name1.c_str() );

            mstate2 = this->selfCollision ? mstate1 : mapper2.createMapping( name2.c_str() );
            mstate2->setName("dofs");
        }
       

        // resize mappers
        unsigned size = contacts->size();

        if ( this->selfCollision )
        {
            mapper1.resize( 2 * size );
        }
        else
        {
            mapper1.resize( size );
            mapper2.resize( size );
        }

        // setup mappers / mappedContacts

        // desired proximity contact distance
//        const double d0 = this->intersectionMethod->getContactDistance() + this->model1->getProximity() + this->model2->getProximity();

        mappedContacts.resize( size );

        for( unsigned i=0 ; i<contacts->size() ; ++i )
        {
            const core::collision::DetectionOutput& o = (*contacts)[i];

            CollisionElement1 elem1(o.elem.first);
            CollisionElement2 elem2(o.elem.second);

            int index1 = elem1.getIndex();
            int index2 = elem2.getIndex();

            // distance between the actual used dof and the geometrical contact point (must be initialized to 0, because if they are confounded, addPoint won't necessarily write the 0...)
            typename DataTypes1::Real r1 = 0.0;
            typename DataTypes2::Real r2 = 0.0;

            // Create mapping for first point
            mappedContacts[i].index1 = mapper1.addPoint(o.point[0], index1, r1);

            // TODO local contact coords (o.baryCoords) are broken !!

            // max: this one is broken :-/
            // mappedContacts[i].index1 = mapper1.addPointB(o.point[0], index1, r1, o.baryCoords[0]);

            // Create mapping for second point
            mappedContacts[i].index2 = this->selfCollision ?
                mapper1.addPoint(o.point[1], index2, r2):
                mapper2.addPoint(o.point[1], index2, r2);

                // max: same here :-/
                        // mapper1.addPointB(o.point[1], index2, r2, o.baryCoords[1]) :
                        // mapper2.addPointB(o.point[1], index2, r2, o.baryCoords[1]);

//            mappedContacts[i].distance = d0 + r1 + r2;

        }

        // poke mappings
        mapper1.update();
        if (!this->selfCollision) mapper2.update();

        if(!delta_node) {
            create_node();
        } else {
            delta_node->setActive( true );
//            simulation::DeactivationVisitor v(sofa::core::ExecParams::defaultInstance(), true);
//            node->executeVisitor(&v);
         	update_node();
        }
    }


    /// @internal for SOFA collision mechanism
    /// called before setting-up new collisions
    void removeResponse() {
        if( delta_node ) {
            mapper1.resize(0);
            mapper2.resize(0);
        }
    }

    /// @internal for SOFA collision mechanism
    /// called when the collision components must be removed from the scene graph
    void cleanup() {

        // should be called only when !keep

        if( delta_node ) {
            mapper1.cleanup();
            if (!this->selfCollision) mapper2.cleanup();
            delta_node->detachFromGraph();
            delta_node.reset();
        }

        mappedContacts.clear();
    }


    /// @internal for SOFA collision mechanism
    /// to check if the collision components must be removed from the scene graph
    /// or if it should be kept but deactivated
    /// when the objects are no longer colliding
    virtual bool keepAlive() { return keep.getValue(); }


protected:

    typedef SReal real;

    // the node that will hold all the stuff
    typedef sofa::simulation::Node node_type;
    node_type::SPtr delta_node;

    // TODO correct real type
    typedef container::MechanicalObject<ResponseDataTypes> delta_dofs_type;
    typename delta_dofs_type::SPtr delta_dofs;


    // the difference mapping used in the delta node (needed by update_node)
    typedef mapping::DifferenceMapping<ResponseDataTypes, ResponseDataTypes> delta_map_type;
    typename delta_map_type::SPtr deltaContactMap;
    typedef mapping::DifferenceMultiMapping<ResponseDataTypes, ResponseDataTypes> delta_multimap_type;
    typename delta_multimap_type::SPtr deltaContactMultiMap;

    typename MechanicalState1::SPtr mstate1;
    typename MechanicalState2::SPtr mstate2;

    typename odesolver::BaseConstraintValue::SPtr baseConstraintValue;

    // all you're left to implement \o/
    virtual void create_node() = 0;
    virtual void update_node() = 0;



    /// @internal
    /// create main node / dof for Compliant Contact
    /// as a differencemapping
    void make_delta() {

        delta_node = node_type::create( this->getName() + "_delta" );

        // TODO vec3types according to the types in interaction !

        // delta dofs
        const size_t size = mappedContacts.size();
        assert( size );

        delta_dofs = core::objectmodel::New<delta_dofs_type>();
        delta_dofs->setName( this->model2->getName() + "_-_" + this->model1->getName()  );
        delta_dofs->resize( size );
        
//        delta_dofs->showObject.setValue(true);
//        delta_dofs->showObjectScale.setValue(10);

        delta_node->addObject( delta_dofs );


        // delta mappings
        if( this->selfCollision ) {
            deltaContactMap = sofa::core::objectmodel::New<delta_map_type>();

            deltaContactMap->setModels( this->mstate1.get(), delta_dofs.get() );

            copyPairs( *deltaContactMap->pairs.beginEdit() );
            deltaContactMap->pairs.endEdit();

            deltaContactMap->setName( "delta_mapping" );

            delta_node->addObject( deltaContactMap.get() );

            down_cast< node_type >(this->mstate1->getContext())->addChild( delta_node.get() );

            deltaContactMap->init();

        } else {
            deltaContactMultiMap = core::objectmodel::New<delta_multimap_type>();

            deltaContactMultiMap->addInputModel( this->mstate1.get() );
            deltaContactMultiMap->addInputModel( this->mstate2.get() );

            deltaContactMultiMap->addOutputModel( delta_dofs.get() );

            copyPairs( *deltaContactMultiMap->pairs.beginEdit() );
            deltaContactMultiMap->pairs.endEdit();

            deltaContactMultiMap->setName( "delta_mapping" );

            delta_node->addObject( deltaContactMultiMap.get() );

            // the scene graph should reflect mapping dependencies
            down_cast< node_type >(this->mstate1->getContext())->addChild( delta_node.get() );
            down_cast< node_type >(this->mstate2->getContext())->addChild( delta_node.get() );

            deltaContactMultiMap->init();
        }

        // ensure all graph context parameters (e.g. dt are well copied)
        delta_node->updateSimulationContext();
    }


    /// @internal
    /// insert a ConstraintValue component in the given graph depending on restitution/damping values
    /// return possible pointer to the activated constraint mask
    template<class contact_dofs_type>
    helper::vector<bool>* addConstraintValue( node_type* node, contact_dofs_type* dofs/*, real damping*/, real restitution=0 )
    {
        assert( restitution>=0 && restitution<=1 );

        if( restitution ) // elastic contact
        {
            odesolver::Restitution::SPtr constraintValue = sofa::core::objectmodel::New<odesolver::Restitution>( dofs );
            node->addObject( constraintValue.get() );
//            constraintValue->dampingRatio.setValue( damping );
            constraintValue->restitution.setValue( restitution );

            // don't activate non-penetrating contacts
            odesolver::Restitution::mask_type& mask = *constraintValue->mask.beginWriteOnly();
            mask.resize( this->mappedContacts.size() );
            for(unsigned i = 0; i < this->mappedContacts.size(); ++i) {
                mask[i] = ( (*this->contacts)[i].value <= 0 );
            }
            constraintValue->mask.endEdit();

            constraintValue->init();
            baseConstraintValue = constraintValue;
            return &mask;
        }
//        else //if( damping ) // damped constraint
//        {
//            odesolver::ConstraintValue::SPtr constraintValue = sofa::core::objectmodel::New<odesolver::ConstraintValue>( dofs );
//            node->addObject( constraintValue.get() );
////            constraintValue->dampingRatio.setValue( damping );
//            constraintValue->init();
//        }
        else if( holonomic.getValue() ) // holonomic constraint (cancel relative velocity, w/o stabilization, contact penetration is not canceled)
        {
            // with stabilization holonomic and stabilization constraint values are equivalent

            typedef odesolver::HolonomicConstraintValue stab_type;
            stab_type::SPtr stab = sofa::core::objectmodel::New<stab_type>( dofs );
            node->addObject( stab.get() );

            // don't stabilize non-penetrating contacts (normal component only)
            odesolver::HolonomicConstraintValue::mask_type& mask = *stab->mask.beginWriteOnly();
            mask.resize(  this->mappedContacts.size() );
            for(unsigned i = 0; i < this->mappedContacts.size(); ++i) {
                mask[i] = ( (*this->contacts)[i].value <= 0 );
            }
            stab->mask.endEdit();

            stab->init();
            baseConstraintValue = stab;
            return &mask;
        }
        else // stabilized constraint
        {
            // stabilizer
            typedef odesolver::Stabilization stab_type;
            stab_type::SPtr stab = sofa::core::objectmodel::New<stab_type>( dofs );
            node->addObject( stab.get() );

            // don't stabilize non-penetrating contacts (normal component only)
            odesolver::Stabilization::mask_type& mask = *stab->mask.beginWriteOnly();
            mask.resize( this->mappedContacts.size() );
            for(unsigned i = 0; i < this->mappedContacts.size(); ++i) {
                mask[i] = ( (*this->contacts)[i].value <= 0 );
            }
            stab->mask.endEdit();

            stab->init();
            baseConstraintValue = stab;
            return &mask;
        }

        return NULL;
    }



//    typedef vector<real> distances_type;
//    void copyDistances( distances_type& res ) const {
//        const unsigned size = mappedContacts.size();
//        assert( size );

//        res.resize( size );
//        for(unsigned i = 0; i < size; ++i) {
//            res[i] = mappedContacts[i].distance;
//        }
//    }

    /// @internal copying the contact normals to the given vector
    typedef helper::vector< defaulttype::Vec<3, real> > normal_type;
    void copyNormals( normal_type& res ) const {
        const unsigned size = mappedContacts.size();
        assert( size );
        assert( size == contacts->size() );

        res.resize(size);

        for(unsigned i = 0; i < size; ++i) {
            res[i] = (*contacts)[i].normal;
        }
    }

    /// @internal copying the contact penetrations to the given vector
    template< class Coord >
    void copyPenetrations( helper::vector<Coord>& res ) const {
        const unsigned size = mappedContacts.size();
        assert( size );
        assert( size == contacts->size() );
        assert( size == res.size() );

        for(unsigned i = 0; i < size; ++i) {
            res[i][0] = (*contacts)[i].value;
        }
    }

    /// @internal copying the contact pair indices to the given vector
    void copyPairs( helper::vector< defaulttype::Vec<2, unsigned> >& res ) const {
        const unsigned size = mappedContacts.size();
        assert( size );
        assert( size == contacts->size() );

        res.resize( size );

        for(unsigned i = 0; i < size; ++i) {
            res[i][0] = this->mappedContacts[i].index1;
            res[i][1] = this->mappedContacts[i].index2;
        }
    }

};




//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////




/// a base class for compliant constraint based contact
template <class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types>
class BaseCompliantConstraintContact : public BaseContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>
{
public:

    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE3(BaseCompliantConstraintContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes),SOFA_TEMPLATE3(BaseContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes));

    typedef BaseContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes> Inherit;
    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::collision::Intersection Intersection;


    Data< SReal > compliance_value; ///< contact compliance: use model contact stiffnesses when < 0, use given value otherwise
    Data< SReal > restitution_coef; ///< global restitution coef

protected:

    BaseCompliantConstraintContact()
        : Inherit()
        , compliance_value( this->initData(&compliance_value, SReal(0.0), "compliance", "contact compliance: use model contact stiffnesses when < 0, use given value otherwise"))
        , restitution_coef( initData(&restitution_coef, SReal(0.0), "restitution", "global restitution coef") )
    {}

    BaseCompliantConstraintContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : Inherit( model1, model2, intersectionMethod )
        , compliance_value( this->initData(&compliance_value, SReal(0.0), "compliance", "contact compliance: use model contact stiffnesses when < 0, use given value otherwise"))
        , restitution_coef( initData(&restitution_coef, SReal(0.0), "restitution", "global restitution coef") )
    {}

};



}
}
}


#endif
