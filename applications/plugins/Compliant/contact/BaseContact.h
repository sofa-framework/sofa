#ifndef BASECONTACT_H
#define BASECONTACT_H

#include <SofaBaseCollision/DefaultContactManager.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <SofaBaseCollision/BaseContactMapper.h>

#include "../initCompliant.h"

#include "../mapping/DifferenceMapping.h"

#include "../utils/edit.h"
#include "../constraint/Restitution.h"
#include "../constraint/Stabilization.h"


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



    Data< SReal > damping_ratio;
    Data<bool> holonomic;

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
    { }

    BaseContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : damping_ratio( this->initData(&damping_ratio, SReal(0.0), "damping", "contact damping (used for stabilization)") )
        , holonomic(this->initData(&holonomic, false, "holonomic", "only enforce null relative velocity, do not try to remove penetration during the dynamics pass"))
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
        assert( contacts );

        if( node )
        {
            mapper1.cleanup();
            mapper2.cleanup();
        }

        // fancy names
        std::string name1 = this->model1->getClassName() + " contact points";
        std::string name2 = this->model2->getClassName() + " contact points";

        // obtain point mechanical models from mappers
        mstate1 = mapper1.createMapping( name1.c_str() );

        mstate2 = this->selfCollision ? mstate1 : mapper2.createMapping( name2.c_str() );
        mstate2->setName("dofs");


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



        // if(! node ) {
        node = create_node();
        // } else {
        // 	update_node();
        // }

        //		assert( dynamic_cast< node_type* >( group ) );

        // TODO is this needed ?!
        // static_cast< node_type* >( group )->addChild( node );
    }


    void removeResponse() {
        // std::cout << "removeResponse" << std::endl;
        if( node ) {
            mapper1.resize(0);
            mapper2.resize(0);

            node->detachFromGraph();
        }

    }

    void cleanup() {
        // std::cout << "cleanup" << std::endl;

        if( node ) {

            mapper1.cleanup();

            if (!this->selfCollision) mapper2.cleanup();

            // TODO can/should we delete node here ?
        }

        mappedContacts.clear();
    }



protected:

    // the node that will hold all the stuff
    typedef sofa::simulation::Node node_type;
    node_type::SPtr node;

    typename MechanicalState1::SPtr mstate1;
    typename MechanicalState2::SPtr mstate2;

    // all you're left to implement \o/
    virtual node_type::SPtr create_node() = 0;
    virtual void update_node(node_type::SPtr node) = 0;

    // TODO correct real type
    typedef container::MechanicalObject<ResponseDataTypes> delta_dofs_type;

    // convenience
    struct delta_type { // TODO to compliantconstraint
        node_type::SPtr node;
        typename delta_dofs_type::SPtr dofs;
    };

    delta_type make_delta() const {  // TODO to compliantconstraint
        node_type::SPtr delta_node = node_type::create( this->getName() + " delta" );

        // TODO vec3types according to the types in interaction !

        // delta dofs
        typename delta_dofs_type::SPtr delta_dofs;
        const unsigned size = mappedContacts.size();
        assert( size );

        delta_dofs = core::objectmodel::New<delta_dofs_type>();
        delta_dofs->setName( this->model2->getName() + " - " + this->model1->getName()  );
        delta_dofs->resize( size );
        
//        delta_dofs->showObject.setValue(true);
//        delta_dofs->showObjectScale.setValue(10);

        delta_node->addObject( delta_dofs );

        // index pairs and stuff
        vector< defaulttype::Vec<2, unsigned> > pairs(size);

        for(unsigned i = 0; i < size; ++i) {
            pairs[i][0] = mappedContacts[i].index1;
            pairs[i][1] = mappedContacts[i].index2;
        }


        // delta mappings
        if( this->selfCollision ) {
            typedef mapping::DifferenceMapping<ResponseDataTypes, ResponseDataTypes> map_type;
            typename map_type::SPtr map = sofa::core::objectmodel::New<map_type>();

            map->setModels( this->mstate1.get(), delta_dofs.get() );

            map->pairs.setValue( pairs );

            map->setName( "delta mapping" );

            delta_node->addObject( map.get() );

            // TODO is there a cleaner way of doing this ?

            // this is of uttermost importance
            assert( dynamic_cast<node_type*>(this->mstate1->getContext()) );
            static_cast< node_type* >(this->mstate1->getContext())->addChild( delta_node.get() );

            map->init();

        } else {
            typedef mapping::DifferenceMultiMapping<ResponseDataTypes, ResponseDataTypes> map_type;
            typename map_type::SPtr map = core::objectmodel::New<map_type>();

            map->addInputModel( this->mstate1.get() );
            map->addInputModel( this->mstate2.get() );

            map->addOutputModel( delta_dofs.get() );

            map->pairs.setValue( pairs );

            map->setName( "delta mapping" );

            delta_node->addObject( map.get() );

            // the scene graph should reflect mapping dependencies
            assert( dynamic_cast<node_type*>(this->mstate1->getContext()) );
            assert( dynamic_cast<node_type*>(this->mstate2->getContext()) );

            static_cast< node_type* >(this->mstate1->getContext())->addChild( delta_node.get() );
            static_cast< node_type* >(this->mstate2->getContext())->addChild( delta_node.get() );

            map->init();

        }

        // ensure all graph context parameters (e.g. dt are well copied)
        delta_node->updateSimulationContext();

        delta_type res;
        res.node = delta_node;
        res.dofs = delta_dofs;

        return res;
    }


    typedef SReal real;
// TODO to compliantconstraint
    /// insert a ConstraintValue component in the given graph depending on restitution/damping values
    template<class contact_dofs_type>
    void addConstraintValue( node_type* node, contact_dofs_type* dofs/*, real damping*/, real restitution=0, unsigned size = 1)
    {
        assert( restitution>=0 && restitution<=1 );

        if( restitution ) // elastic contact
        {
            odesolver::Restitution::SPtr constraintValue = sofa::core::objectmodel::New<odesolver::Restitution>( dofs );
            node->addObject( constraintValue.get() );
//            constraintValue->dampingRatio.setValue( damping );
            constraintValue->restitution.setValue( restitution );

            // don't activate non-penetrating contacts
            edit(constraintValue->mask)->resize( size * this->mappedContacts.size() );
            for(unsigned i = 0; i < this->mappedContacts.size(); ++i) {
				// (max) watch out: editor is destroyed just after
				// operator* is called, thus endEdit is called at this
				// time !
                (*edit(constraintValue->mask))[size * i] = ( (*this->contacts)[i].value <= 0 );

				for(unsigned j = 1; j < size; ++j) {
					(*edit(constraintValue->mask))[size * i + j] = false;
				}
            }

            constraintValue->init();
        }
//        else //if( damping ) // damped constraint
//        {
//            odesolver::ConstraintValue::SPtr constraintValue = sofa::core::objectmodel::New<odesolver::ConstraintValue>( dofs );
//            node->addObject( constraintValue.get() );
////            constraintValue->dampingRatio.setValue( damping );
//            constraintValue->init();
//        }
        else // stabilized constraint
        {
            // stabilizer
            typedef odesolver::Stabilization stab_type;
            stab_type::SPtr stab = sofa::core::objectmodel::New<stab_type>( dofs );
            stab->m_holonomic = holonomic.getValue();
            node->addObject( stab.get() );

            // don't stabilize non-penetrating contacts (normal component only)
            edit(stab->mask)->resize( size * this->mappedContacts.size() );
            for(unsigned i = 0; i < this->mappedContacts.size(); ++i) {
                (*edit(stab->mask))[size * i] = ( (*this->contacts)[i].value <= 0 );

                for(unsigned j = 1; j < size; ++j) {
                    (*edit(stab->mask))[size * i + j] = false;
                }
            }

            stab->init();
        }
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

    typedef vector< defaulttype::Vec<3, real> > normal_type;
    void copyNormals( normal_type& res ) const {
        const unsigned size = mappedContacts.size();
        assert( size );
        assert( size == contacts->size() );

        res.resize(size);

        for(unsigned i = 0; i < size; ++i) {
            res[i] = (*contacts)[i].normal;
        }
    }

    typedef vector< real > penetration_type;
    void copyPenetrations( penetration_type& res ) const {
        const unsigned size = mappedContacts.size();
        assert( size );
        assert( size == contacts->size() );

        res.resize(size);

        for(unsigned i = 0; i < size; ++i) {
            res[i] = (*contacts)[i].value;
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


    Data< SReal > compliance_value;
    Data< SReal > restitution_coef;

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
