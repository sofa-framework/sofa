/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_COLLISIONMODEL_H
#define SOFA_CORE_COLLISIONMODEL_H

#include <vector>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/CollisionElement.h>

#include <sofa/defaulttype/RGBAColor.h>

namespace sofa
{

namespace core
{

namespace visual
{
class VisualParams;
}


/**
 *  \brief Abstract CollisionModel interface.
 *
 *  A CollisionModel contains a list of same-type elements. It can be part of a
 *  list of CollisionModels, each describing a level in a bounding-volume
 *  hierarchy.
 *
 *  Each CollisionModel stores a pointer to the next model in the hierarchy
 *  (i.e. finer / lower / child level) as well as the previous model (i.e.
 *  coarser / upper / parent level). The first CollisionModel in this list is
 *  the root of the hierarchy and contains only one element. The last
 *  CollisionModel contains the leaves of the hierarchy which are the real
 *  elements of the object.
 *
 *  Each element inside CollisionModels except for the last one can have a list
 *  of children. There are 2 types of child elements:
 *  \li internal children: child elements of the same type as their parent (often
 *    corresponding to non-final elements)
 *  \li external children: child elements of a different type (often corresponding
 *    to the final elements)
 *
 */
class SOFA_CORE_API CollisionModel : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(CollisionModel, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(CollisionModel)

    enum{
        AABB_TYPE = 0,
        OBB_TYPE,
        CAPSULE_TYPE,
        SPHERE_TYPE,
        TRIANGLE_TYPE,
        LINE_TYPE,
        POINT_TYPE,
        TETRAHEDRON_TYPE,
        RDISTANCE_GRIDE_TYPE,
        FFDDISTANCE_GRIDE_TYPE,
        CYLINDER_TYPE,
        ENUM_TYPE_SIZE
    };

    typedef CollisionElementIterator Iterator;
    typedef topology::BaseMeshTopology Topology;
    typedef sofa::defaulttype::Vector3::value_type Real;
protected:
    /// Constructor
    CollisionModel()
        : bActive(initData(&bActive, true, "active", "flag indicating if this collision model is active and should be included in default collision detections"))
        , bMoving(initData(&bMoving, true, "moving", "flag indicating if this object is changing position between iterations"))
        , bSimulated(initData(&bSimulated, true, "simulated", "flag indicating if this object is controlled by a simulation"))
        , bSelfCollision(initData(&bSelfCollision, false, "selfCollision", "flag indication if the object can self collide"))
        , proximity(initData(&proximity, (SReal)0.0, "proximity", "Distance to the actual (visual) surface"))
        , contactStiffness(initData(&contactStiffness, (SReal)10.0, "contactStiffness", "Contact stiffness"))
        , contactFriction(initData(&contactFriction, (SReal)0.0, "contactFriction", "Contact friction coefficient (dry or viscous or unused depending on the contact method)"))
        , contactRestitution(initData(&contactRestitution, (SReal)0.0, "contactRestitution", "Contact coefficient of restitution"))
        , contactResponse(initData(&contactResponse, "contactResponse", "if set, indicate to the ContactManager that this model should use the given class of contacts.\nNote that this is only indicative, and in particular if both collision models specify a different class it is up to the manager to choose."))
        , color(initData(&color, defaulttype::RGBAColor(1,0,0,1), "color", "color used to display the collision model if requested"))
        , group(initData(&group,"group","IDs of the groups containing this model. No collision can occur between collision models included in a common group (e.g. allowing the same object to have multiple collision models)"))
        , size(0)
        , numberOfContacts(0)
        , previous(initLink("previous", "Previous (coarser / upper / parent level) CollisionModel in the hierarchy."))
        , next(initLink("next", "Next (finer / lower / child level) CollisionModel in the hierarchy."))
        , userData(NULL)
    {
    }
    /// Destructor
    virtual ~CollisionModel()
    {

    }
public:
    virtual void bwdInit() override
    {
        getColor4f(); //init the color to default value
    }

    /// Return true if there are no elements
    bool empty() const
    {
        return size==0;
    }

    /// Get the number of elements.
    int getSize() const
    {
        return size;
    }

    /// Return true if this model process self collision
    bool getSelfCollision() const
    {
        return bSelfCollision.getValue();
    }

    /// set a value to bSelfCollision
    void setSelfCollision(bool _bSelfCollision)
    {
        bSelfCollision = _bSelfCollision ;
    }

    /// Get the number of contacts attached to the collision model
    int getNumberOfContacts() const
    {
        return numberOfContacts;
    }

    /// Set the number of contacts attached to the collision model
    void setNumberOfContacts(int i)
    {
        numberOfContacts = i;
    }

    /// Set the number of elements.
    virtual void resize(int s)
    {
        size = s;
    }

    /// Return an iterator to the first element.
    Iterator begin()
    {
        return Iterator(this,0);
    }

    /// Return an iterator pointing after the last element.
    Iterator end()
    {
        return Iterator(this,size);
    }

    /// Return the next (finer / lower / child level) CollisionModel in the hierarchy.
    CollisionModel* getNext()
    {
        return next.get();
    }

    /// Return the previous (coarser / upper / parent level) CollisionModel in the hierarchy.
    CollisionModel* getPrevious()
    {
        return previous.get();
    }

    /// Set the previous (coarser / upper / parent level) CollisionModel in the hierarchy.
    void setPrevious(CollisionModel::SPtr val)
    {
        CollisionModel::SPtr p = previous.get();
        if (p == val) return;
        if (p)
        {
            if (p->next.get()) p->next.get()->previous.reset();
            p->next.set(NULL);
        }
        if (val)
        {
            if (val->next.get()) val->next.get()->previous.set(NULL);
        }
        previous.set(val);
        if (val)
            val->next.set(this);
    }

    /// \brief Return true if this CollisionModel should be used for collisions.
    ///
    /// Default to true.
    virtual bool isActive() const { return bActive.getValue() && getContext()->isActive(); }

    /// \brief Set true if this CollisionModel should be used for collisions.
    virtual void setActive(bool val=true) { bActive.setValue(val); }

    /// \brief Return true if this CollisionModel is changing position between
    /// iterations.
    ///
    /// Default to true.
    virtual bool isMoving() const { return bMoving.getValue(); }

    /// \brief Set true if this CollisionModel is changing position between
    /// iterations.
    virtual void setMoving(bool val=true) { bMoving.setValue(val); }

    /// \brief Return true if this CollisionModel is attached to a simulation.
    /// It is false for immobile or procedurally animated objects that don't
    /// use contact forces
    ///
    /// Default to true.
    virtual bool isSimulated() const { return bSimulated.getValue(); }

    /// \brief Set true if this CollisionModel is attached to a simulation.
    virtual void setSimulated(bool val=true) { bSimulated.setValue(val); }

    /// Create or update the bounding volume hierarchy.
    virtual void computeBoundingTree(int maxDepth=0) = 0;

    /// \brief Create or update the bounding volume hierarchy, accounting for motions
    /// within the given timestep.
    ///
    /// Default to computeBoundingTree().
    virtual void computeContinuousBoundingTree(double /*dt*/, int maxDepth=0) { computeBoundingTree(maxDepth); }

    /// \brief Return the list (as a pair of iterators) of <i>internal children</i> of
    /// an element.
    ///
    /// Internal children are child elements of the same type as their parent
    /// (often corresponding to non-final elements). This distinction is used
    /// to optimize the intersection tests inside the hierarchy, as internal
    /// children can be processed without dynamically retrieving a new
    /// intersection method.
    ///
    /// Default to empty (i.e. two identical iterators)
    virtual std::pair<CollisionElementIterator,CollisionElementIterator> getInternalChildren(int /*index*/) const
    {
        return std::make_pair(CollisionElementIterator(),CollisionElementIterator());
    }

    /// \brief Return the list (as a pair of iterators) of <i>external children</i> of
    /// an element.
    ///
    /// External children are child elements of a different type than their
    /// parent (often corresponding to the final elements).
    ///
    /// Default to empty (i.e. two identical iterators)
    virtual std::pair<CollisionElementIterator,CollisionElementIterator> getExternalChildren(int /*index*/) const
    {
        return std::make_pair(CollisionElementIterator(),CollisionElementIterator());
    }


    /// \brief Checks if the element(index) is a leaf and a primitive of the collision model.
    ///
    /// Default to true since triangle model, line model, etc. does not have this method implemented and they
    /// are themselves (normally) leaves and primitives
    virtual bool isLeaf( int /*index*/ ) const
    {
        return true;  //e.g. Triangle will return true
    }


    /// \brief Test if this model can collide with another model.
    ///
    /// Note that this test is only related to <b>what</b> are the two models
    /// (i.e. which type, attached to which object) and not <b>where</b> they
    /// are in space. It is used to prune unnecessary or invalid collisions
    /// (i.e. vertices of an object should be tested with triangles of another
    /// but not the same object).
    ///
    /// Depending on selfCollision value if the collision models are attached to the same
    /// context (i.e. the same node in the scenegraph).
    /// If both models are included in a common "group", they won't collide
    virtual bool canCollideWith(CollisionModel* model)
    {
        if (model->getContext() == this->getContext()) // models are in the Node -> is self collision activated?
            return bSelfCollision.getValue();
        else if( this->group.getValue().empty() || model->group.getValue().empty() ) // one model has no group -> always collide
            return true;
        else
        {
            std::set<int>::const_iterator it = group.getValue().begin(), itend = group.getValue().end();
            for( ; it != itend ; ++it )
                if( model->group.getValue().count(*it)>0 ) // both models are included in the same group -> do not collide
                    return false;

            return true;
        }
    }

    /// \brief Test if two elements can collide with each other.
    ///
    /// This method should be implemented by models supporting
    /// self-collisions to prune tests between adjacent elements.
    ///
    /// Default to true. Note that this method assumes that canCollideWith(model2)
    /// was already used to test if the collision models can collide.
    virtual bool canCollideWithElement(int /*index*/, CollisionModel* /*model2*/, int /*index2*/) { return true; }


    /// Render an collision element.
    virtual void draw(const core::visual::VisualParams* /*vparams*/,int /*index*/) {}

    /// Render the whole collision model.
    virtual void draw(const core::visual::VisualParams* ) override
    {
    }


    /// Return the first (i.e. root) CollisionModel in the hierarchy.
    CollisionModel* getFirst()
    {
        CollisionModel *cm = this;
        CollisionModel *cm2;
        while ((cm2 = cm->getPrevious())!=NULL)
            cm = cm2;
        return cm;
    }

    /// Return the last (i.e. leaf) CollisionModel in the hierarchy.
    CollisionModel* getLast()
    {
        CollisionModel *cm = this;
        CollisionModel *cm2;
        while ((cm2 = cm->getNext())!=NULL)
            cm = cm2;
        return cm;
    }


    /// Helper method to get or create the previous model in the hierarchy.
    template<class DerivedModel>
    DerivedModel* createPrevious()
    {
        CollisionModel::SPtr prev = previous.get();
        typename DerivedModel::SPtr pmodel = sofa::core::objectmodel::SPtr_dynamic_cast<DerivedModel>(prev);
        if (pmodel.get() == NULL)
        {
            int level = 0;
            CollisionModel *cm = getNext();
            CollisionModel* root = this;
            while (cm) { root = cm; cm = cm->getNext(); ++level; }
            pmodel = sofa::core::objectmodel::New<DerivedModel>();
            pmodel->setName("BVLevel",level);
            root->addSlave(pmodel); //->setContext(getContext());
            pmodel->setMoving(isMoving());
            pmodel->setSimulated(isSimulated());
            pmodel->proximity.setValue(proximity.getValue());
            //pmodel->group.setValue(group_old.getValue());
            pmodel->group.beginEdit()->insert(group.getValue().begin(),group.getValue().end());
            pmodel->group.endEdit();
            //previous=pmodel;
            //pmodel->next = this;
            setPrevious(pmodel);
            if (prev)
            {

            }
        }
        return pmodel.get();
    }

    /// @name Experimental methods
    /// @{

    /// Get distance to the actual (visual) surface
    SReal getProximity() { return proximity.getValue(); }

    /// Get contact stiffness
    SReal getContactStiffness(int /*index*/) { return contactStiffness.getValue(); }
    /// Set contact stiffness
    void setContactStiffness(SReal stiffness) { contactStiffness.setValue(stiffness); }

    /// Get contact friction (damping) coefficient
    SReal getContactFriction(int /*index*/) { return contactFriction.getValue(); }
    /// Set contact friction (damping) coefficient
    void setContactFriction(SReal friction) { contactFriction.setValue(friction); }

    /// Get contact coefficient of restitution
     SReal getContactRestitution(int /*index*/) { return contactRestitution.getValue(); }
    /// Set contact coefficient of restitution
    void setContactRestitution(SReal restitution) { contactRestitution.setValue(restitution); }

    /// Contact response algorithm
    std::string getContactResponse() { return contactResponse.getValue(); }



    /// Return the group IDs containing this model.
    const std::set<int>& getGroups() const { return group.getValue(); }

    /// add the group ID to this model.
    void addGroup(const int groupId) { group.beginEdit()->insert(groupId); group.endEdit(); }

    /// Set the group IDs to this model
    void setGroups(const std::set<int>& ids) { group.setValue(ids); }
    /// @}


    /// Topology associated to the collision model
    virtual Topology* getTopology() { return getContext()->getMeshTopology(); }

    /// BaseMeshTopology associated to the collision model
    virtual sofa::core::topology::BaseMeshTopology* getMeshTopology() { return getContext()->getMeshTopology(); }

    /// Get a color that can be used to display this CollisionModel
    const float* getColor4f();
    /// Set a color that can be used to display this CollisionModel

    void setColor4f(const float *c) {
        color.setValue(defaulttype::RGBAColor(c[0],c[1],c[2],c[3]));
    }

    /// Set of differents parameters
    void setProximity       (const SReal a)        { proximity.setValue(a); }
    void setContactResponse (const std::string &a) { contactResponse.setValue(a); }

    /// Returns an int corresponding to the type of this.
    /// Useful for optimizations involving static_cast.
    int getEnumType() const {
        return enum_type;
    }

    /// Set user data
    void SetUserData(void* pUserData)  { userData = pUserData; }

    /// Get user data
    void* GetUserData() { return userData; }

protected:

    /// flag indicating if this collision model is active and should be included in default
    /// collision detections
    Data<bool> bActive;
    ///flag indicating if this object is changing position between iterations
    Data<bool> bMoving;
    /// flag indicating if this object is controlled by a simulation
    Data<bool> bSimulated;
    /// flag indication if the object can self collide
    Data<bool> bSelfCollision;
    /// Distance to the actual (visual) surface
    Data<SReal> proximity;
    /// Default contact stiffness
    Data<SReal> contactStiffness;
    /// Default contact friction (damping) coefficient
    Data<SReal> contactFriction;
    /// Default contact coefficient of restitution
    Data<SReal> contactRestitution;
    /// If non-empty, indicate to the ContactManager that this model should use the
    /// given class of contacts. Note that this is only indicative, and in particular if both
    /// collision models specify a different class it is up to the manager to choose.
    Data<std::string> contactResponse;

    /// color used to display the collision model if requested
    Data<defaulttype::RGBAColor> color;

    /// No collision can occur between collision
    /// models included in a common group (i.e. sharing a common id)
    Data< std::set<int> > group;

    /// Number of collision elements
    int size;

    /// number of contacts attached to the collision model
    int numberOfContacts;

    /// Pointer to the previous (coarser / upper / parent level) CollisionModel in the hierarchy.
    SingleLink<CollisionModel,CollisionModel,BaseLink::FLAG_DOUBLELINK|BaseLink::FLAG_STRONGLINK> previous;

    /// Pointer to the next (finer / lower / child level) CollisionModel in the hierarchy.
    SingleLink<CollisionModel,CollisionModel,BaseLink::FLAG_DOUBLELINK> next;

    /// an int corresponding to the type of this.
    /// Useful for optimizations involving static_cast
    int enum_type;

    void* userData;

public:

    virtual bool insertInNode( objectmodel::BaseNode* node ) override;
    virtual bool removeInNode( objectmodel::BaseNode* node ) override;

};

} // namespace core

} // namespace sofa

#endif
