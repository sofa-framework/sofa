/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/CollisionElement.h>

//todo(dmarchal 2018-06-19) I really wonder why a collision model has a dependency to a RGBAColors.
#include <sofa/type/RGBAColor.h>

namespace sofa::core
{

namespace visual
{
class VisualParams;
}


class CollisionElementActiver
{
public:
    CollisionElementActiver() {}
    virtual ~CollisionElementActiver() {}
    virtual bool isCollElemActive(sofa::Index /*index*/, core::CollisionModel * /*cm*/ = nullptr) { return true; }
    static CollisionElementActiver* getDefaultActiver() { static CollisionElementActiver defaultActiver; return &defaultActiver; }
};

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
    typedef sofa::type::Vec3::value_type Real;
    using Index = sofa::Index;
    using Size = sofa::Size;

protected:
    /// Constructor
    CollisionModel() ;

    /// Destructor
    ~CollisionModel() override {}

public:
    void bwdInit() override;

    /// Return true if there are no elements
    bool empty() const
    {
        return size==0;
    }

    /// Get the number of elements.
    Size getSize() const
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
    Size getNumberOfContacts() const
    {
        return d_numberOfContacts.getValue();
    }

    /// Set the number of contacts attached to the collision model
    void setNumberOfContacts(Size i)
    {
        d_numberOfContacts.setValue(i);
    }

    /// Set the number of elements.
    virtual void resize(Size s)
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
    void setPrevious(CollisionModel::SPtr val) ;

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
    virtual void computeContinuousBoundingTree(SReal /*dt*/, int maxDepth=0) { computeBoundingTree(maxDepth); }

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
    virtual std::pair<CollisionElementIterator,CollisionElementIterator> getInternalChildren(Index /*index*/) const
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
    virtual std::pair<CollisionElementIterator,CollisionElementIterator> getExternalChildren(Index /*index*/) const
    {
        return std::make_pair(CollisionElementIterator(),CollisionElementIterator());
    }

    /// \brief Checks if the element(index) is a leaf and a primitive of the collision model.
    ///
    /// Default to true since triangle model, line model, etc. does not have this method implemented and they
    /// are themselves (normally) leaves and primitives
    virtual bool isLeaf(Index /*index*/ ) const
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
    virtual bool canCollideWith(CollisionModel* model) ;

    /// \brief Test if two elements can collide with each other.
    ///
    /// This method should be implemented by models supporting
    /// self-collisions to prune tests between adjacent elements.
    ///
    /// Default to true. Note that this method assumes that canCollideWith(model2)
    /// was already used to test if the collision models can collide.
    virtual bool canCollideWithElement(Index /*index*/, CollisionModel* /*model2*/, Index /*index2*/) { return true; }

    /// Render an collision element.
    virtual void draw(const core::visual::VisualParams* /*vparams*/, Index /*index*/) {}

    /// Render the whole collision model.
    void draw(const core::visual::VisualParams* ) override {}

    /// Return the first (i.e. root) CollisionModel in the hierarchy.
    CollisionModel* getFirst();

    /// Return the last (i.e. leaf) CollisionModel in the hierarchy.
    CollisionModel* getLast();

    /// Helper method to get or create the previous model in the hierarchy.
    template<class DerivedModel>
    DerivedModel* createPrevious()
    {
        CollisionModel::SPtr prev = previous.get();
        typename DerivedModel::SPtr pmodel = sofa::core::objectmodel::SPtr_dynamic_cast<DerivedModel>(prev);
        if (pmodel.get() == nullptr)
        {
            int level = 0;
            CollisionModel *cm = getNext();
            CollisionModel* root = this;
            while (cm) 
            {
                root = cm;
                cm = cm->getNext();
                ++level;
            }

            pmodel = sofa::core::objectmodel::New<DerivedModel>();
            pmodel->setName("BVLevel", level);
            root->addSlave(pmodel); 
            pmodel->setMoving(isMoving());
            pmodel->setSimulated(isSimulated());
            pmodel->proximity.setParent(&proximity);
			
            pmodel->group.beginEdit()->insert(group.getValue().begin(), group.getValue().end());
            pmodel->group.endEdit();
            pmodel->f_listening.setParent(&f_listening);
            pmodel->f_printLog.setParent(&f_printLog);
			
            setPrevious(pmodel);			
        }
        return pmodel.get();
    }

    /// @name Experimental methods
    /// @{

    /// Get distance to the actual (visual) surface
    [[nodiscard]] SReal getProximity() const { return proximity.getValue(); }

    /// Get contact stiffness
    [[nodiscard]] SReal getContactStiffness(Index /*index*/) const { return contactStiffness.getValue(); }
    /// Set contact stiffness
    void setContactStiffness(SReal stiffness) { contactStiffness.setValue(stiffness); }
    /// Get contact stiffness
    [[nodiscard]] bool isContactStiffnessSet() const { return contactStiffness.isSet(); }

    /// Get contact friction (damping) coefficient
    [[nodiscard]] SReal getContactFriction(Index /*index*/) const { return contactFriction.getValue(); }
    /// Set contact friction (damping) coefficient
    void setContactFriction(SReal friction) { contactFriction.setValue(friction); }

    /// Get contact coefficient of restitution
    [[nodiscard]] SReal getContactRestitution(Index /*index*/) const { return contactRestitution.getValue(); }
    /// Set contact coefficient of restitution
    void setContactRestitution(SReal restitution) { contactRestitution.setValue(restitution); }

    /// Contact response algorithm
    [[nodiscard]] std::string getContactResponse() const { return contactResponse.getValue(); }

    /// Return the group IDs containing this model.
    [[nodiscard]] const std::set<int>& getGroups() const { return group.getValue(); }

    /// add the group ID to this model.
    void addGroup(const int groupId) { group.beginEdit()->insert(groupId); group.endEdit(); }

    /// Set the group IDs to this model
    void setGroups(const std::set<int>& ids) { group.setValue(ids); }
    /// @}

    /// BaseMeshTopology associated to the collision model. TODO: epernod remove virtual pure method by l_topology.get as soons as new link will be available
    virtual sofa::core::topology::BaseMeshTopology* getCollisionTopology() { return nullptr; }

    /// Get a color that can be used to display this CollisionModel
    const float* getColor4f();
    /// Set a color that can be used to display this CollisionModel

    void setColor4f(const float *c);

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
    Data<sofa::type::RGBAColor> color;

    /// No collision can occur between collision
    /// models included in a common group (i.e. sharing a common id)
    Data< std::set<int> > group;

    /// Number of collision elements
    Size size;

    /// number of contacts attached to the collision model
    Data<Size> d_numberOfContacts;

    /// Pointer to the previous (coarser / upper / parent level) CollisionModel in the hierarchy.
    SingleLink<CollisionModel,CollisionModel,BaseLink::FLAG_DOUBLELINK|BaseLink::FLAG_STRONGLINK> previous;

    /// Pointer to the next (finer / lower / child level) CollisionModel in the hierarchy.
    SingleLink<CollisionModel,CollisionModel,BaseLink::FLAG_DOUBLELINK> next;

    /// an int corresponding to the type of this.
    /// Useful for optimizations involving static_cast
    int enum_type;

    void* userData;

    /// Pointer to the  Controller component heritating from CollisionElementActiver
    SingleLink<CollisionModel, sofa::core::objectmodel::BaseObject, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_collElemActiver;

public:
    CollisionElementActiver *myCollElemActiver; ///< CollisionElementActiver that activate or deactivate collision element during execution

    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

};
} // namespace sofa::core
