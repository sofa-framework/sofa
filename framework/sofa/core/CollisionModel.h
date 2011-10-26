/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_COLLISIONMODEL_H
#define SOFA_CORE_COLLISIONMODEL_H

#include <vector>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/CollisionElement.h>


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
        , proximity(initData(&proximity, 0.0, "proximity", "Distance to the actual (visual) surface"))
        , contactStiffness(initData(&contactStiffness, 10.0, "contactStiffness", "Default contact stiffness"))
        , contactFriction(initData(&contactFriction, 0.01, "contactFriction", "Default contact friction (damping) coefficient"))
        , contactResponse(initData(&contactResponse, "contactResponse", "if set, indicate to the ContactManager that this model should use the given class of contacts.\nNote that this is only indicative, and in particular if both collision models specify a different class it is up to the manager to choose."))
        , group(initData(&group, 0, "group", "If not zero, ID of a group containing this model. No collision can occur between collision models of the same group (allowing the same object to have multiple collision models)"))
        , color(initData(&color, defaulttype::Vec4f(1,0,0,1), "color", "color used to display the collision model if requested"))
        , size(0), numberOfContacts(0)
        , previous(initLink("previous", "Previous (coarser / upper / parent level) CollisionModel in the hierarchy."))
        , next(initLink("next", "Next (finer / lower / child level) CollisionModel in the hierarchy."))
    {
    }

    /// Destructor
    virtual ~CollisionModel()
    {

    }
public:
    virtual void bwdInit()
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
    /// Default to false if the collision models are attached to the same
    /// context (i.e. the same node in the scenegraph).
    virtual bool canCollideWith(CollisionModel* model)
    {
        if (model != this && this->group.getValue() != 0 && this->group.getValue() == model->group.getValue())
            return false;
        else if (model->getContext() != this->getContext())
            return true;
        else return bSelfCollision.getValue();
    }
    //virtual bool canCollideWith(CollisionModel* model) { return model != this; }

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
    virtual void draw(const core::visual::VisualParams* )
    {
#ifndef SOFA_DEPRECATE_OLD_API
        draw();
#endif
    }

#ifndef SOFA_DEPRECATE_OLD_API
    virtual void draw() {}

    virtual void draw(int /*index*/) {}
#endif

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
            pmodel->group.setValue(group.getValue());
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
    double getProximity() { return proximity.getValue(); }

    /// Get contact stiffness
    double getContactStiffness(int /*index*/) { return contactStiffness.getValue(); }
    /// Set contact stiffness
    void setContactStiffness(double stiffness) { contactStiffness.setValue(stiffness); }

    /// Get contact friction (damping) coefficient
    double getContactFriction(int /*index*/) { return contactFriction.getValue(); }
    /// Set contact friction (damping) coefficient
    void setContactFriction(double friction) { contactFriction.setValue(friction); }

    /// Contact response algorithm
    std::string getContactResponse() { return contactResponse.getValue(); }

    /// If not zero, ID of a group containing this model. No collision can occur between collision
    /// models of the same group (allowing the same object to have multiple collision models)
    int getGroup() const { return group.getValue(); }

    /// Set ID of group of this model. No collision can occur between collision
    /// models of the same group (allowing the same object to have multiple collision models)
    void setGroup(const int groupId) { group.setValue(groupId); }

    /// @}

    /// Topology associated to the collision model
    virtual Topology* getTopology() { return getContext()->getMeshTopology(); }

    /// BaseMeshTopology associated to the collision model
    virtual sofa::core::topology::BaseMeshTopology* getMeshTopology() { return getContext()->getMeshTopology(); }

    /// Get a color that can be used to display this CollisionModel
    const float* getColor4f();
    /// Set a color that can be used to display this CollisionModel
    void setColor4f(const float *c) {color.setValue(defaulttype::Vec4f(c[0],c[1],c[2],c[3]));};

    /// Set of differents parameters
    void setProximity       (const double a)      { proximity.setValue(a)        ;} ;
    void setContactResponse (const std::string &a) { contactResponse.setValue(a)  ;} ;


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
    Data<double> proximity;
    /// Default contact stiffness
    Data<double> contactStiffness;
    /// Default contact friction (damping) coefficient
    Data<double> contactFriction;
    /// contactResponse", "if set, indicate to the ContactManager that this model should use the
    /// given class of contacts.\nNote that this is only indicative, and in particular if both
    /// collision models specify a different class it is up to the manager to choose.
    Data<std::string> contactResponse;
    /// If not zero, ID of a group containing this model. No collision can occur between collision
    /// models of the same group (allowing the same object to have multiple collision models)
    Data<int> group;
    /// color used to display the collision model if requested
    Data<defaulttype::Vec4f> color;

    /// Number of collision elements
    int size;

    /// number of contacts attached to the collision model
    int numberOfContacts;

    /// Pointer to the previous (coarser / upper / parent level) CollisionModel in the hierarchy.
    SingleLink<CollisionModel,CollisionModel,BaseLink::FLAG_DOUBLELINK|BaseLink::FLAG_STRONGLINK> previous;

    /// Pointer to the next (finer / lower / child level) CollisionModel in the hierarchy.
    SingleLink<CollisionModel,CollisionModel,BaseLink::FLAG_DOUBLELINK> next;

};

} // namespace core

} // namespace sofa

#endif
