#ifndef SOFA_CORE_COLLISIONMODEL_H
#define SOFA_CORE_COLLISIONMODEL_H

#include <vector>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/CollisionElement.h>

namespace sofa
{

namespace core
{

/// Abstract CollisionModel interface.
///
/// A CollisionModel contains a list of same-type elements. It can be part of a
/// list of CollisionModels, each describing a level in a bounding-volume
/// hierarchy.
///
/// Each CollisionModel stores a pointer to the next model in the hierarchy
/// (i.e. finer / lower / child level) as well as the previous model (i.e.
/// coarser / upper / parent level). The first CollisionModel in this list is
/// the root of the hierarchy and contains only one element. The last
/// CollisionModel contains the leaves of the hierarchy which are the real
/// elements of the object.
///
/// Each element inside CollisionModels except for the last one can have a list
/// of children. There are 2 types of child elements:
/// * internal children: child elements of the same type as their parent (often
///   corresponding to non-final elements)
/// * external children: child elements of a different type (often corresponding
///   to the final elements)
///
class CollisionModel : public virtual objectmodel::BaseObject
{
public:

    typedef CollisionElementIterator Iterator;

    CollisionModel()
        : size(0), previous(NULL), next(NULL)
    {
    }

    virtual ~CollisionModel() { }

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

    /// Return true if there are no elements
    bool empty() const
    {
        return size==0;
    }

    /// Return the next (i.e. finer / lower / child level)
    /// CollisionModel in the hierarchy.
    CollisionModel* getNext()
    {
        return next;
    }

    /// Return the previous (i.e. coarser / upper / parent level)
    /// CollisionModel in the hierarchy.
    CollisionModel* getPrevious()
    {
        return previous;
    }

    /// Set the next (i.e. finer / lower / child level)
    /// CollisionModel in the hierarchy.
    void setNext(CollisionModel* val)
    {
        next = val;
    }

    /// Set the previous (i.e. coarser / upper / parent level)
    /// CollisionModel in the hierarchy.
    void setPrevious(CollisionModel* val)
    {
        previous = val;
    }

    /// Return true if this CollisionModel should be used for collisions.
    ///
    /// Default to true.
    virtual bool isActive() { return true; }

    /// Return true if this CollisionModel is attached to an immobile
    /// <i>obstacle</i> object.
    ///
    /// Default to false.
    virtual bool isStatic() { return false; }

    /// Create or update the bounding volume hierarchy.
    virtual void computeBoundingTree(int maxDepth=0) = 0;

    /// Create or update the bounding volume hierarchy, accounting for motions
    /// within the given timestep.
    ///
    /// Default to computeBoundingTree().
    virtual void computeContinuousBoundingTree(double /*dt*/, int maxDepth=0) { computeBoundingTree(maxDepth); }

    /// Return the list (as a pair of iterators) of <i>internal children</i> of
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

    /// Return the list (as a pair of iterators) of <i>external children</i> of
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


    /// Checks if the element(index) is a leaf and a primitive of the collision model
    /// Default to true since triangle model, line model, etc. does not have this method implemented and they
    /// are themselves (normally) leaves and primitives
    virtual bool isLeaf( int /*index*/ ) const
    {
        return true;  //e.g. Triangle will return true
    }


    /// Test if this model can collide with another model.
    ///
    /// Note that this test is only related to <b>what</b> are the two models
    /// (i.e. which type, attached to which object) and not <b>where</b> they
    /// are in space. It is used to prune unnecessary or invalid collisions
    /// (i.e. vertices of an object should be tested with triangles of another
    /// but not the same object).
    ///
    /// Default to false if the collision models are attached to the same
    /// context (i.e. the same node in the scenegraph).
    virtual bool canCollideWith(CollisionModel* model) { return model->getContext() != this->getContext(); }
    //virtual bool canCollideWith(CollisionModel* model) { return model != this; }

    /// Test if two elements can collide with each other.
    ///
    /// This method should be implemented by models supporting
    /// self-collisions to prune tests between adjacent elements.
    ///
    /// Default to canCollideWith(model2)
    virtual bool canCollideWithElement(int /*index*/, CollisionModel* model2, int /*index2*/) { return canCollideWith(model2); }

    /// Render an collision element.
    virtual void draw(int /*index*/) {}

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

protected:

    /// Number of collision elements
    int size;

    /// Pointer to the previous (i.e. coarser / upper / parent level)
    /// CollisionModel in the hierarchy.
    CollisionModel* previous;

    /// Pointer to the next (i.e. finer / lower / child level)
    /// CollisionModel in the hierarchy.
    CollisionModel* next;

    /// Helper method to get or create the previous model in the hierarchy.
    template<class DerivedModel>
    DerivedModel* createPrevious()
    {
        DerivedModel* pmodel = dynamic_cast<DerivedModel*>(previous);
        if (pmodel == NULL)
        {
            if (previous != NULL)
                delete previous;
            pmodel = new DerivedModel();
            pmodel->setContext(getContext());
            pmodel->setStatic(isStatic());
            previous = pmodel;
            pmodel->setNext(this);
        }
        return pmodel;
    }
};

} // namespace core

} // namespace sofa

#endif
