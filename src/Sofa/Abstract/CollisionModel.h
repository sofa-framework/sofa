#ifndef SOFA_ABSTRACT_COLLISIONMODEL_H
#define SOFA_ABSTRACT_COLLISIONMODEL_H

#include <vector>
#include "BaseObject.h"
#include "CollisionElement.h"

namespace Sofa
{

namespace Abstract
{

class CollisionModel : public virtual BaseObject
{
public:

    typedef CollisionElementIterator Iterator;

    CollisionModel()
        : size(0), previous(NULL), next(NULL)
    {
    }

    virtual ~CollisionModel() { }

    virtual void resize(int s)
    {
        size = s;
        //bbox.resize(s);
        //timestamp.resize(s);
    }

    Iterator begin()
    {
        return Iterator(this,0);
    }

    Iterator end()
    {
        return Iterator(this,size);
    }

    bool empty() const
    {
        return size==0;
    }

    CollisionModel* getNext()
    {
        return next;
    }

    CollisionModel* getPrevious()
    {
        return previous;
    }

    void setNext(CollisionModel* val)
    {
        next = val;
    }

    void setPrevious(CollisionModel* val)
    {
        previous = val;
    }

    virtual bool isActive() { return true; }

    virtual bool isStatic() { return false; }

    virtual void computeBoundingTree(int maxDepth=0) = 0;

    virtual void computeContinuousBoundingTree(double /*dt*/, int maxDepth=0) { computeBoundingTree(maxDepth); }

    virtual std::pair<CollisionElementIterator,CollisionElementIterator> getInternalChildren(int /*index*/) const
    {
        return std::make_pair(CollisionElementIterator(),CollisionElementIterator());
    }

    virtual std::pair<CollisionElementIterator,CollisionElementIterator> getExternalChildren(int /*index*/) const
    {
        return std::make_pair(CollisionElementIterator(),CollisionElementIterator());
    }

    virtual bool canCollideWith(CollisionModel* model) { return model->getContext() != this->getContext(); }
    //virtual bool canCollideWith(CollisionModel* model) { return model != this; }

    virtual bool canCollideWithElement(int /*index*/, CollisionModel* model2, int /*index2*/) { return canCollideWith(model2); }

    virtual void draw(int /*index*/) {}

    CollisionModel* getFirst()
    {
        CollisionModel *cm = this;
        CollisionModel *cm2;
        while ((cm2 = cm->getPrevious())!=NULL)
            cm = cm2;
        return cm;
    }

    CollisionModel* getLast()
    {
        CollisionModel *cm = this;
        CollisionModel *cm2;
        while ((cm2 = cm->getNext())!=NULL)
            cm = cm2;
        return cm;
    }

    /*
    	const double* getBBoxMin(int index) const { return bbox[index].min; }

    	const double* getBBoxMax(int index) const { return bbox[index].max; }
    */

    /*
    	static void clearAllVisits()
    	{
    		CurrentTimeStamp(1);
    	}

    	bool visited(int index) const
    	{
    		return timestamp[index] == CurrentTimeStamp();
    	}

    	void setVisited(int index)
    	{
    		timestamp[index] = CurrentTimeStamp();
    	}
    */

    // Nodes of the hierarchy of the collision model
    std::vector< CollisionElementIterator* > nodes;


protected:
    int size;

    CollisionModel* previous;
    CollisionModel* next;

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

    /*
    	struct BBox
    	{
    		double min[3];
    		double max[3];
    	};
    	std::vector<BBox> bbox;
    */

    /*
    	std::vector<int> timestamp;

    	static int CurrentTimeStamp(int incr=0)
    	{
    		static int ts = 0;
    		ts += incr;
    		return ts;
    	}
    */
};

} // namespace Abstract

} // namespace Sofa

#endif
