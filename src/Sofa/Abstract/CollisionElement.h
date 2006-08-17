#ifndef SOFA_ABSTRACT_COLLISIONELEMENT_H
#define SOFA_ABSTRACT_COLLISIONELEMENT_H

#include <vector>

namespace Sofa
{

namespace Abstract
{

class CollisionModel;
class CollisionElementIterator;

template<class Model>
class TCollisionElementIterator
{
public:

    TCollisionElementIterator(Model* model=NULL, int index=0)
        : model(model), index(index)
    {
    }

    template<class Model2>
    bool operator==(const TCollisionElementIterator<Model2>& i) const
    {
        return this->model == i.getCollisionModel() && this->index == i.getIndex();
    }

    template<class Model2>
    bool operator!=(const TCollisionElementIterator<Model2>& i) const
    {
        return this->model != i.getCollisionModel() || this->index != i.getIndex();
    }

    void operator++()
    {
        ++index;
    }

    void operator++(int)
    {
        ++index;
    }

    Model* getCollisionModel() const
    {
        return model;
    }

    int getIndex() const
    {
        return index;
    }

    bool valid() const
    {
        return model!=NULL;
    }

    std::pair<CollisionElementIterator,CollisionElementIterator> getInternalChildren() const;

    std::pair<CollisionElementIterator,CollisionElementIterator> getExternalChildren() const;

    bool canCollideWith(TCollisionElementIterator<Model>& elem)
    {
        return model->canCollideWithElement(index, elem.model, elem.index);
    }

    void draw()
    {
        model->draw(index);
    }

    /*
    	const double* getBBoxMin() const
    	{
    		return model->getBBoxMin(index);
    	}

    	const double* getBBoxMax() const
    	{
    		return model->getBBoxMax(index);
    	}
    */

    /*
    	bool visited() const
    	{
    		return model->visited(index);
    	}

    	void setVisited()
    	{
    		model->setVisited(index);
    	}
    */

    TCollisionElementIterator<Model>* operator->()
    {
        return this;
    }

    const TCollisionElementIterator<Model>* operator->() const
    {
        return this;
    }

protected:
    Model* model;
    int index;
};

class CollisionElementIterator : public TCollisionElementIterator<CollisionModel>
{
public:
    CollisionElementIterator(CollisionModel* model=NULL, int index=0)
        : TCollisionElementIterator<CollisionModel>(model, index)
    {
    }
    template<class DerivedModel>
    CollisionElementIterator(const TCollisionElementIterator<DerivedModel>& i)
        : TCollisionElementIterator<CollisionModel>(i.getCollisionModel(), i.getIndex())
    {
    }
    template<class DerivedModel>
    void operator=(const TCollisionElementIterator<DerivedModel>& i)
    {
        this->model = i.getCollisionModel();
        this->index = i.getIndex();
    }
};


template<class Model>
std::pair<CollisionElementIterator,CollisionElementIterator> TCollisionElementIterator<Model>::getInternalChildren() const
{
    return model->getInternalChildren(index);
}

template<class Model>
std::pair<CollisionElementIterator,CollisionElementIterator> TCollisionElementIterator<Model>::getExternalChildren() const
{
    return model->getExternalChildren(index);
}

} // namespace Abstract

} // namespace Sofa

#endif
