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
#ifndef SOFA_CORE_COLLISIONELEMENT_H
#define SOFA_CORE_COLLISIONELEMENT_H
#include <sofa/core/core.h>

#include <cstddef>
#include <vector>

namespace sofa
{

namespace core
{

namespace visual
{
class VisualParams;
} // namespace visual

class CollisionModel;
class CollisionElementIterator;

/**
 *  \brief Base class for reference to an collision element defined by its <i>index</i>
 *
 */
class BaseCollisionElementIterator
{
public:
    typedef std::vector<int>::const_iterator VIterator;

    /// Constructor.
    /// In most cases it will be used by the CollisionModel to
    /// create interators to its elements (such as in the begin() and end()
    /// methods).
    BaseCollisionElementIterator(int index=0)
        : index(index), it(emptyVector.begin()), itend(emptyVector.end())
    {
    }

    /// Constructor.
    /// This constructor should be used in case a vector of indices is used.
    BaseCollisionElementIterator(int index, VIterator it, VIterator itend)
        : index(index), it(it), itend(itend)
    {
    }

    /// Constructor.
    /// This constructor should be used in case a vector of indices is used.
    BaseCollisionElementIterator(VIterator it, VIterator itend)
        : index(-1), it(it), itend(itend)
    {
        if (it != itend) index = *it;
    }

    /// @name Iterator Interface
    /// @{

    /// Increment this iterator to reference the next element.
    void next()
    {
        if (it == itend)
            ++index;
        else
        {
            ++it;
            if (it != itend) index = *it;
        }
    }

    /// Increment this iterator to reference the next element.
    void operator++()
    {
        next();
    }

    /// Increment this iterator to reference the next element.
    void operator++(int)
    {
        next();
    }

    /// Return the index of the referenced element inside the CollisionModel.
    ///
    /// This methods should rarely be used.
    /// Users should call it.draw() instead of model->draw(it.getIndex()).
    int getIndex() const
    {
        return index;
    }

    /// Return the current iterator in the vector of indices, in case such a vector is currently used
    const VIterator& getVIterator() const
    {
        return it;
    }

    /// Return the end iterator in the vector of indices, in case such a vector is currently used
    const VIterator& getVIteratorEnd() const
    {
        return itend;
    }

    /// @}

protected:
    int index;      ///< index of the referenced element inside the CollisionModel.
    VIterator it; ///< current position in a vector of indices, in case this iterator traverse a non-contiguous set of indices
    VIterator itend; ///< end position in a vector of indices, in case this iterator traverse a non-contiguous set of indices
    static std::vector<int> SOFA_CORE_API emptyVector; ///< empty vector to be able to initialize the iterator to an empty pair
};

/**
 *  \brief Reference to an collision element defined by its <i>index</i> inside
 *  a given collision <i>model</i>.
 *
 *  A CollisionElementIterator is only a temporary iterator and must not
 *  contain any data. It only contains inline non-virtual methods calling the
 *  appropriate methods in the parent model object.
 *  This class is a template in order to store reference to a specific type of
 *  element (such as a Cube in a CubeModel).
 *
 */
template<class TModel>
class TCollisionElementIterator : public BaseCollisionElementIterator
{
public:
    typedef TModel Model;
    typedef std::vector<int>::const_iterator VIterator;

    /// Constructor.
    /// In most cases it will be used by the CollisionModel to
    /// create interators to its elements (such as in the begin() and end()
    /// methods).
    TCollisionElementIterator(Model* model=NULL, int index=0)
        : BaseCollisionElementIterator(index), model(model)
    {
    }

    /// Constructor.
    /// This constructor should be used in case a vector of indices is used.
    TCollisionElementIterator(Model* model, int index, VIterator it, VIterator itend)
        : BaseCollisionElementIterator(index, it, itend), model(model)
    {
    }

    /// Constructor.
    /// This constructor should be used in case a vector of indices is used.
    TCollisionElementIterator(Model* model, VIterator it, VIterator itend)
        : BaseCollisionElementIterator(it, itend), model(model)
    {
    }

    /// @name Iterator Interface
    /// @{

    /// Compare two iterators.
    /// Note that even it the iterators are of different types, they can point to the same element.
    template<class Model2>
    bool operator==(const TCollisionElementIterator<Model2>& i) const
    {
        return this->model == i.getCollisionModel() && this->index == i.getIndex();
    }

    /// Compare two iterators.
    /// Note that even it the iterators are of different types, they can point to the same element.
    template<class Model2>
    bool operator!=(const TCollisionElementIterator<Model2>& i) const
    {
        return this->model != i.getCollisionModel() || this->index != i.getIndex();
    }

    /// Test if this iterator is initialized with a valid CollisionModel.
    /// Note that it does not test if the referenced element inside the CollisionModel is valid.
    bool valid() const
    {
        return model!=NULL;
    }

    /// Return the CollisionModel containing the referenced element.
    Model* getCollisionModel() const
    {
        return model;
    }

    /// @}

    /// @name Wrapper methods to access data and methods inside the CollisionModel.
    /// @{

    /// Return the list (as a pair of iterators) of <i>internal children</i> of this element.
    ///
    /// @see CollisionModel::getInternalChildren
    std::pair<CollisionElementIterator,CollisionElementIterator> getInternalChildren() const;

    /// Return the list (as a pair of iterators) of <i>external children</i> of this element.
    ///
    /// @see CollisionModel::getExternalChildren
    std::pair<CollisionElementIterator,CollisionElementIterator> getExternalChildren() const;

    /// Test if this element is a leaf.
    ///
    /// @return true if the element(index) is leaf. i.e. If it is a primitive itself.
    bool isLeaf ( ) const
    {
        return model->isLeaf(index);
    }

    /// Test if this element can collide with another element.
    ///
    /// @see CollisionModel::canCollideWithElement
    bool canCollideWith(TCollisionElementIterator<Model>& elem)
    {
        return ((model != elem.model) || (index < elem.index)) && model->canCollideWithElement(index, elem.model, elem.index);
    }

    /// Distance to the actual (visual) surface
    double getProximity() { return model->getProximity(); }

    /// Contact stiffness
    double getContactStiffness() { return model->getContactStiffness(index); }

    /// Contact friction (damping) coefficient
    double getContactFriction() { return model->getContactFriction(index); }


    /// Render this element.
    ///
    /// @see CollisionModel::draw
    void draw(const core::visual::VisualParams* vparams)
    {
        model->draw(vparams,index);
    }
    /// @}

    Model* model;   ///< CollisionModel containing the referenced element.
};

/**
 *  \brief Reference to an abstract collision element.
 *
 *  You can think of a CollisionElementIterator as a glorified pointer to a
 *  collision element. It is only there to create a reference to it, not to
 *  actual contain its data. Classes derived from TCollisionElementIterator
 *  does not store any data, but just provide methods allowing to access the
 *  additionnal data stored inside the derived CollisionModel. For instance,
 *  the Cube class adds the minVect() / maxVect() methods to retrieve the
 *  corners of the cube, however this data is not stored inside Cube, instead
 *  it is stored inside the CubeData class within CubeModel.
 *
 */
class CollisionElementIterator : public TCollisionElementIterator<CollisionModel>
{
public:
    /// Constructor.
    /// In most cases it will be used by the CollisionModel to
    /// create interators to its elements (such as in the begin() and end()
    /// methods).
    CollisionElementIterator(CollisionModel* model=NULL, int index=0)
        : TCollisionElementIterator<CollisionModel>(model, index)
    {
    }

    /// Constructor.
    /// This constructor should be used in case a vector of indices is used.
    CollisionElementIterator(CollisionModel* model, VIterator it, VIterator itend)
        : TCollisionElementIterator<CollisionModel>(model, it, itend)
    {
    }

    /// Constructor.
    /// This constructor should be used in case a vector of indices is used.
    CollisionElementIterator(CollisionModel* model, int index, VIterator it, VIterator itend)
        : TCollisionElementIterator<CollisionModel>(model, index, it, itend)
    {
    }

    /// Automatic conversion from a reference to an element in a derived model.
    template<class DerivedModel>
    CollisionElementIterator(const TCollisionElementIterator<DerivedModel>& i)
        : TCollisionElementIterator<CollisionModel>(i.getCollisionModel(), i.getIndex(), i.getVIterator(), i.getVIteratorEnd())
    {
    }

    /// Automatic conversion from a reference to an element in a derived model.
    template<class DerivedModel>
    void operator=(const TCollisionElementIterator<DerivedModel>& i)
    {
        this->model = i.getCollisionModel();
        this->index = i.getIndex();
        this->it = i.getVIterator();
        this->itend = i.getVIteratorEnd();
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

} // namespace core

} // namespace sofa

#endif
