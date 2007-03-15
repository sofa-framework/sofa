/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COLLISIONELEMENT_H
#define SOFA_CORE_COLLISIONELEMENT_H

#include <vector>

namespace sofa
{

namespace core
{

class CollisionModel;
class CollisionElementIterator;

/// Reference to an collision element defined by its <i>index</i> inside a
/// given collision <i>model</i>.
///
/// A CollisionElementIterator is only a temporary iterator and must not
/// contain any data. It only contains inline non-virtual methods calling the
/// appropriate methods in the parent model object.
/// This class is a template in order to store reference to a specific type of
/// element (such as a Cube in a CubeModel).
///
template<class Model>
class TCollisionElementIterator
{
public:

    /// Constructor. In most cases it will be used by the CollisionModel to
    /// create interators to its elements (such as in the begin() and end()
    /// methods).
    /// @todo Should this be protected and only available to the CollisionModel?
    TCollisionElementIterator(Model* model=NULL, int index=0)
        : model(model), index(index)
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

    /// Increment this iterator to reference the next element.
    void operator++()
    {
        ++index;
    }

    /// Increment this iterator to reference the next element.
    void operator++(int)
    {
        ++index;
    }

    /// Return the CollisionModel containing the referenced element.
    Model* getCollisionModel() const
    {
        return model;
    }

    /// Return the index of the referenced element inside the CollisionModel.
    ///
    /// This methods should rarely be used.
    /// Users should call it.draw() instead of model->draw(it.getIndex()).
    int getIndex() const
    {
        return index;
    }

    /// Test if this iterator is initialized with a valid CollisionModel.
    /// Note that it does not test if the referenced element inside the CollisionModel is valid.
    bool valid() const
    {
        return model!=NULL;
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
        return model->canCollideWithElement(index, elem.model, elem.index);
    }

    /// Render this element.
    ///
    /// @see CollisionModel::draw
    void draw()
    {
        model->draw(index);
    }

    /// @}


protected:
    Model* model;   ///< CollisionModel containing the referenced element.
    int index;      ///< index of the referenced element inside the CollisionModel.


};

/// Reference to an abstract collision element.
///
/// You can think of a CollisionElementIterator as a glorified pointer to a
/// collision element. It is only there to create a reference to it, not to
/// actual contain its data. Classes derived from TCollisionElementIterator
/// does not store any data, but just provide methods allowing to access the
/// additionnal data stored inside the derived CollisionModel. For instance,
/// the Cube class adds the minVect() / maxVect() methods to retrieve the
/// corners of the cube, however this data is not stored inside Cube, instead
/// it is stored inside the CubeData class within CubeModel.
///
class CollisionElementIterator : public TCollisionElementIterator<CollisionModel>
{
public:
    /// Constructor. In most cases it will be used by the CollisionModel to
    /// create interators to its elements (such as in the begin() and end()
    /// methods).
    /// @todo Should this be protected and only available to the CollisionModel?
    CollisionElementIterator(CollisionModel* model=NULL, int index=0)
        : TCollisionElementIterator<CollisionModel>(model, index)
    {
    }

    /// Automatic conversion from a reference to an element in a derived model.
    template<class DerivedModel>
    CollisionElementIterator(const TCollisionElementIterator<DerivedModel>& i)
        : TCollisionElementIterator<CollisionModel>(i.getCollisionModel(), i.getIndex())
    {
    }

    /// Automatic conversion from a reference to an element in a derived model.
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

} // namespace core

} // namespace sofa

#endif
