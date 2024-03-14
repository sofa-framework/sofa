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

#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <typeindex>
#include <typeinfo>

namespace sofa::core::collision
{

template<class TIntersectionClass>
class BaseIntersectorCreator
{
public:
    virtual ~BaseIntersectorCreator() {}

    virtual std::tuple<std::type_index, std::shared_ptr<void>> addIntersectors(TIntersectionClass* object) = 0;

    virtual std::string name() const = 0;
};

template<class TIntersectionClass>
class IntersectorFactory
{
protected:
    typedef BaseIntersectorCreator<TIntersectionClass> Creator;
    typedef std::vector<Creator*> CreatorVector;
    CreatorVector creatorVector;
    // keep track of already created TIntersectorClass instances for each template combination
    // when the factory is destroyed, the refcount of the instances is zero and they are cleaned up correctly
    std::unordered_map<std::type_index, std::shared_ptr<void>> intersectorCache;

public:

    bool registerCreator(Creator* creator)
    {
        this->creatorVector.push_back(creator);
        return true;
    }

    void addIntersectors(TIntersectionClass* object)
    {
        typename CreatorVector::iterator it = creatorVector.begin();
        typename CreatorVector::iterator end = creatorVector.end();
        while (it != end)
        {
            BaseIntersectorCreator<TIntersectionClass>* creator = (*it);
            std::tuple<std::type_index, std::shared_ptr<void>> intersectorHandleInfo = creator->addIntersectors(object);
            // add the specific TIntersectorClass and the respective pointer to the map
            // if an old one of the same type is replaced, the old one will be cleaned up because its refcount is reduced to zero.
            intersectorCache[std::get<0>(intersectorHandleInfo)] = std::get<1>(intersectorHandleInfo);
            ++it;
        }
    }

    static IntersectorFactory<TIntersectionClass>* getInstance()
    {
        static IntersectorFactory<TIntersectionClass> instance;
        return &instance;
    }
};

template<class TIntersectionClass, class TIntersectorClass>
class IntersectorCreator : public BaseIntersectorCreator<TIntersectionClass>
{
public:
    IntersectorCreator(std::string name) : m_name(name)
    {
        IntersectorFactory<TIntersectionClass>::getInstance()->registerCreator(this);
    }
    virtual ~IntersectorCreator() {}

    virtual std::tuple<std::type_index, std::shared_ptr<void>> addIntersectors(TIntersectionClass* object)
    {
        return std::make_tuple(std::type_index(typeid(TIntersectorClass)), std::make_shared<TIntersectorClass>(object));
    }

    virtual std::string name() const { return m_name; }
protected:
    std::string m_name;
};
} // namespace sofa::core::collision
