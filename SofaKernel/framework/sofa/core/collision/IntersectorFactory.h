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
#ifndef SOFA_CORE_COLLISION_INTERSECTORFACTORY_H
#define SOFA_CORE_COLLISION_INTERSECTORFACTORY_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/helper/FnDispatcher.h>

namespace sofa
{

namespace core
{

namespace collision
{

template<class TIntersectionClass>
class BaseIntersectorCreator
{
public:
    virtual ~BaseIntersectorCreator() {}

    virtual void addIntersectors(TIntersectionClass* object) = 0;

    virtual std::string name() const = 0;
};

template<class TIntersectionClass>
class IntersectorFactory
{
protected:
    typedef BaseIntersectorCreator<TIntersectionClass> Creator;
    typedef std::vector<Creator*> CreatorVector;
    CreatorVector creatorVector;

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
            creator->addIntersectors(object);
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

    virtual void addIntersectors(TIntersectionClass* object)
    {
        new TIntersectorClass(object);
    }

    virtual std::string name() const { return m_name; }
protected:
    std::string m_name;
};

} // namespace collision

} // namespace core

} // namespace sofa

#endif
