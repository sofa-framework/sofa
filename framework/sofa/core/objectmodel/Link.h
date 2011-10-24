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
#ifndef SOFA_CORE_OBJECTMODEL_LINK_H
#define SOFA_CORE_OBJECTMODEL_LINK_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/core/ExecParams.h>
#include <sofa/helper/vector.h>
#include <string>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class DDGNode;

template<class TDestType, bool strongLink>
class LinkTraitsDestPtr;

template<class TDestType>
class LinkTraitsDestPtr<TDestType, false>
{
public:
    typedef TDestType* T;
    static TDestType* get(T p) { return p; }
};

template<class TDestType>
class LinkTraitsDestPtr<TDestType, true>
{
public:
    typedef typename TDestType::SPtr T;
    static TDestType* get(const T& p) { return p.get(); }
};

template<class TDestType, class TDestPtr, bool storePath>
class LinkTraitsValueType;

template<class TDestType, class TDestPtr>
class LinkTraitsValueType<TDestType,TDestPtr, false>
{
public:
    typedef TDestPtr T;
    std::string name(const T& ptr)
    {
        if (!ptr) return std::string();
        else return ptr->getName();
    }
};

template<class TDestType, class TDestPtr>
class LinkTraitsValueType<TDestType,TDestPtr, true>
{
public:
    struct T
    {
        TDestPtr ptr;
        std::string path;
        operator TDestPtr() const { return ptr; }
        TDestType* operator*() const { return &(*ptr); }
        TDestType* operator->() const { return &(*ptr); }
        bool operator == (TDestType* p) { return ptr == p; }
        bool operator != (TDestType* p) { return ptr != p; }
    };
    std::string name(const T& v)
    {
        if (v.path) return v.path;
        else if (!v.ptr) return std::string();
        else return v.ptr->getName();
    }
};

template<class TDestType, class TValueType, bool multiLink>
class LinkTraitsContainer;


/// Class to hold 0-or-1 pointer. The interface is similar to std::vector (size/[]/begin/end), plus an automatic convertion to one pointer.
template < class T, class TPtr = T* >
class SinglePtr
{
protected:
    TPtr elems[1];
public:
    typedef T pointed_type;
    typedef TPtr value_type;
    typedef value_type const * const_iterator;
    typedef value_type const * const_reverse_iterator;

    SinglePtr()
    {
        elems[0] = TPtr();
    }
    const_iterator begin() const
    {
        return elems;
    }
    const_iterator end() const
    {
        return (!elems[0])?elems:elems+1;
    }
    const_reverse_iterator rbegin() const
    {
        return begin();
    }
    const_reverse_iterator rend() const
    {
        return end();
    }
    unsigned int size() const
    {
        return (!elems[0])?0:1;
    }
    bool empty() const
    {
        return !elems[0];
    }
    void clear()
    {
        elems[0] = TPtr();
    }
    const TPtr& operator[](unsigned int i) const
    {
        return elems[i];
    }
    TPtr& operator[](unsigned int i)
    {
        return elems[i];
    }
    const TPtr& operator()(unsigned int i) const
    {
        return elems[i];
    }
    TPtr& operator()(unsigned int i)
    {
        return elems[i];
    }
    operator T*() const
    {
        return elems[0];
    }
    T* operator->() const
    {
        return elems[0];
    }
};

template<class TDestType, class TValueType>
class LinkTraitsContainer<TDestType, TValueType, false>
{
public:
    typedef SinglePtr<TDestType, TValueType> T;
    //typedef helper::fixed_array<TValueType,1> T;
    static void clear(T& c)
    {
        c.clear();
    }
    static unsigned int add(T& c, TDestType* v)
    {
        c(0) = v;
        return 0;
    }
    static unsigned int find(const T& c, TDestType* v)
    {
        if (c(0) == v) return 0;
        else return 1;
    }
    static void remove(T& c, unsigned index)
    {
        if (!index)
            c.clear();
    }
};

template<class TDestType, class TValueType>
class LinkTraitsContainer<TDestType, TValueType, true>
{
public:
    typedef helper::vector<TValueType> T;
    static void clear(T& c)
    {
        c.clear();
    }
    static unsigned int add(T& c, TValueType v)
    {
        unsigned int index = c.size();
        c.push_back(v);
        return index;
    }
    static unsigned int find(const T& c, TValueType v)
    {
        unsigned int s = c.size();
        for (unsigned int i=0; i<s; ++i)
            if (c[i] == v) return i;
        return s;
    }
    static void remove(T& c, unsigned index)
    {
        unsigned int s = c.size();
        for (unsigned int i=index+1; i < s; ++i)
            c[i-1] = c[i];
        c.resize(s-1);
    }
};

template<class Type>
class LinkTraitsPtrCasts
{
public:
    static sofa::core::objectmodel::Base* getBase(sofa::core::objectmodel::Base* b) { return b; }
    static sofa::core::objectmodel::Base* getBase(sofa::core::objectmodel::BaseData* d) { return d->getOwner(); }
    static sofa::core::objectmodel::BaseData* getData(sofa::core::objectmodel::Base* /*b*/) { return NULL; }
    static sofa::core::objectmodel::BaseData* getData(sofa::core::objectmodel::BaseData* d) { return d; }
};

template<>
class LinkTraitsPtrCasts<DDGNode>
{
public:
    static sofa::core::objectmodel::Base* getBase(sofa::core::objectmodel::DDGNode* n)
    {
        sofa::core::objectmodel::BaseData* d = dynamic_cast<sofa::core::objectmodel::BaseData*>(n);
        if (d) return d->getOwner();
        return dynamic_cast<sofa::core::objectmodel::Base*>(n);
    }

    static sofa::core::objectmodel::BaseData* getData(sofa::core::objectmodel::DDGNode* n)
    {
        return dynamic_cast<sofa::core::objectmodel::BaseData*>(n);
    }
};

/**
 *  \brief Container of all links in the scenegraph, from a given type of object (Owner) to another (Dest)
 *
 */
template<class TOwnerType, class TDestType, unsigned TFlags>
class Link : public BaseLink
{
public:
    typedef TOwnerType OwnerType;
    typedef TDestType DestType;
    enum { ActiveFlags = TFlags };
#define ACTIVEFLAG(f) ((ActiveFlags & (f)) != 0)
    typedef LinkTraitsDestPtr<DestType, ACTIVEFLAG(FLAG_STRONGLINK)> TraitsDestPtr;
    typedef typename TraitsDestPtr::T DestPtr;
    typedef LinkTraitsValueType<DestType, DestPtr, ACTIVEFLAG(FLAG_STOREPATH)> TraitsValueType;
    typedef typename TraitsValueType::T ValueType;
    typedef LinkTraitsContainer<DestType, ValueType, ACTIVEFLAG(FLAG_MULTILINK)> TraitsContainer;
    typedef typename TraitsContainer::T Container;
    typedef typename Container::const_iterator const_iterator;
    typedef typename Container::const_reverse_iterator const_reverse_iterator;
    //typedef LinkTraitsValidatorFn<OwnerType, DestPtr, ACTIVEFLAG(FLAG_MULTILINK)> TraitsValidatorFn;
    typedef void (OwnerType::*ValidatorFn)(DestPtr, DestPtr&);
    typedef void (OwnerType::*ValidatorIndexFn)(DestPtr, DestPtr&, unsigned int);
    typedef LinkTraitsPtrCasts<TOwnerType> TraitsOwnerCasts;
    typedef LinkTraitsPtrCasts<TDestType> TraitsDestCasts;
#undef ACTIVEFLAG

    Link(const InitLink<OwnerType>& init)
        : BaseLink(init, ActiveFlags), m_owner(init.owner), m_validator(NULL), m_validatorIndex(NULL)
    {
    }

    virtual ~Link()
    {
    }

    void setValidator(ValidatorFn fn)
    {
        m_validator = fn;
        m_validatorIndex = NULL;
    }

    void setValidator(ValidatorIndexFn fn)
    {
        m_validator = NULL;
        m_validatorIndex = fn;
    }

    unsigned int size() const
    {
        return (unsigned int)m_value[core::ExecParams::currentAspect()].size();
    }

    bool empty() const
    {
        return m_value[core::ExecParams::currentAspect()].empty();
    }

    DestType* get(unsigned int index=0) const
    {
        const int aspect = core::ExecParams::currentAspect();
        if (index < m_value[aspect].size())
            return TraitsDestPtr::get(m_value[aspect][index]);
        else
            return NULL;
    }

    const Container& getValue() const
    {
        return m_value[core::ExecParams::currentAspect()];
    }

    const_iterator begin() const
    {
        return m_value[core::ExecParams::currentAspect()].begin();
    }

    const_iterator end() const
    {
        return m_value[core::ExecParams::currentAspect()].end();
    }

    const_reverse_iterator rbegin() const
    {
        return m_value[core::ExecParams::currentAspect()].rbegin();
    }

    const_reverse_iterator rend() const
    {
        return m_value[core::ExecParams::currentAspect()].rend();
    }

    void reset(unsigned int index=0)
    {
        change(NULL, index);
    }

    void set(DestPtr v, unsigned int index=0)
    {
        change(v, index);
    }

    bool add(DestPtr v)
    {
        if (!v) return false;
        const int aspect = core::ExecParams::currentAspect();
        unsigned int index = TraitsContainer::add(m_value[aspect],v);
        changed(DestPtr(), m_value[aspect][index], index);
        return true;
    }

    bool remove(DestPtr v)
    {
        if (!v) return false;
        const int aspect = core::ExecParams::currentAspect();
        unsigned int index = TraitsContainer::find(m_value[aspect],v);
        if (index >= m_value[aspect].size()) return false;
        m_value[aspect][index] = NULL;
        changed(v, m_value[aspect][index], index);
        TraitsContainer::remove(m_value[aspect],index);
        return true;
    }

    unsigned int getSize() const
    {
        return size();
    }

    Base* getLinkedBase(unsigned int index=0) const
    {
        return TraitsDestCasts::getBase(get(index));
    }
    BaseData* getLinkedData(unsigned int index=0) const
    {
        return TraitsDestCasts::getData(get(index));
    }

    /// Copy the value of an aspect into another one.
    virtual void copyAspect(int destAspect, int srcAspect)
    {
        m_value[destAspect] = m_value[srcAspect];
    }

    /// Release memory allocated for the specified aspect.
    virtual void releaseAspect(int aspect)
    {
        TraitsContainer::clear(m_value[aspect]);
    }

    sofa::core::objectmodel::Base* getOwnerBase() const
    {
        return TraitsOwnerCasts::getBase(m_owner);
    }
    sofa::core::objectmodel::BaseData* getOwnerData() const
    {
        return TraitsOwnerCasts::getData(m_owner);
    }

protected:
    OwnerType* m_owner;
    ValidatorFn m_validator;
    ValidatorIndexFn m_validatorIndex;
    helper::fixed_array<Container, SOFA_DATA_MAX_ASPECTS> m_value;

    void changed(DestPtr before, DestPtr& after, unsigned int index)
    {
        if (m_validator)
            (m_owner->*m_validator)(before, after);
        else if (m_validatorIndex)
            (m_owner->*m_validatorIndex)(before, after, index);
    }

    void change(DestPtr v, unsigned int index)
    {
        DestPtr& val = m_value[core::ExecParams::currentAspect()][index];
        DestPtr before = val;
        val = v;
        changed(before, val, index);
    }
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
