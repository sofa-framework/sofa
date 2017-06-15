/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_CORE_OBJECTMODEL_LINK_H
#define SOFA_CORE_OBJECTMODEL_LINK_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/core/ExecParams.h>
#include <sofa/helper/stable_vector.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

template<class TDestType, class TDestPtr, bool strongLink, bool storePath>
class LinkTraitsValueType;

template<class TDestType, class TDestPtr, bool strongLink>
class LinkTraitsValueType<TDestType,TDestPtr,strongLink, false>
{
public:
    typedef TDestPtr T;
    static bool path(const T& /*ptr*/, std::string& /*str*/)
    {
        return false;
    }
    static const TDestPtr& get(const T& v) { return v; }
    static void set(T& v, const TDestPtr& ptr) { v = ptr; }
    static void setPath(T& /*ptr*/, const std::string& /*name*/) {}
};

template<class TDestType, class TDestPtr, bool strongLink>
class LinkTraitsValueType<TDestType,TDestPtr,strongLink, true>
{
public:
    typedef LinkTraitsDestPtr<TDestType, strongLink> TraitsDestPtr;

    struct T
    {
        TDestPtr ptr;
        std::string path;
        T() : ptr(TDestPtr()) {}
        explicit T(const TDestPtr& p) : ptr(p) {}
        operator TDestType*() const { return TraitsDestPtr::get(ptr); }
        void operator=(const TDestPtr& v) { if (v != ptr) { ptr = v; path.clear(); } }
        TDestType& operator*() const { return *ptr; }
        TDestType* operator->() const { return TraitsDestPtr::get(ptr); }
        TDestType* get() const { return TraitsDestPtr::get(ptr); }
        bool operator!() const { return !ptr; }
        bool operator==(const TDestPtr& p) const { return ptr == p; }
        bool operator!=(const TDestPtr& p) const { return ptr != p; }
    };
    static bool path(const T& v, std::string& str)
    {
        if (v.path.empty()) return false;
        str = v.path;
        return true;
    }
    static const TDestPtr& get(const T& v) { return v.ptr; }
    static void set(T& v, const TDestPtr& ptr) { if (v.ptr && ptr != v.ptr) v.path.clear(); v.ptr = ptr; }
    static void setPath(T& v, const std::string& name) { v.path = name; }
};

template<class TDestType, class TDestPtr, class TValueType, bool multiLink>
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
    const_iterator cbegin() const
    {
        return begin();
    }
    const_iterator cend() const
    {
        return end();
    }
    const_reverse_iterator crbegin() const
    {
        return rbegin();
    }
    const_reverse_iterator crend() const
    {
        return rend();
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
    const TPtr& get() const
    {
        return elems[0];
    }
    TPtr& get()
    {
        return elems[0];
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

template<class TDestType, class TDestPtr, class TValueType>
class LinkTraitsContainer<TDestType, TDestPtr, TValueType, false>
{
public:
    typedef SinglePtr<TDestType, TValueType> T;
    //typedef helper::fixed_array<TValueType,1> T;
    static void clear(T& c)
    {
        c.clear();
    }
    static unsigned int add(T& c, TDestPtr v)
    {
        c.get() = v;
        return 0;
    }
    static unsigned int find(const T& c, TDestPtr v)
    {
        if (c.get() == v) return 0;
        else return 1;
    }
    static void remove(T& c, unsigned index)
    {
        if (!index)
            c.clear();
    }
};

template<class TDestType, class TDestPtr, class TValueType>
class LinkTraitsContainer<TDestType, TDestPtr, TValueType, true>
{
public:
    /// Container type.
    /// We use stable_vector to allow insertion/removal of elements
    /// while iterators are used (required to add/remove objects
    /// while visitors are in progress).
    typedef sofa::helper::stable_vector<TValueType> T;
    static void clear(T& c)
    {
        c.clear();
    }
    static unsigned int add(T& c, TDestPtr v)
    {
        unsigned int index = static_cast<unsigned int>(c.size());
        c.push_back(TValueType(v));
        return index;
    }
    static unsigned int find(const T& c, TDestPtr v)
    {
        size_t s = c.size();
        for (size_t i=0; i<s; ++i)
            if (c[i] == v) return static_cast<unsigned int>(i);
        return static_cast<unsigned int>(s);
    }
    static void remove(T& c, unsigned index)
    {
        c.erase( c.begin()+index );
    }
};

template<class OwnerType, class DestType, bool data>
class LinkTraitsFindDest;

template<class OwnerType, class DestType>
class LinkTraitsFindDest<OwnerType, DestType, false>
{
public:
    static bool findLinkDest(OwnerType* owner, DestType*& ptr, const std::string& path, const BaseLink* link)
    {
        return owner->findLinkDest(ptr, path, link);
    }
    template<class TContext>
    static bool checkPath(const std::string& path, TContext* context)
    {
        DestType* ptr = NULL;
        return context->findLinkDest(ptr, path, NULL);
    }
};

template<class OwnerType, class DestType>
class LinkTraitsFindDest<OwnerType, DestType, true>
{
public:
    static bool findLinkDest(OwnerType* owner, DestType*& ptr, const std::string& path, const BaseLink* link)
    {
        return owner->findDataLinkDest(ptr, path, link);
    }
    template<class TContext>
    static bool checkPath(const std::string& path, TContext* context)
    {
        DestType* ptr = NULL;
        return context->findDataLinkDest(ptr, path, NULL);
    }
};

template<class Type>
class LinkTraitsPtrCasts;

/**
 *  \brief Container of all links in the scenegraph, from a given type of object (Owner) to another (Dest)
 *
 */
template<class TOwnerType, class TDestType, unsigned TFlags>
class TLink : public BaseLink
{
public:
    typedef TOwnerType OwnerType;
    typedef TDestType DestType;
    enum { ActiveFlags = TFlags };
#define ACTIVEFLAG(f) ((ActiveFlags & (f)) != 0)
    typedef LinkTraitsDestPtr<DestType, ACTIVEFLAG(FLAG_STRONGLINK)> TraitsDestPtr;
    typedef typename TraitsDestPtr::T DestPtr;
    typedef LinkTraitsValueType<DestType, DestPtr, ACTIVEFLAG(FLAG_STRONGLINK), ACTIVEFLAG(FLAG_STOREPATH)> TraitsValueType;
    typedef typename TraitsValueType::T ValueType;
    typedef LinkTraitsContainer<DestType, DestPtr, ValueType, ACTIVEFLAG(FLAG_MULTILINK)> TraitsContainer;
    typedef typename TraitsContainer::T Container;
    typedef typename Container::const_iterator const_iterator;
    typedef typename Container::const_reverse_iterator const_reverse_iterator;
    typedef LinkTraitsFindDest<OwnerType, DestType, ACTIVEFLAG(FLAG_DATALINK)> TraitsFindDest;
    typedef LinkTraitsPtrCasts<TOwnerType> TraitsOwnerCasts;
    typedef LinkTraitsPtrCasts<TDestType> TraitsDestCasts;
#undef ACTIVEFLAG

    TLink()
        : BaseLink(ActiveFlags)
    {
    }

    TLink(const InitLink<OwnerType>& init)
        : BaseLink(init, ActiveFlags), m_owner(init.owner)
    {
        if (m_owner) m_owner->addLink(this);
    }

    virtual ~TLink()
    {
    }

    size_t size(const core::ExecParams* params = 0) const
    {
        return (size_t)m_value[core::ExecParams::currentAspect(params)].size();
    }

    bool empty(const core::ExecParams* params = 0) const
    {
        return m_value[core::ExecParams::currentAspect(params)].empty();
    }

    const Container& getValue(const core::ExecParams* params = 0) const
    {
        return m_value[core::ExecParams::currentAspect(params)];
    }

    const_iterator begin(const core::ExecParams* params = 0) const
    {
        return m_value[core::ExecParams::currentAspect(params)].cbegin();
    }

    const_iterator end(const core::ExecParams* params = 0) const
    {
        return m_value[core::ExecParams::currentAspect(params)].cend();
    }

    const_reverse_iterator rbegin(const core::ExecParams* params = 0) const
    {
        return m_value[core::ExecParams::currentAspect(params)].crbegin();
    }

    const_reverse_iterator rend(const core::ExecParams* params = 0) const
    {
        return m_value[core::ExecParams::currentAspect(params)].crend();
    }

    bool add(DestPtr v)
    {
        if (!v) return false;
        const int aspect = core::ExecParams::currentAspect();
        unsigned int index = TraitsContainer::add(m_value[aspect],v);
        this->updateCounter(aspect);
        added(v, index);
        return true;
    }

    bool add(DestPtr v, const std::string& path)
    {
        if (!v && path.empty()) return false;
        const int aspect = core::ExecParams::currentAspect();
        unsigned int index = TraitsContainer::add(m_value[aspect],v);
        TraitsValueType::setPath(m_value[aspect][index],path);
        this->updateCounter(aspect);
        added(v, index);
        return true;
    }

    bool addPath(const std::string& path)
    {
        if (path.empty()) return false;
        DestType* ptr = NULL;
        if (m_owner)
            TraitsFindDest::findLinkDest(m_owner, ptr, path, this);
        return add(ptr, path);
    }

    bool remove(DestPtr v)
    {
        if (!v) return false;
        const int aspect = core::ExecParams::currentAspect();
        unsigned int index = TraitsContainer::find(m_value[aspect],v);
        if (index >= m_value[aspect].size()) return false;
        TraitsContainer::remove(m_value[aspect],index);
        this->updateCounter(aspect);
        removed(v, index);
        return true;
    }

    bool removePath(const std::string& path)
    {
        if (path.empty()) return false;
        const int aspect = core::ExecParams::currentAspect();
        unsigned int n = m_value[aspect].size();
        for (unsigned int index=0; index<n; ++index)
        {
            std::string p = getPath(index);
            if (p == path)
            {
                DestPtr v = m_value[aspect][index];
                TraitsContainer::remove(m_value[aspect],index);
                this->updateCounter(aspect);
                removed(v, index);
                return true;
            }
        }
        return false;
    }

    const BaseClass* getDestClass() const
    {
        return DestType::GetClass();
    }

    const BaseClass* getOwnerClass() const
    {
        return OwnerType::GetClass();
    }

    size_t getSize() const
    {
        return size();
    }

    std::string getPath(unsigned int index) const
    {
        const int aspect = core::ExecParams::currentAspect();
        if (index >= m_value[aspect].size())
            return std::string();
        std::string path;
        const ValueType& value = m_value[aspect][index];
        if (!TraitsValueType::path(value, path))
        {
            DestType* ptr = TraitsDestPtr::get(TraitsValueType::get(value));
            if (ptr)
                path = BaseLink::CreateString(TraitsDestCasts::getBase(ptr), TraitsDestCasts::getData(ptr),
                        TraitsOwnerCasts::getBase(m_owner));
        }
        return path;
    }

    Base* getLinkedBase(unsigned int index=0) const
    {
        return TraitsDestCasts::getBase(getIndex(index));
    }
    BaseData* getLinkedData(unsigned int index=0) const
    {
        return TraitsDestCasts::getData(getIndex(index));
    }
    std::string getLinkedPath(unsigned int index=0) const
    {
        return getPath(index);
    }

    /// @name Serialization API
    /// @{

    /// Read the command line
    virtual bool read( const std::string& str )
    {
        if (str.empty())
            return true;

        bool ok = true;

        // Allows spaces in links values for single links
        if (!getFlag(BaseLink::FLAG_MULTILINK))
        {
            DestType* ptr = NULL;

            if (m_owner && !TraitsFindDest::findLinkDest(m_owner, ptr, str, this))
            {
                // This is not an error, as the destination can be added later in the graph
                // instead, we will check for failed links after init is completed
                //ok = false;
            }
            else if (str[0] != '@')
            {
                ok = false;
            }

            add(ptr, str);
        }
        else
        {
            Container& container = m_value[core::ExecParams::currentAspect()];
            std::istringstream istr(str.c_str());
            std::string path;

            // Find the target of each path, and store those targets in
            // a temporary vector of (pointer, path) pairs
            typedef std::vector< std::pair<DestPtr, std::string> > PairVector;
            PairVector newList;
            while (istr >> path)
            {
                DestType *ptr = NULL;
                if (m_owner && !TraitsFindDest::findLinkDest(m_owner, ptr, path, this))
                {
                    // This is not an error, as the destination can be added later in the graph
                    // instead, we will check for failed links after init is completed
                    //ok = false;
                }
                else if (path[0] != '@')
                {
                    ok = false;
                }
                newList.push_back(std::make_pair(ptr, path));
            }

            // Add the objects that are not already present to the container of this Link
            for (typename PairVector::iterator i = newList.begin(); i != newList.end(); i++)
            {
                const DestPtr ptr = i->first;
                const std::string& path = i->second;

                if (TraitsContainer::find(container, ptr) == container.size()) // Not found
                    add(ptr, path);
            }

            // Remove the objects from the container that are not in the new list
            for (size_t i = 0; i != container.size(); i++)
            {
                DestPtr dest(container[i]);
                bool destFound = false;
                typename PairVector::iterator j = newList.begin();
                while (j != newList.end() && !destFound)
                {
                    if (j->first == dest)
                        destFound = true;
                    j++;
                }

                if (!destFound)
                    remove(dest);
            }
        }

        return ok;
    }


    /// Check that a given path is valid, that the pointed object exists and is of the right type
    template <class TContext>
    static bool CheckPath( const std::string& path, TContext* context)
    {
        if (path.empty())
            return false;
        if (!context)
        {
            std::string p,d;
            return BaseLink::ParseString( path, &p, (ActiveFlags & FLAG_DATALINK) ? &d : NULL, NULL);
        }
        else
        {
            return TraitsFindDest::checkPath(path, context);
        }
    }

    /// @}

    /// Copy the value of an aspect into another one.
    virtual void copyAspect(int destAspect, int srcAspect)
    {
        BaseLink::copyAspect(destAspect, srcAspect);
        m_value[destAspect] = m_value[srcAspect];
    }

    /// Release memory allocated for the specified aspect.
    virtual void releaseAspect(int aspect)
    {
        BaseLink::releaseAspect(aspect);
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

    void setOwner(OwnerType* owner)
    {
        m_owner = owner;
        m_owner->addLink(this);
    }

protected:
    OwnerType* m_owner;
    helper::fixed_array<Container, SOFA_DATA_MAX_ASPECTS> m_value;

    DestType* getIndex(unsigned int index) const
    {
        const int aspect = core::ExecParams::currentAspect();
        if (index < m_value[aspect].size())
            return TraitsDestPtr::get(TraitsValueType::get(m_value[aspect][index]));
        else
            return NULL;
    }

    virtual void added(DestPtr ptr, unsigned int index) = 0;
    virtual void removed(DestPtr ptr, unsigned int index) = 0;
};

/**
 *  \brief Container of vectors of links in the scenegraph, from a given type of object (Owner) to another (Dest)
 *
 */
template<class TOwnerType, class TDestType, unsigned TFlags>
class MultiLink : public TLink<TOwnerType,TDestType,TFlags|BaseLink::FLAG_MULTILINK>
{
public:
    typedef TLink<TOwnerType,TDestType,TFlags|BaseLink::FLAG_MULTILINK> Inherit;
    typedef TOwnerType OwnerType;
    typedef TDestType DestType;
    typedef typename Inherit::TraitsDestPtr TraitsDestPtr;
    typedef typename Inherit::DestPtr DestPtr;
    typedef typename Inherit::TraitsValueType TraitsValueType;
    typedef typename Inherit::ValueType ValueType;
    typedef typename Inherit::TraitsContainer TraitsContainer;
    typedef typename Inherit::Container Container;
    typedef typename Inherit::TraitsOwnerCasts TraitsOwnerCasts;
    typedef typename Inherit::TraitsDestCasts TraitsDestCasts;
    typedef typename Inherit::TraitsFindDest TraitsFindDest;

    typedef void (OwnerType::*ValidatorFn)(DestPtr v, unsigned int index, bool add);

    MultiLink(const BaseLink::InitLink<OwnerType>& init)
        : Inherit(init), m_validator(NULL)
    {
    }

    MultiLink(const BaseLink::InitLink<OwnerType>& init, DestPtr val)
        : Inherit(init), m_validator(NULL)
    {
        if (val) this->add(val);
    }

    virtual ~MultiLink()
    {
    }

    void setValidator(ValidatorFn fn)
    {
        m_validator = fn;
    }

    /// Check that a given list of path is valid, that the pointed object exists and is of the right type
    template<class TContext>
    static bool CheckPaths( const std::string& str, TContext* context)
    {
        if (str.empty())
            return false;
        std::istringstream istr( str.c_str() );
        std::string path;
        bool ok = true;
        while (istr >> path)
        {
            ok &= TLink<TOwnerType,TDestType,TFlags|BaseLink::FLAG_MULTILINK>::CheckPath(path, context);
        }
        return ok;
    }

    /// Update pointers in case the pointed-to objects have appeared
    /// @return false if there are broken links
    virtual bool updateLinks()
    {
        if (!this->m_owner) return false;
        bool ok = true;
        const int aspect = core::ExecParams::currentAspect();
        std::size_t n = this->size();
		for (std::size_t i = 0; i<n; ++i)
        {
            ValueType& value = this->m_value[aspect][i];
            std::string path;
            if (TraitsValueType::path(value, path))
            {
                DestType* ptr = TraitsDestPtr::get(TraitsValueType::get(value));
                if (!ptr)
                {
                    TraitsFindDest::findLinkDest(this->m_owner, ptr, path, this);
                    if (ptr)
                    {
                        DestPtr v = ptr;
                        TraitsValueType::set(value,v);
                        this->updateCounter(aspect);
                        this->added(v, i);
                    }
                    else
                    {
                        ok = false;
                    }
                }
            }
        }
        return ok;
    }

    DestType* get(unsigned int index, const core::ExecParams* params = 0) const
    {
        const int aspect = core::ExecParams::currentAspect(params);
        if (index < this->m_value[aspect].size())
            return TraitsDestPtr::get(TraitsValueType::get(this->m_value[aspect][index]));
        else
            return NULL;
    }

    DestType* operator[](unsigned int index) const
    {
        return get(index);
    }

protected:
    ValidatorFn m_validator;

    void added(DestPtr val, unsigned int index)
    {
        if (m_validator)
            (this->m_owner->*m_validator)(val, index, true);
    }

    void removed(DestPtr val, unsigned int index)
    {
        if (m_validator)
            (this->m_owner->*m_validator)(val, index, false);
    }
};

/**
 *  \brief Container of single links in the scenegraph, from a given type of object (Owner) to another (Dest)
 *
 */
template<class TOwnerType, class TDestType, unsigned TFlags>
class SingleLink : public TLink<TOwnerType,TDestType,TFlags&~BaseLink::FLAG_MULTILINK>
{
public:
    typedef TLink<TOwnerType,TDestType,TFlags&~BaseLink::FLAG_MULTILINK> Inherit;
    typedef TOwnerType OwnerType;
    typedef TDestType DestType;
    typedef typename Inherit::TraitsDestPtr TraitsDestPtr;
    typedef typename Inherit::DestPtr DestPtr;
    typedef typename Inherit::TraitsValueType TraitsValueType;
    typedef typename Inherit::ValueType ValueType;
    typedef typename Inherit::TraitsContainer TraitsContainer;
    typedef typename Inherit::Container Container;
    typedef typename Inherit::TraitsOwnerCasts TraitsOwnerCasts;
    typedef typename Inherit::TraitsDestCasts TraitsDestCasts;
    typedef typename Inherit::TraitsFindDest TraitsFindDest;

    typedef void (OwnerType::*ValidatorFn)(DestPtr before, DestPtr& after);

    SingleLink()
        : m_validator(NULL)
    {
    }

    SingleLink(const BaseLink::InitLink<OwnerType>& init)
        : Inherit(init), m_validator(NULL)
    {
    }

    SingleLink(const BaseLink::InitLink<OwnerType>& init, DestPtr val)
        : Inherit(init), m_validator(NULL)
    {
        if (val) this->add(val);
    }

    virtual ~SingleLink()
    {
    }

    void setValidator(ValidatorFn fn)
    {
        m_validator = fn;
    }

    std::string getPath() const
    {
        return Inherit::getPath(0);
    }

    DestType* get(const core::ExecParams* params = 0) const
    {
        const int aspect = core::ExecParams::currentAspect(params);
        return TraitsDestPtr::get(TraitsValueType::get(this->m_value[aspect].get()));
    }

    void reset()
    {
        const int aspect = core::ExecParams::currentAspect();
        ValueType& value = this->m_value[aspect].get();
        const DestPtr before = TraitsValueType::get(value);
        if (!before) return;
        TraitsValueType::set(value, NULL);
        this->updateCounter(aspect);
        changed(before, NULL);
    }

    void set(DestPtr v)
    {
        const int aspect = core::ExecParams::currentAspect();
        ValueType& value = this->m_value[aspect].get();
        const DestPtr before = TraitsValueType::get(value);
        if (v == before) return;
        TraitsValueType::set(value, v);
        this->updateCounter(aspect);
        changed(before, v);
    }

    void set(DestPtr v, const std::string& path)
    {
        const int aspect = core::ExecParams::currentAspect();
        ValueType& value = this->m_value[aspect].get();
        const DestPtr before = TraitsValueType::get(value);
        if (v != before)
            TraitsValueType::set(value, v);
        TraitsValueType::setPath(value, path);
        this->updateCounter(aspect);
        if (v != before)
            changed(before, v);
    }

    void setPath(const std::string& path)
    {
        if (path.empty()) { reset(); return; }
        DestType* ptr = NULL;
        if (this->m_owner)
            TraitsFindDest::findLinkDest(this->m_owner, ptr, path, this);
        set(ptr, path);
    }

    /// Update pointers in case the pointed-to objects have appeared
    /// @return false if there are broken links
    virtual bool updateLinks()
    {
        if (!this->m_owner) return false;
        bool ok = true;
        const int aspect = core::ExecParams::currentAspect();
        ValueType& value = this->m_value[aspect].get();
        std::string path;
        if (TraitsValueType::path(value, path))
        {
            DestType* ptr = TraitsDestPtr::get(TraitsValueType::get(value));
            if (!ptr)
            {
                TraitsFindDest::findLinkDest(this->m_owner, ptr, path, this);
                if (ptr)
                {
                    set(ptr, path);
                }
                else
                {
                    ok = false;
                }
            }
        }
        return ok;
    }

#ifndef SOFA_MAYBE_DEPRECATED
    // Convenient operators to make a SingleLink appear as a regular pointer
    operator DestType*() const
    {
        return get();
    }
    DestType* operator->() const
    {
        return get();
    }
    DestType& operator*() const
    {
        return *get();
    }

    DestPtr operator=(DestPtr v)
    {
        set(v);
        return v;
    }
#endif

protected:
    ValidatorFn m_validator;


    void added(DestPtr val, unsigned int /*index*/)
    {
        if (m_validator)
        {
            DestPtr after = val;
            (this->m_owner->*m_validator)(NULL, after);
            if (after != val)
                TraitsValueType::set(this->m_value[core::ExecParams::currentAspect()].get(), after);
        }
    }

    void removed(DestPtr val, unsigned int /*index*/)
    {
        if (m_validator)
        {
            DestPtr after = NULL;
            (this->m_owner->*m_validator)(val, after);
            if (after)
                TraitsValueType::set(this->m_value[core::ExecParams::currentAspect()].get(), after);
        }
    }

    void changed(DestPtr before, DestPtr val)
    {
        if (m_validator)
        {
            DestPtr after = val;
            (this->m_owner->*m_validator)(before, after);
            if (after != val)
                TraitsValueType::set(this->m_value[core::ExecParams::currentAspect()].get(), after);
        }
    }
};

} // namespace objectmodel

} // namespace core

// the SingleLink class is used everywhere
using core::objectmodel::SingleLink;

// the MultiLink class is used everywhere
using core::objectmodel::MultiLink;

} // namespace sofa

#endif
