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

#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/type/stable_vector.h>
#include <sofa/core/PathResolver.h>
#include <sofa/core/sptr.h>
#include <sofa/core/fwd.h>

#include <functional>

namespace sofa
{
namespace core::objectmodel
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
    typedef typename sofa::core::sptr<TDestType> T;
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
template < class T, class TDestPtr, class TPtr = T* >
class SinglePtr
{
protected:
    bool isEmpty {true};
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
        return (isEmpty)?elems:elems+1;
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
    std::size_t size() const
    {
        return !isEmpty;
    }
    void resize(size_t size)
    {
        if(size == 1)
            isEmpty=false;
        else
            isEmpty=true;
    }
    bool empty() const
    {
        return isEmpty;
    }
    void clear()
    {
        isEmpty = true;
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
    void add(TDestPtr v)
    {
        isEmpty = false;
        elems[0] = v;
    }
    void addBegin(TDestPtr v)
    {
        isEmpty = false;
        elems[0] = v;
    }
    const TPtr& operator[](std::size_t i) const
    {
        return elems[i];
    }
    TPtr& operator[](std::size_t i)
    {
        return elems[i];
    }
    const TPtr& operator()(std::size_t i) const
    {
        return elems[i];
    }
    TPtr& operator()(std::size_t i)
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
    typedef SinglePtr<TDestType, TDestPtr, TValueType> T;
    static void resize(T& c, size_t newsize)
    {
        c.resize(newsize);
    }
    static void clear(T& c)
    {
        c.clear();
    }
    static std::size_t add(T& c, TDestPtr v)
    {
        c.add(v);
        return 0;
    }
    static std::size_t find(const T& c, TDestPtr v)
    {
        if (c.get() == v) return 0;
        else return 1;
    }
    static void remove(T& c, std::size_t index)
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
    typedef sofa::type::stable_vector<TValueType> T;
    static void clear(T& c)
    {
        c.clear();
    }
    static std::size_t add(T& c, TDestPtr v)
    {
        const std::size_t index = c.size();
        c.push_back(TValueType(v));
        return index;
    }
    static std::size_t addBegin(T& c, TDestPtr v)
    {
        c.insert(c.begin(), TValueType(v));
        return 0;
    }
    static std::size_t find(const T& c, TDestPtr v)
    {
        const size_t s = c.size();
        for (size_t i=0; i<s; ++i)
            if (c[i] == v) return i;
        return s;
    }
    static void remove(T& c, std::size_t index)
    {
        c.erase( c.begin()+index );
    }
};

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
    static constexpr unsigned ActiveFlags = TFlags;
    static constexpr bool IsStrongLink = (ActiveFlags & FLAG_STRONGLINK) != 0;
    static constexpr bool IsMultiLink = (ActiveFlags & FLAG_MULTILINK) != 0;
    static constexpr bool StorePath = (ActiveFlags & FLAG_STOREPATH) != 0;

    typedef LinkTraitsDestPtr<DestType, IsStrongLink> TraitsDestPtr;
    typedef typename TraitsDestPtr::T DestPtr;
    typedef LinkTraitsValueType<DestType, DestPtr, IsStrongLink, StorePath> TraitsValueType;
    typedef typename TraitsValueType::T ValueType;
    typedef LinkTraitsContainer<DestType, DestPtr, ValueType, IsMultiLink> TraitsContainer;
    typedef typename TraitsContainer::T Container;
    typedef typename Container::const_iterator const_iterator;
    typedef typename Container::const_reverse_iterator const_reverse_iterator;

    TLink()
        : BaseLink(ActiveFlags)
    {
    }

    TLink(const InitLink<OwnerType>& init)
        : BaseLink(init, ActiveFlags), m_owner(init.owner)
    {
        if (m_owner) m_owner->addLink(this);
    }

    ~TLink() override
    {
    }

    size_t size() const
    {
        return static_cast<size_t>(m_value.size());
    }

    bool empty() const
    {
        return m_value.empty();
    }

    const Container& getValue() const
    {
        return m_value;
    }

    const_iterator begin() const
    {
        return m_value.cbegin();
    }

    const_iterator end() const
    {
        return m_value.cend();
    }

    const_reverse_iterator rbegin() const
    {
        return m_value.crbegin();
    }

    const_reverse_iterator rend() const
    {
        return m_value.crend();
    }

    void clear()
    {
        TraitsContainer::clear(m_value);
    }

    bool addBegin(DestPtr v)
    {
        if (!v)
            return false;
        const std::size_t index = TraitsContainer::addBegin(m_value,v);
        updateCounter();
        added(v, index);
        return true;
    }

    bool add(DestPtr v)
    {
        if (!v)
            return false;
        const std::size_t index = TraitsContainer::add(m_value,v);
        updateCounter();
        added(v, index);
        return true;
    }

    bool add(DestPtr v, const std::string& path)
    {
        if (!v && path.empty())
            return false;
        std::size_t index = TraitsContainer::add(m_value,v);
        TraitsValueType::setPath(m_value[index],path);
        updateCounter();
        added(v, index);
        return true;
    }

    bool addPath(const std::string& path)
    {
        if (path.empty())
            return false;
        DestType* ptr = nullptr;
        if (m_owner)
            PathResolver::FindLinkDest(m_owner, ptr, path, this);
        return add(ptr, path);
    }

    bool remove(DestPtr v)
    {
        if (!v)
            return false;
        return removeAt(TraitsContainer::find(m_value,v));
    }

    bool removeAt(std::size_t index)
    {
        if (index >= m_value.size())
            return false;

        DestPtr v = TraitsDestPtr::get(TraitsValueType::get(m_value[index]));
        TraitsContainer::remove(m_value,index);
        updateCounter();
        removed(v, index);
        return true;
    }

    bool removePath(const std::string& path)
    {
        if (path.empty()) return false;
        const std::size_t n = m_value.size();
        for (std::size_t index=0; index<n; ++index)
        {
            std::string p = getPath(index);
            if (p == path)
                return removeAt(index);
        }
        return false;
    }

    const BaseClass* getDestClass() const override
    {
        return sofa::core::objectmodel::base::GetClass<DestType>();
    }

    const BaseClass* getOwnerClass() const override
    {
        return sofa::core::objectmodel::base::GetClass<OwnerType>();
    }

    size_t getSize() const override
    {
        return size();
    }

    sofa::core::objectmodel::Base* getOwnerBase() const override
    {
        return m_owner;
    }

    void setOwner(OwnerType* owner)
    {
        m_owner = owner;
        if (!owner) return;
        m_owner->addLink(this);
    }

protected:
    OwnerType* m_owner {nullptr};
    Container m_value;

    DestType* getIndex(std::size_t index) const
    {
        if (index < m_value.size())
            return TraitsDestPtr::get(TraitsValueType::get(m_value[index]));
        return nullptr;
    }

    virtual void added(DestPtr ptr, std::size_t index) = 0;
    virtual void removed(DestPtr ptr, std::size_t index) = 0;

    void _doClear_() override
    {
         TraitsContainer::clear(m_value);
    }

    /// Set a new link entry from a Base*
    /// returns false if neither base & path are provided or if the provided base object has the wrong type.
    bool _doAdd_(Base* baseptr, const std::string& path) override
    {
        /// If the pointer is null and the path empty we do nothing
        if(!baseptr && path.empty())
            return false;

        /// Downcast the pointer to a compatible type and
        /// If the types are not compatible with the Link we returns false
        auto destptr = castTo<DestType*>(baseptr);
        if(baseptr && !destptr)
        {
            return false;
        }

        /// TLink:adding accepts nullptr (for a not yet resolved link).
        return TLink::add(destptr, path);
    }

    /// Returns false on type mismatch
    bool _doSet_(Base* baseptr, const size_t index) override
    {
        assert(index < m_value.size());
        auto destptr = castTo<DestType*>(baseptr);

        if(!destptr)
            return false;

        TraitsValueType::set(m_value[index], destptr);
        return true;
    }

    Base* _doGet_(const size_t index=0) const override
    {
        return sofa::core::castToBase(getIndex(index));
    }

    std::string _doGetLinkedPath_(const std::size_t index=0) const override
    {
        if (index >= m_value.size())
            return std::string();
        std::string path;
        const ValueType& value = m_value[index];
        if (!TraitsValueType::path(value, path))
        {
            DestType* ptr = TraitsDestPtr::get(TraitsValueType::get(value));
            if (ptr)
                path = BaseLink::CreateString(sofa::core::castToBase(ptr), m_owner);
        }
        return path;
    }

    Base* _doGetOwner_() const override { return m_owner; }
    void _doSetOwner_(Base* owner) override
    {
        m_owner = dynamic_cast<OwnerType*>(owner);
    }

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

    using ValidatorFn = std::function<void(DestPtr, std::size_t, bool)>;

    MultiLink() : m_validator{nullptr} {}

    MultiLink(const BaseLink::InitLink<OwnerType>& init)
        : Inherit(init), m_validator(nullptr)
    {
    }

    MultiLink(const BaseLink::InitLink<OwnerType>& init, DestPtr val)
        : Inherit(init), m_validator(nullptr)
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

    DestType* get(std::size_t index) const
    {
        if (index < this->m_value.size())
            return TraitsDestPtr::get(TraitsValueType::get(this->m_value[index]));
        else
            return nullptr;
    }

    DestType* operator[](std::size_t index) const
    {
        return get(index);
    }

protected:
    ValidatorFn m_validator;

    void added(DestPtr val, std::size_t index)
    {
        if (m_validator)
            m_validator(val, index, true);
    }

    void removed(DestPtr val, std::size_t index)
    {
        if (m_validator)
            m_validator(val, index, false);
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
    using Inherit::updateCounter;
    using Inherit::m_value;
    using Inherit::m_owner;

    using ValidatorFn = std::function<void(DestPtr, DestPtr&)>;

    SingleLink()
        : m_validator(nullptr)
    {
    }

    SingleLink(const BaseLink::InitLink<OwnerType>& init)
        : Inherit(init), m_validator(nullptr)
    {
    }

    SingleLink(const BaseLink::InitLink<OwnerType>& init, DestPtr val)
        : Inherit(init), m_validator(nullptr)
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

    DestType* get() const
    {
        return TraitsDestPtr::get(TraitsValueType::get(m_value.get()));
    }

    void reset()
    {
        set(nullptr);
    }

    void set(DestPtr v)
    {
        TraitsContainer::resize(m_value, 1);
        ValueType& value = m_value.get();
        const DestPtr before = TraitsValueType::get(value);
        if (v == before) return;
        TraitsValueType::set(value, v);
        updateCounter();
        changed(before, v);
    }

    void set(DestPtr v, const std::string& path)
    {
        TraitsContainer::resize(m_value, 1);
        ValueType& value = m_value.get();
        const DestPtr before = TraitsValueType::get(value);
        if (v != before)
            TraitsValueType::set(value, v);
        TraitsValueType::setPath(value, path);
        updateCounter();
        if (v != before)
            changed(before, v);
    }

    void setPath(const std::string& path)
    {
        TraitsContainer::resize(m_value, 1);
        if (path.empty())
        {
            set(nullptr);
            return;
        }

        DestType* ptr = nullptr;
        if (m_owner)
            PathResolver::FindLinkDest(m_owner, ptr, path, this);
        set(ptr, path);
    }

    /// Convenient operators to make a SingleLink appear as a regular pointer
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
protected:
    ValidatorFn m_validator;


    void added(DestPtr val, std::size_t /*index*/)
    {
        if (m_validator)
        {
            DestPtr after = val;
            m_validator(nullptr, after);
            if (after != val)
                TraitsValueType::set(m_value.get(), after);
        }
    }

    void removed(DestPtr val, std::size_t /*index*/)
    {
        if (m_validator)
        {
            DestPtr after = nullptr;
            m_validator(val, after);
            if (after)
                TraitsValueType::set(m_value.get(), after);
        }
    }

    void changed(DestPtr before, DestPtr val)
    {
        if (m_validator)
        {
            DestPtr after = val;
            m_validator(before, after);
            if (after != val)
                TraitsValueType::set(this->m_value.get(), after);
        }
    }
};
} // namespace core::objectmodel

// the SingleLink class is used everywhere
using core::objectmodel::SingleLink;

// the MultiLink class is used everywhere
using core::objectmodel::MultiLink;

} // namespace sofa
