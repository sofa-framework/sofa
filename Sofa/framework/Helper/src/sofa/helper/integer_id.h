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
#ifndef SOFA_HELPER_INTEGER_ID_H
#define SOFA_HELPER_INTEGER_ID_H

#include <sofa/helper/config.h>
#include <sofa/type/vector.h>
#include <sofa/helper/accessor.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/Factory.h>
#include <sofa/type/trait/Rebind.h>
#include <limits>

namespace sofa
{

namespace helper
{

typedef const char* (*integer_id_name)();

template < integer_id_name Name, typename Index = unsigned int, Index DefaultId = (Index)-1 >
class integer_id
{
public:
    typedef integer_id<Name, Index, DefaultId> Id;
    typedef int sindex_type;
protected:
    Index index;
public:
    static const char* getName() { return (*Name)(); }
    static Index getDefaultId() { return DefaultId; }

    integer_id() : index(DefaultId) {}
    explicit integer_id(Index i) : index(i) {}
    integer_id(const Id& i) : index(i.index) {}

    Id& operator=(const Id& i)
    {
        index = i.index;
        return *this;
    }

    Index getId() const { return index; }
    void setId(Index i) { index = i; }

    bool isValid() const
    {
        return index != DefaultId;
    }
    bool isValid(Index size) const
    {
        return (unsigned)index < (unsigned)size;
    }

    bool operator==(const Id& a) const
    {
        return index == a.index;
    }

    bool operator==(const Index& i) const
    {
        return index == i;
    }

    bool operator!=(const Id& a) const
    {
        return index != a.index;
    }

    bool operator!=(const Index& i) const
    {
        return index != i;
    }

    bool operator<(const Id& a) const
    {
        return index < a.index;
    }

    bool operator<(const Index& i) const
    {
        return index < i;
    }

    bool operator<=(const Id& a) const
    {
        return index <= a.index;
    }

    bool operator<=(const Index& i) const
    {
        return index <= i;
    }

    bool operator>(const Id& a) const
    {
        return index > a.index;
    }

    bool operator>(const Index& i) const
    {
        return index > i;
    }

    bool operator>=(const Id& a) const
    {
        return index >= a.index;
    }

    bool operator>=(const Index& i) const
    {
        return index >= i;
    }

    template<typename int_type>
    Id operator+(int_type i) const
    {
        return Id(index + i);
    }

    template<typename int_type>
    Id& operator+=(int_type i)
    {
        index += i;
        return *this;
    }

    template<typename int_type>
    Id operator-(int_type i) const
    {
        return Id(index - i);
    }

    sindex_type operator-(const Id& i) const
    {
        return (sindex_type)(index - i.index);
    }
    
    template<typename int_type>
    Id& operator-=(int_type i)
    {
        index -= i;
        return *this;
    }
    
    Id& operator++()
    {
        ++index;
        return *this;
    }

    Id operator++(int)
    {
        Id old = *this;
        index++;
        return old;
    }
    
    Id& operator--()
    {
        --index;
        return *this;
    }
    
    Id operator--(int)
    {
        Id old = *this;
        index--;
        return old;
    }

    /// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const Id& i )
    {
        return os << i.getId();
    }

    /// Input stream
    inline friend std::istream& operator>> ( std::istream& in, Id& i )
    {
        Index v = i.getDefaultId();
        if (in >> v)
            i.setId(v);
        return in;
    }

};

void SOFA_HELPER_API vector_access_failure(const void* vec, unsigned size, unsigned i, const std::type_info& type, const char* tindex)
{
    msg_error("vector") << "in vector<" << gettypename(type) << ", integer_id<" << tindex << "> > " << std::hex << (long)vec << std::dec << " size " << size << " : invalid index " << (int)i;
    BackTrace::dump();
    assert(i < size);
}

template <class T, class TIndex, bool CheckIndices =
#ifdef SOFA_VECTOR_ACCESS_FAILURE
    true
#else
    false
#endif
    , class MemoryManager = CPUMemoryManager<T> >
class vector_id : public vector<T, MemoryManager>
{
public:
    typedef vector<T, MemoryManager> Inherit;
    typedef T value_type;
    typedef TIndex Index;
    typedef Index ID;
    typedef typename Inherit::Size Size;
    typedef typename Inherit::reference reference;
    typedef typename Inherit::const_reference const_reference;
    typedef typename Inherit::iterator iterator;
    typedef typename Inherit::const_iterator const_iterator;

    template<class T2>
    using rebind_to = vector_id< T2, TIndex, CheckIndices, type::rebind_to<MemoryManager, T2> >;


    /// Basic constructor
    vector_id() : Inherit() {}
    /// Constructor
    vector_id(Size n, const T& value): Inherit(n,value) {}
    /// Constructor
    vector_id(int n, const T& value): Inherit(n,value) {}
    /// Constructor
    vector_id(long n, const T& value): Inherit(n,value) {}
    /// Constructor
    explicit vector_id(Size n): Inherit(n) {}
    /// Constructor
    vector_id(const std::vector<T>& x): Inherit(x) {}

    /// Constructor
    vector_id(const_iterator first, const_iterator last): Inherit(first,last) {}

    /// Read/write random access, with explicit Index
    reference at(Index n)
    {
        if (CheckIndices)
        {
            if (!n.isValid(this->size()))
                vector_access_failure(this, this->size(), n.getId(), typeid(T), n.getName());
        }
        return *(this->begin() + n.getId());
    }

    /// Read-only random access, with explicit Index
    const_reference at(Index n) const
    {
        if (CheckIndices)
        {
            if (!n.isValid(this->size()))
                vector_access_failure(this, this->size(), n.getId(), typeid(T), n.getName());
        }
        return *(this->begin() + n.getId());
    }

    /// Read/write random access, with explicit Index
    reference operator()(Index n)
    {
        return at(n);
    }

    /// Read-only random access, with explicit Index
    const_reference operator()(Index n) const
    {
        return at(n);
    }

    /// Read/write random access, with explicit Index
    reference operator[](Index n)
    {
        return at(n);
    }

    /// Read-only random access
    const_reference operator[](Index n) const
    {
        return at(n);
    }

    Index push_back(const_reference v)
    {
        Index i(this->size());
        Inherit::push_back(v);
        return i;
    }
protected:
    
    /// Read/write random access with regular index type, protected to force use of explicit Index
    reference operator[](Size n)
    {
        return at(Index(n));
    }
    
    /// Read-only random access with regular index type, protected to force use of explicit Index
    const_reference operator[](Size n) const
    {
        return at(Index(n));
    }

};


/// ReadAccessor implementation class for vector_id types
template<class T>
class ReadAccessorVectorId
{
public:
    typedef T container_type;
    typedef typename container_type::Index Index;
    typedef typename container_type::Size Size;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    const container_type& vref;

public:
    ReadAccessorVectorId(const container_type& container) : vref(container) {}
    ~ReadAccessorVectorId() {}

    const container_type& ref() const { return vref; }

    bool empty() const { return vref.empty(); }
    Size size() const { return vref.size(); }
    const_reference operator[](Index i) const { return vref[i]; }
    const_reference operator()(Index i) const { return vref(i); }

    const_iterator begin() const { return vref.begin(); }
    const_iterator end() const { return vref.end(); }

    inline friend std::ostream& operator<< ( std::ostream& os, const ReadAccessorVectorId<T>& vec )
    {
        return os << vec.vref;
    }

};

/// WriteAccessor implementation class for vector_id types
template<class T>
class WriteAccessorVectorId
{
public:
    typedef T container_type;
    typedef typename container_type::Index Index;
    typedef typename container_type::Size Size;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    container_type& vref;

public:
    WriteAccessorVectorId(container_type& container) : vref(container) {}
    ~WriteAccessorVectorId() {}

    const container_type& ref() const { return vref; }
    container_type& wref() { return vref; }

    bool empty() const { return vref.empty(); }
    Size size() const { return vref.size(); }

    const_reference operator[](Index i) const { return vref[i]; }
    const_reference operator()(Index i) const { return vref(i); }
    reference operator[](Index i) { return vref[i]; }
    reference operator()(Index i) { return vref(i); }

    const_iterator begin() const { return vref.begin(); }
    iterator begin() { return vref.begin(); }
    const_iterator end() const { return vref.end(); }
    iterator end() { return vref.end(); }

    void clear() { vref.clear(); }
    void resize(Size s, bool /*init*/ = true) { vref.resize(s); }
    void reserve(Size s) { vref.reserve(s); }
    Index push_back(const_reference v) { return vref.push_back(v); }

    inline friend std::ostream& operator<< ( std::ostream& os, const WriteAccessorVectorId<T>& vec )
    {
        return os << vec.vref;
    }

    inline friend std::istream& operator>> ( std::istream& in, WriteAccessorVectorId<T>& vec )
    {
        return in >> vec.vref;
    }

};

// Support for vector_id

template <class T, class TIndex, bool CheckIndices, class MemoryManager>
class ReadAccessor< vector_id<T, TIndex, CheckIndices, MemoryManager> > : public ReadAccessorVectorId< vector_id<T, TIndex, CheckIndices, MemoryManager> >
{
public:
    typedef ReadAccessorVectorId< vector_id<T, TIndex, CheckIndices, MemoryManager> > Inherit;
    typedef typename Inherit::container_type container_type;
    ReadAccessor(const container_type& c) : Inherit(c) {}
};

template <class T, class TIndex, bool CheckIndices, class MemoryManager>
class WriteAccessor< vector_id<T, TIndex, CheckIndices, MemoryManager> > : public WriteAccessorVectorId< vector_id<T, TIndex, CheckIndices, MemoryManager> >
{
public:
    typedef WriteAccessorVectorId< vector_id<T, TIndex, CheckIndices, MemoryManager> > Inherit;
    typedef typename Inherit::container_type container_type;
    WriteAccessor(container_type& c) : Inherit(c) {}
};

} // namespace helper

} // namespace sofa

#endif
