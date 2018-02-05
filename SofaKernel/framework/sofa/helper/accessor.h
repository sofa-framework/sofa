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
#ifndef SOFA_HELPER_ACCESSOR_H
#define SOFA_HELPER_ACCESSOR_H

#include <sofa/helper/helper.h>
#include <sofa/helper/vector.h>
#include <iostream>

namespace sofa
{

namespace helper
{

/** A ReadAccessor is a proxy class, holding a reference to a given container
 *  and providing access to its data, using an unified interface (similar to
 *  std::vector), hiding API differences within containers.
 *
 *  Other advantadges of using a ReadAccessor are :
 *
 *  - It can be faster that the default methods and operators of the container,
 *  as verifications and changes notifications can be handled in the accessor's
 *  constructor and destructor instead of at each item access.
 *
 *  - No modifications to the container will be done by mistake
 *
 *  - Accesses can be logged for debugging or task dependencies analysis.
 *
 *  The default implementation provides only minimal set of methods and
 *  operators, sufficient for scalar types but which should be overloaded for
 *  more complex types.
 *  Various template specializations are typically used, especially for core::objectmodel::Data<T>
 */
template<class T>
class ReadAccessor
{
public:
    typedef T container_type;
    typedef T value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef const value_type& const_reference;
    typedef const value_type* const_pointer;

protected:
    const container_type* vref;

public:
    explicit ReadAccessor(const container_type& container) : vref(&container) {}

    const_reference ref() const { return *vref; }

    operator  const_reference () const { return  *vref; }
    const_pointer   operator->() const { return vref; }
    const_reference operator* () const { return  *vref; }

    inline friend std::ostream& operator<< ( std::ostream& os, const ReadAccessor<T>& vec )
    {
        return os << *vec.vref;
    }
};

/** A WriteAccessor is a proxy class, holding a reference to a given container
 *  and providing access to its data, using an unified interface (similar to
 *  std::vector), hiding API differences within some containers.
 *
 *  Other advantadges of using a WriteAccessor are :
 *
 *  - It can be faster that the default methods and operators of the container,
 *  as verifications and changes notifications can be handled in the accessor's
 *  constructor and destructor instead of at each item access.
 *
 *  - Accesses can be logged for debugging or task dependencies analysis.
 *
 *  The default implementation provides only minimal set of methods and
 *  operators, sufficient for scalar types but which should be overloaded for
 *  more complex types.
 *  Various template specializations are typically used, especially for core::objectmodel::Data<T>
 */
template<class T>
class WriteAccessor
{
public:
    typedef T container_type;
    typedef T value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef const value_type& const_reference;
    typedef const value_type* const_pointer;

protected:
    container_type* vref;

public:
    explicit WriteAccessor(container_type& container) : vref(&container) {}

    const_reference ref() const { return *vref; }
    reference wref() { return *vref; }

    operator  const_reference () const { return  *vref; }
    const_pointer   operator->() const { return vref; }
    const_reference operator* () const { return  *vref; }

    operator  reference () { return  *vref; }
    pointer   operator->() { return vref; }
    reference operator* () { return  *vref; }

    template<class T2>
    void operator=(const T2& v)
    {
        vref = &v;
    }

    inline friend std::ostream& operator<< ( std::ostream& os, const WriteAccessor<T>& vec )
    {
        return os << *vec.vref;
    }

    inline friend std::istream& operator>> ( std::istream& in, WriteAccessor<T>& vec )
    {
        return in >> *vec.vref;
    }
};


/** Identical to WriteAccessor for default implementation, but different for some template specializations such as  core::objectmodel::Data<T>
*/
template<class T>
class WriteOnlyAccessor : public WriteAccessor<T>
{
protected:
    typedef WriteAccessor<T> Inherit;
    typedef typename Inherit::container_type container_type;

public:
    explicit WriteOnlyAccessor(container_type& container) : WriteAccessor<T>(container) {}
};



//////////////////////////




/// ReadAccessor implementation class for vector types
template<class T>
class ReadAccessorVector
{
public:
    typedef T container_type;
    typedef typename container_type::size_type size_type;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    const container_type* vref;

public:
    ReadAccessorVector(const container_type& container) : vref(&container) {}

    const container_type& ref() const { return *vref; }

    bool empty() const { return vref->empty(); }
    size_type size() const { return vref->size(); }
	const_reference operator[](size_type i) const { return (*vref)[i]; }

    const_iterator begin() const { return vref->begin(); }
    const_iterator end() const { return vref->end(); }

    inline friend std::ostream& operator<< ( std::ostream& os, const ReadAccessorVector<T>& vec )
    {
        return os << *vec.vref;
    }

};

/// WriteAccessor implementation class for vector types
template<class T>
class WriteAccessorVector
{
public:
    typedef T container_type;
    typedef typename container_type::size_type size_type;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    container_type* vref;

public:
    WriteAccessorVector(container_type& container) : vref(&container) {}
    
    const container_type& ref() const { return *vref; }
    container_type& wref() { return *vref; }

    bool empty() const { return vref->empty(); }
    size_type size() const { return vref->size(); }

	const_reference operator[](size_type i) const { return (*vref)[i]; }
	reference operator[](size_type i) { return (*vref)[i]; }

    const_iterator begin() const { return vref->begin(); }
    iterator begin() { return vref->begin(); }
    const_iterator end() const { return vref->end(); }
    iterator end() { return vref->end(); }

    void clear() { vref->clear(); }
    void resize(size_type s, bool /*init*/ = true) { vref->resize(s); }
    void reserve(size_type s) { vref->reserve(s); }
    void push_back(const value_type& v) { vref->push_back(v); }

    inline friend std::ostream& operator<< ( std::ostream& os, const WriteAccessorVector<T>& vec )
    {
        return os << *vec.vref;
    }

    inline friend std::istream& operator>> ( std::istream& in, WriteAccessorVector<T>& vec )
    {
        return in >> *vec.vref;
    }

};

// Support for std::vector

template<class T, class Alloc>
class ReadAccessor< std::vector<T,Alloc> > : public ReadAccessorVector< std::vector<T,Alloc> >
{
public:
    typedef ReadAccessorVector< std::vector<T,Alloc> > Inherit;
    typedef typename Inherit::container_type container_type;
    ReadAccessor(const container_type& c) : Inherit(c) {}
};

template<class T, class Alloc>
class WriteAccessor< std::vector<T,Alloc> > : public WriteAccessorVector< std::vector<T,Alloc> >
{
public:
    typedef WriteAccessorVector< std::vector<T,Alloc> > Inherit;
    typedef typename Inherit::container_type container_type;
    WriteAccessor(container_type& c) : Inherit(c) {}
};

template<class T, class Alloc>
class ReadAccessor< helper::vector<T,Alloc> > : public ReadAccessorVector< helper::vector<T,Alloc> >
{
public:
    typedef ReadAccessorVector< helper::vector<T,Alloc> > Inherit;
    typedef typename Inherit::container_type container_type;
    ReadAccessor(const container_type& c) : Inherit(c) {}
};

template<class T, class Alloc>
class WriteAccessor< helper::vector<T,Alloc> > : public WriteAccessorVector< helper::vector<T,Alloc> >
{
public:
    typedef WriteAccessorVector< helper::vector<T,Alloc> > Inherit;
    typedef typename Inherit::container_type container_type;
    WriteAccessor(container_type& c) : Inherit(c) {}
};



} // namespace helper

} // namespace sofa

#endif
