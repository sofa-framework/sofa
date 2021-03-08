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

#include <sofa/helper/config.h>
#include <sofa/type/trait/is_container.h>

#include <iosfwd>        ///< Needed to declare the operator<< and >> as deleted.
                         /// Remove it when the operator are completely removed

namespace sofa::helper
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
template<class T, class Enable = void>
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
};

template<class U>
[[deprecated("Custom operator<< for accessor have been deprecated in #PR1808. Just replace std::cout << myaccessor by std::cout << myccessor.ref()")]]
std::ostream& operator<<( std::ostream& os, const ReadAccessor<U>& vec ) = delete;


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
template<class T, class Enable = void>
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
};

template<class U>
[[deprecated("Custom operator<< for accessor have been deprecated in #PR1808. Just replace std::cout << myaccessor by std::cout << myccessor.ref()")]]
std::ostream& operator<< ( std::ostream& os, const WriteAccessor<U>& vec ) = delete;

template<class U>
[[deprecated("Custom operator<< for accessor have been deprecated in #PR1808. Just replace std::cout << myaccessor by std::cout << myccessor.ref()")]]
std::istream& operator>> ( std::istream& in, WriteAccessor<U>& vec ) = delete;

/** Identical to WriteAccessor for default implementation, but different for some template specializations such as  core::objectmodel::Data<T>
*/
template<class T, class Enable = void>
class WriteOnlyAccessor : public WriteAccessor<T, Enable>
{
protected:
    typedef WriteAccessor<T> Inherit;
    typedef typename Inherit::container_type container_type;

public:
    explicit WriteOnlyAccessor(container_type& container) : WriteAccessor<T, Enable>(container) {}
};



////////////////////////// ReadAccessor for wrapping around vector like object //////////////////////
/// ReadAccessor implementation class for vector types
template<class T>
class ReadAccessorVector
{
public:
    typedef T container_type;
    typedef const T const_container_type;
    typedef typename container_type::Size Size;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    const container_type* vref;

public:
    ReadAccessorVector(const container_type& container) : vref(&container) {}

    bool empty() const { return vref->empty(); }
    Size size() const { return vref->size(); }
	const_reference operator[](Size i) const { return (*vref)[i]; }

    const_iterator begin() const { return vref->begin(); }
    const_iterator end() const { return vref->end(); }

    ///////// Access the container for reading ////////////////
    operator  const_container_type () const { return  *vref; }
    const_container_type* operator->() const { return vref; }
    const_container_type& operator* () const { return  *vref; }
    const_container_type& ref() const { return *vref; }          ///< this duplicate operator* (remove ?)
    ///////////////////////////////////////////////////////////
};

template<class U>
[[deprecated("Custom operator<< for accessor have been deprecated in #PR1808. Just replace std::cout << myaccessor by std::cout << myccessor.ref()")]]
std::ostream& operator<< ( std::ostream& os, const ReadAccessorVector<U>& vec ) = delete;


/// WriteAccessor implementation class for vector types
template<class T>
class WriteAccessorVector
{
public:
    typedef T container_type;
    typedef const T const_container_type;
    typedef typename container_type::Size Size;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    container_type* vref;

public:
    WriteAccessorVector(container_type& container) : vref(&container) {}
    
    bool empty() const { return vref->empty(); }
    Size size() const { return vref->size(); }

	const_reference operator[](Size i) const { return (*vref)[i]; }
	reference operator[](Size i) { return (*vref)[i]; }

    const_iterator begin() const { return vref->begin(); }
    iterator begin() { return vref->begin(); }
    const_iterator end() const { return vref->end(); }
    iterator end() { return vref->end(); }

    void clear() { vref->clear(); }
    void resize(Size s, bool /*init*/ = true) { vref->resize(s); }
    void reserve(Size s) { vref->reserve(s); }
    void push_back(const value_type& v) { vref->push_back(v); }

    ////// Access the container in reading & writing //////
    operator  container_type () { return  *vref; }
    container_type* operator->() { return vref; }
    container_type& operator* () { return  *vref; }
    container_type& wref() { return *vref; }
    ///////////////////////////////////////////////////////

    ///////// Access the container for reading ////////////////
    operator  const_container_type () const { return  *vref; }
    const_container_type* operator->() const { return vref; }
    const_container_type& operator* () const { return  *vref; }

    /// this one duplicate operator*
    const container_type& ref() const { return *vref; }
    ///////////////////////////////////////////////////////////
};
///////////////////////////////////////////////////////////////////////////////////////////////

/// Support for std::vector
template<class VectorLikeType>
class ReadAccessor<VectorLikeType,
        typename std::enable_if<sofa::type::trait::is_container<VectorLikeType>::value>::type> : public ReadAccessorVector< VectorLikeType >
{
public:
    typedef ReadAccessorVector< VectorLikeType > Inherit;
    typedef typename Inherit::container_type container_type;
    ReadAccessor(const container_type& c) : Inherit(c) {}
};

template<class VectorLikeType>
class WriteAccessor<VectorLikeType,
        typename std::enable_if<sofa::type::trait::is_container<VectorLikeType>::value>::type> : public WriteAccessorVector< VectorLikeType >
{
public:
    typedef WriteAccessorVector< VectorLikeType > Inherit;
    typedef typename Inherit::container_type container_type;
    WriteAccessor(container_type& c) : Inherit(c) {}
};

template<class VectorLikeType>
class WriteOnlyAccessor<VectorLikeType,
        typename std::enable_if<sofa::type::trait::is_container<VectorLikeType>::value>::type> : public WriteAccessorVector< VectorLikeType >
{
public:
    typedef WriteAccessorVector< VectorLikeType > Inherit;
    typedef typename Inherit::container_type container_type;
    WriteOnlyAccessor(container_type& c) : Inherit(c) {}
};

/// Returns a read accessor from the provided Data<>
/// Example of use:
///   auto points = getReadAccessor(d_points)
template<class D>
sofa::helper::ReadAccessor<D> getReadAccessor(D& c)
{
    return sofa::helper::ReadAccessor<D>{ c };
}

/// Returns a write only accessor from the provided Data<>
/// Example of use:
///   auto points = getWriteOnlyAccessor(d_points)
template<class D>
sofa::helper::WriteAccessor<D> getWriteAccessor(D& c)
{
    return sofa::helper::WriteAccessor<D>{ c };
}

/// Returns a write only accessor from the provided Data<>
/// WriteOnly accessors are faster than WriteAccessor because
/// as the data is only read this means there is no need to pull
/// the data from the parents
/// Example of use:
///   auto points = getWriteOnlyAccessor(d_points)
template<class D>
sofa::helper::WriteOnlyAccessor<D> getWriteOnlyAccessor(D& c)
{
    return sofa::helper::WriteOnlyAccessor<D>{ c };
}

} /// namespace sofa::core

