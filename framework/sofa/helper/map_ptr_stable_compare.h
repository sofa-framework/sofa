/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_MAP_PTR_STABLE_COMPARE_H
#define SOFA_HELPER_MAP_PTR_STABLE_COMPARE_H

#include <sofa/helper/helper.h>

#include <map>

namespace sofa
{

namespace helper
{

/// An object transforming pointers to stable ids, i.e. whose value depend on the order the pointers
/// are processed, and not their (potentially random) value
template <typename T>
class ptr_stable_id
{
public:
	ptr_stable_id() : counter(0) {}
	unsigned int operator()(T* p)
	{
		unsigned int id = 0;
        typename std::map<T*,unsigned int>::iterator it = idMap.find(p);
		if (it != idMap.end())
		{
			id = it->second;
		}
		else
		{
			id = ++counter;
			idMap.insert(std::make_pair(p, id));
		}
		return id;
	}
protected:
	mutable unsigned int counter;
	mutable std::map<T*,unsigned int> idMap;
};

/// A comparison object that order pointers in a stable way, i.e. in the order pointers are presented
template <typename T>
class ptr_stable_compare;

template <typename T>
class ptr_stable_compare<T*>
{
public:
	// This operator must be declared const in order to be used within const methods
	// such as std::map::find()
	bool operator()(T* a, T* b) const
	{
		return (ids(a) < ids(b));
	}
protected:
	mutable ptr_stable_id<T> ids;
};

template <typename T>
class ptr_stable_compare< std::pair<T*,T*> >
{
public:
	// This operator must be declared const in order to be used within const methods
	// such as std::map::find()
	bool operator()(const std::pair<T*,T*>& a, const std::pair<T*,T*>& b) const
	{
		return (std::make_pair(ids(a.first),ids(a.second)) < std::make_pair(ids(b.first),ids(b.second)));
	}
protected:
	mutable ptr_stable_id<T> ids;
};

/// A map container that order pointers in a stable way, i.e. in the order pointers are presented
template< typename Key, typename Tp >
class map_ptr_stable_compare : public std::map<Key, Tp, ptr_stable_compare<Key> >
{
public:
	typedef std::map<Key, Tp, ptr_stable_compare<Key> > Inherit;
    /// Key
	typedef typename Inherit::key_type               key_type;
    /// Tp
	typedef typename Inherit::mapped_type            mapped_type;
    /// pair<Key,Tp>
	typedef typename Inherit::value_type             value_type;
    /// reference to a value (read-write)
    typedef typename Inherit::reference              reference;
    /// const reference to a value (read only)
    typedef typename Inherit::const_reference        const_reference;
    /// iterator
    typedef typename Inherit::iterator               iterator;
    /// const iterator
    typedef typename Inherit::const_iterator         const_iterator;
    /// reverse iterator
    typedef typename Inherit::reverse_iterator       reverse_iterator;
    /// const reverse iterator
    typedef typename Inherit::const_reverse_iterator const_reverse_iterator;

    /// Basic constructor
    map_ptr_stable_compare() {}
    /// Constructor
    map_ptr_stable_compare(const Inherit& x): Inherit(x) {}
    /// Constructor
    map_ptr_stable_compare<Key, Tp>& operator=(const Inherit& x)
    {
        Inherit::operator = (x);
        return (*this);
    }

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    map_ptr_stable_compare(InputIterator first, InputIterator last): Inherit(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    map_ptr_stable_compare(const_iterator first, const_iterator last): Inherit(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */
};

} // namespace helper

} // namespace sofa

#endif
