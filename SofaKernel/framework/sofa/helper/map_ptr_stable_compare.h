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
#ifndef SOFA_HELPER_MAP_PTR_STABLE_COMPARE_H
#define SOFA_HELPER_MAP_PTR_STABLE_COMPARE_H

#include <sofa/helper/helper.h>
#include <map>
#include <memory>

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
	ptr_stable_id() 
		: counter(new unsigned int(0))
		, idMap(new std::map<T*, unsigned int>) {}

	unsigned int operator()(T* p)
	{
		unsigned int id = 0;
        typename std::map<T*,unsigned int>::iterator it = idMap->find(p);
		if (it != idMap->end())
		{
			id = it->second;
		}
		else
		{
			id = ++(*counter);
			idMap->insert(std::make_pair(p, id));
		}
		return id;
	}

    inline unsigned int id(T* p)
    {
       return this->operator()(p);
    }

protected:
    mutable std::shared_ptr<unsigned int> counter;
    mutable std::shared_ptr< std::map<T*,unsigned int> > idMap;
};

/// A comparison object that order pointers in a stable way, i.e. in the order pointers are presented
template <typename T>
class ptr_stable_compare;

template <typename T>
class ptr_stable_compare<T*>
{
public:
    // wrap the ptr_stable_id<T> into an opaque type
    typedef ptr_stable_id<T> stable_map_type;
	// This operator must be declared const in order to be used within const methods
	// such as std::map::find()
	bool operator()(T* a, T* b) const
	{
		return (m_ids->id(a) < m_ids->id(b));
	}

    explicit ptr_stable_compare( const ptr_stable_id<T>* ids ):m_ids(ids)
    {
    } 

protected:
    /// memory is owned by the map_ptr_stable_compare instance
	mutable ptr_stable_id<T>* m_ids;
};

template <typename T>
class ptr_stable_compare< std::pair<T*,T*> >
{
public:
    // wrap the ptr_stable_id<T> into an opaque type
    typedef ptr_stable_id<T> stable_id_map_type;
	// This operator must be declared const in order to be used within const methods
	// such as std::map::find()
	bool operator()(const std::pair<T*,T*>& a, const std::pair<T*,T*>& b) const
	{
		return (std::make_pair(m_ids->id(a.first), m_ids->id(a.second)) < std::make_pair(m_ids->id(b.first), m_ids->id(b.second)) );
	}

    explicit ptr_stable_compare( ptr_stable_id<T>* ids):m_ids(ids)
    {
    }
    
    ptr_stable_id<T>* get_stable_id_map() const 
    {
        return m_ids;
    } 

protected:
    /// memory is owned by the map_ptr_stable_compare instance
	mutable ptr_stable_id<T>* m_ids;

private:
    ptr_stable_compare():m_ids(NULL){}
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
    /// compare
    typedef typename Inherit::key_compare            key_compare;
    /// stable map type used by the compare object
    typedef typename key_compare::stable_id_map_type stable_id_map_type;

    /// Basic constructor
    map_ptr_stable_compare()
    :Inherit( key_compare(new stable_id_map_type() ) )
    ,m_stable_id_map(Inherit::key_comp().get_stable_id_map())
    {}

    /// Copy constructor
    map_ptr_stable_compare(const map_ptr_stable_compare& other)
    :Inherit( other.begin(), other.end(), key_compare( new stable_id_map_type( *other.m_stable_id_map) ) )
    ,m_stable_id_map(Inherit::key_comp().get_stable_id_map())
    {
    }

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    map_ptr_stable_compare(InputIterator first, InputIterator last)
    :Inherit(first,last, key_compare(new stable_id_map_type()))
    ,m_stable_id_map(Inherit::key_comp().get_stable_id_map())
    {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    map_ptr_stable_compare(const_iterator first, const_iterator last)
    :Inherit(first,last, key_compare(new stable_id_map_type()) ) 
    ,m_stable_id_map(Inherit::key_comp().get_stable_id_map())
    {}
#endif /* __STL_MEMBER_TEMPLATES */

private:
    /// smart ptr for memory ownership
    std::unique_ptr<stable_id_map_type> m_stable_id_map;
};

} // namespace helper

} // namespace sofa

#endif
