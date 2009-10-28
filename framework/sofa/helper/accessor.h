/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_HELPER_ACCESSOR_H
#define SOFA_HELPER_ACCESSOR_H

#include <sofa/helper/helper.h>
#include <iostream>

namespace sofa
{

namespace helper
{

/** A ReadAccessor is a proxy class, holding a reference to a given container
 *  and providing access to its data, using an unified interface (similar to
 *  std::vector), hiding API differences within some containers.
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
 */

template<class T>
class ReadAccessor
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
    const container_type& ref;

public:
    ReadAccessor(const container_type& container) : ref(container) {}
    ~ReadAccessor() {}

    size_type size() const { return ref.size(); }
    const_reference operator[](size_type i) const { return ref[i]; }

    const_iterator begin() const { return ref.begin(); }
    const_iterator end() const { return ref.end(); }

    inline friend std::ostream& operator<< ( std::ostream& os, const ReadAccessor<T>& vec )
    {
        return os << vec;
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
 */

template<class T>
class WriteAccessor
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
    container_type& ref;

public:
    WriteAccessor(container_type& container) : ref(container) {}
    ~WriteAccessor() {}

    size_type size() const { return ref.size(); }

    const_reference operator[](size_type i) const { return ref[i]; }
    reference operator[](size_type i) { return ref[i]; }

    const_iterator begin() const { return ref.begin(); }
    iterator begin() { return ref.begin(); }
    const_iterator end() const { return ref.end(); }
    iterator end() { return ref.end(); }

    void clear() { ref.clear(); }
    void resize(size_type s, bool /*init*/ = true) { ref.resize(s); }
    void reserve(size_type s) { ref.reserve(s); }
    void push_back(const_reference v) { ref.push_back(v); }

    inline friend std::ostream& operator<< ( std::ostream& os, const WriteAccessor<T>& vec )
    {
        return os << vec.ref;
    }

    inline friend std::istream& operator>> ( std::istream& in, WriteAccessor<T>& vec )
    {
        return in >> vec.ref;
    }

};

} // namespace helper

} // namespace sofa

#endif
