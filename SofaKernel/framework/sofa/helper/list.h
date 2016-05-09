/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_LIST_H
#define SOFA_HELPER_LIST_H

#include <sofa/helper/helper.h>

#include <list>
#include <iostream>
#include <sstream>
#include <string>

namespace sofa
{

namespace helper
{

//======================================================================
///	Same as std::list, + input/output operators
///
///   \see sofa::helper::set
///
//======================================================================

template< class T, class Alloc = std::allocator<T> >
class list: public std::list<T, Alloc>
{
public:

    /// size_type
    typedef typename std::list<T,Alloc>::size_type size_type;
    /// reference to a value (read-write)
    typedef typename std::list<T,Alloc>::reference reference;
    /// const reference to a value (read only)
    typedef typename std::list<T,Alloc>::const_reference const_reference;
    /// iterator
    typedef typename std::list<T,Alloc>::iterator iterator;
    /// const iterator
    typedef typename std::list<T,Alloc>::const_iterator const_iterator;

    /// Basic constructor
    list() {}
    /// Constructor by copy
    list(const std::list<T, Alloc>& x): std::list<T,Alloc>(x) {}
    /// Constructor
    list<T, Alloc>& operator=(const std::list<T, Alloc>& x)
    {
        std::list<T,Alloc>::operator = (x);
        return (*this);
    }


#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    list(InputIterator first, InputIterator last): std::list<T,Alloc>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    list(const_iterator first, const_iterator last): std::list<T,Alloc>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    std::ostream& write(std::ostream& os) const
    {
        if( !this->empty() )
        {
            const_iterator i=this->begin();
            os << *i;
            ++i;
            for( ; i!=this->end(); ++i )
                os << ' ' << *i;
        }
        return os;
    }

    std::istream& read(std::istream& in)
    {
        T t = T();
        this->clear();
        while(in>>t)
        {
            this->push_back(t);
        }
        if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
        return in;
    }

    /// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const list<T,Alloc>& vec )
    {
        return vec.write(os);
    }


    /// Input stream
    inline friend std::istream& operator>> ( std::istream& in, list<T,Alloc>& vec )
    {
        return vec.read(in);
    }

};


} // namespace helper

} // namespace sofa

#endif
