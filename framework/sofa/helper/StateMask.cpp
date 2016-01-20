/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "StateMask.h"

//#include <boost/functional/hash.hpp>



namespace sofa
{

namespace helper
{

#ifdef SOFA_USE_MASK

//    static boost::hash<StateMask::InternalStorage> s_maskHash;

    void StateMask::resize( size_t size )
    {
        mask.resize( size );
        /*assign( size, true );*/
    }

    void StateMask::assign( size_t size, bool value )
    {
        mask.assign( size, value );
    }

    void StateMask::activate( bool a )
    {
        activated = a;
    }

    size_t StateMask::nbActiveDofs() const
    {
        size_t t = 0;
        for( size_t i = 0 ; i<size() ; ++i )
            if( getEntry(i) ) t++;
        return t;
    }

//    size_t StateMask::getHash() const
//    {
//        return s_maskHash(mask);
//    }

#else

    void StateMask::resize( size_t size )
    {
        m_size = size;
    }

    void StateMask::assign( size_t size, bool /*value*/ )
    {
        m_size = size;
    }

    void StateMask::activate( bool a )
    {
        activated = a;
    }

    size_t StateMask::nbActiveDofs() const
    {
        return m_size;
    }

#endif

} // namespace helper

} // namespace sofa
