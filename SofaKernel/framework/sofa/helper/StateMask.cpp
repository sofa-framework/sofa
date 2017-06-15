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

#include "StateMask.h"

#include <functional>

namespace sofa
{

namespace helper
{

#ifdef SOFA_USE_MASK

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

    size_t StateMask::getHash() const
    {
        return std::hash<std::vector<bool> >()(mask);
    }


#endif

} // namespace helper

} // namespace sofa
