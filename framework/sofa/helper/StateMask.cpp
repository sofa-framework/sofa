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

#include "StateMask.h"

namespace sofa
{

namespace helper
{

    void StateMask::resize( size_t size )
    {
        mask.resize( size );
        /*assign( size, true );*/
    }

    void StateMask::assign( size_t size, bool value )
    {
        mask.assign( size, value );
    }

    bool StateMask::getActivatedEntry( size_t index ) const
    {
        return activated ? mask[index] : true; // a 'if' at each check rather than a single 'if' per mapping function is the price to pay no to have duplicated code in mappings
        // TODO: implementing it with a fonction pointer?
    }

    void StateMask::activate( bool a )
    {
        activated = a;
    }


} // namespace helper

} // namespace sofa
