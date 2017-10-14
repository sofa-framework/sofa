/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_ENGINE_DISPLACEMENTTRANSFORMENGINE_INL
#define SOFA_COMPONENT_ENGINE_DISPLACEMENTTRANSFORMENGINE_INL

#include "DisplacementTransformEngine.h"

namespace sofa
{

namespace component
{

namespace engine
{

///////////////////////////////////////////////////////////////
/// DisplacementTransformEngine
///////////////////////////////////////////////////////////////
template < class DataTypes, class OutputType >
DisplacementTransformEngine< DataTypes, OutputType >::DisplacementTransformEngine()
:
Inherit()
, d_x0( initData( &d_x0, "x0", "Rest position" ) )
, d_x( initData( &d_x, "x", "Current position" ) )
, d_displacements( initData( &d_displacements, "displacements", "Displacement transforms with respect to original rigid positions") )
{
    addInput( &d_x0 );
    addInput( &d_x );
    addOutput( &d_displacements );
    setDirtyValue();
}

template < class DataTypes, class OutputType >
void DisplacementTransformEngine< DataTypes, OutputType >::init()
{
    /// parent method
    Inherit::init();

    /// Computation of inverse matrix
    const VecCoord& x0 = d_x0.getValue();
    inverses.resize(x0.size());
    for( size_t i=0; i<x0.size(); ++i )
    {
        setInverse( inverses[i], x0[i] );
    }
}

template < class DataTypes, class OutputType >
void DisplacementTransformEngine< DataTypes, OutputType >::update()
{
    /// parent method
    Inherit::init();

    const VecCoord& x = d_x.getValue();
    const VecCoord& x0 = d_x0.getValue();
    const size_t size = x.size();
    const size_t size0 = x0.size();

    /// Check the size of x0
    if( size != size0 )
    {
        msg_info() << "x and x0 have not the same size: respectively " << size << " and " << size0 ;
        return;
    }

    cleanDirty();

    /// Clean the output
    helper::vector< OutputType >& displacements = *d_displacements.beginWriteOnly();
    displacements.resize(size);

    for( unsigned int i = 0; i < size; ++i )
    {
        mult( displacements[i], inverses[i], x[i] );
    }
    d_displacements.endEdit();
}

} /// namespace engine
} /// namespace component
} /// namespace sofa

#endif /// SOFA_COMPONENT_ENGINE_DisplacementMatrixEngine_INL
