/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_INL
#define SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_INL

#include "FrameBlendingMapping.h"
#include <sofa/component/mapping/SkinningMapping.inl>
#include <sofa/core/Mapping.inl>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/gl/glText.inl>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/core/Mapping.inl>

#include <string>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


template <class TIn, class TOut>
FrameBlendingMapping<TIn, TOut>::FrameBlendingMapping (core::State<In>* from, core::State<Out>* to )
    : Inherit ( from, to )
    , targetFrameNumber ( initData ( &targetFrameNumber, false, "targetFrameNumber","Target frames number" ) )
    , targetSampleNumber ( initData ( &targetSampleNumber, false, "targetSampleNumber","Target samples number" ) )
{
}

template <class TIn, class TOut>
FrameBlendingMapping<TIn, TOut>::~FrameBlendingMapping ()
{
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::init()
{
    Inherit::init();
}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::apply ( typename Out::VecCoord& /*out*/, const typename In::VecCoord& /*in*/ )
{
    serr<<"WARNING : FrameBlendingMapping<TIn, TOut>::apply does nothing " << endl;
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJ ( typename Out::VecDeriv& /*out*/, const typename In::VecDeriv& /*in*/ )
{
    serr<<"WARNING : FrameBlendingMapping<TIn, TOut>::applyJ does nothing " << endl;
}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJT ( typename In::VecDeriv& /*out*/, const typename Out::VecDeriv& /*in*/ )
{
    serr<<"WARNING : FrameBlendingMapping<TIn, TOut>::applyJT does nothing " << endl;
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJT ( typename In::MatrixDeriv& /*out*/, const typename Out::MatrixDeriv& /*in*/ )
{
    serr<<"WARNING : FrameBlendingMapping<TIn, TOut>::applyJT does nothing " << endl;
}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
