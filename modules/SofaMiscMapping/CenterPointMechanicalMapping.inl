/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_CENTERPOINTMAPPING_INL
#define SOFA_COMPONENT_MAPPING_CENTERPOINTMAPPING_INL

#include "CenterPointMechanicalMapping.h"

#include <sofa/core/topology/BaseMeshTopology.h>


namespace sofa
{

namespace component
{

namespace mapping
{


template <class TIn, class TOut>
CenterPointMechanicalMapping<TIn, TOut>::CenterPointMechanicalMapping()
    : Inherit()
    , inputTopo(NULL)
    , outputTopo(NULL)
{
}

template <class TIn, class TOut>
CenterPointMechanicalMapping<TIn, TOut>::~CenterPointMechanicalMapping()
{
}

template <class TIn, class TOut>
void CenterPointMechanicalMapping<TIn, TOut>::init()
{
    inputTopo = this->fromModel->getContext()->getMeshTopology();
    outputTopo = this->toModel->getContext()->getMeshTopology();
    this->Inherit::init();
}

template <class TIn, class TOut>
void CenterPointMechanicalMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/, Data< typename Out::VecCoord >& _out, const Data< typename In::VecCoord >& _in)
{
    helper::WriteAccessor< Data< typename Out::VecCoord > > out = _out;
    helper::ReadAccessor< Data< typename In::VecCoord > > in = _in;

    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = inputTopo->getHexahedra();

    if(out.size() < hexahedra.size())
        out.resize(hexahedra.size());

    for(unsigned int i = 0; i < hexahedra.size(); ++i)
    {
        out[i] =(in[ hexahedra[i][0] ]
                + in[ hexahedra[i][1] ]
                + in[ hexahedra[i][2] ]
                + in[ hexahedra[i][3] ]
                + in[ hexahedra[i][4] ]
                + in[ hexahedra[i][5] ]
                + in[ hexahedra[i][6] ]
                + in[ hexahedra[i][7] ]) * 0.125f;
    }
}

template <class TIn, class TOut>
void CenterPointMechanicalMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/, Data< typename Out::VecDeriv >& _out, const Data< typename In::VecDeriv >& _in)
{
    helper::WriteAccessor< Data< typename Out::VecDeriv > > out = _out;
    helper::ReadAccessor< Data< typename In::VecDeriv > > in = _in;

    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = inputTopo->getHexahedra();

    if(out.size() < hexahedra.size())
        out.resize(hexahedra.size());

    for(unsigned int i = 0; i < hexahedra.size(); ++i)
    {
        out[i] =(in[ hexahedra[i][0] ]
                + in[ hexahedra[i][1] ]
                + in[ hexahedra[i][2] ]
                + in[ hexahedra[i][3] ]
                + in[ hexahedra[i][4] ]
                + in[ hexahedra[i][5] ]
                + in[ hexahedra[i][6] ]
                + in[ hexahedra[i][7] ]) * 0.125f;
    }
}

template <class TIn, class TOut>
void CenterPointMechanicalMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/, Data< typename In::VecDeriv >& _out, const Data< typename Out::VecDeriv >& _in)
{
    helper::WriteAccessor< Data< typename In::VecDeriv > > out = _out;
    helper::ReadAccessor< Data< typename Out::VecDeriv > > in = _in;

    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = inputTopo->getHexahedra();

    for(unsigned int i = 0; i <hexahedra.size(); ++i)
    {
        if( in.size() <= i ) continue;

        typename Out::Deriv val = in[i] * 0.125f;

        out[ hexahedra[i][0] ] += val;
        out[ hexahedra[i][1] ] += val;
        out[ hexahedra[i][2] ] += val;
        out[ hexahedra[i][3] ] += val;
        out[ hexahedra[i][4] ] += val;
        out[ hexahedra[i][5] ] += val;
        out[ hexahedra[i][6] ] += val;
        out[ hexahedra[i][7] ] += val;
    }
}

template <class TIn, class TOut>
void CenterPointMechanicalMapping<TIn, TOut>::applyJT(const core::ConstraintParams * /*cparams*/, Data< typename In::MatrixDeriv >& /*out*/, const Data< typename Out::MatrixDeriv >& /*in*/)
{
    // TODO

    return;
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
