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


template <class BaseMapping>
CenterPointMechanicalMapping<BaseMapping>::CenterPointMechanicalMapping(In* from, Out* to)
    : Inherit(from, to)
    , inputTopo(NULL)
    , outputTopo(NULL)
{
}

template <class BaseMapping>
CenterPointMechanicalMapping<BaseMapping>::~CenterPointMechanicalMapping()
{
}

template <class BaseMapping>
void CenterPointMechanicalMapping<BaseMapping>::init()
{
    inputTopo = this->fromModel->getContext()->getMeshTopology();
    outputTopo = this->toModel->getContext()->getMeshTopology();
    this->Inherit::init();
}

template <class BaseMapping>
void CenterPointMechanicalMapping<BaseMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
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

template <class BaseMapping>
void CenterPointMechanicalMapping<BaseMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
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

template <class BaseMapping>
void CenterPointMechanicalMapping<BaseMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
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

template <class BaseMapping>
void CenterPointMechanicalMapping<BaseMapping>::applyJT( typename In::MatrixDeriv& /*out*/, const typename Out::MatrixDeriv& /*in*/ )
{
    // TODO

    return;
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
