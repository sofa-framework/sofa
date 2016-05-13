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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_HexahedronCompositeFEMEdgeBasedMapping_H
#define SOFA_COMPONENT_MAPPING_HexahedronCompositeFEMEdgeBasedMapping_H

//#include <sofa/core/behavior/MechanicalMapping.h>
//#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/vector.h>

//#include <sofa/defaulttype/Mat.h>

#include <SofaNonUniformFem/HexahedronCompositeFEMMapping.h>

namespace sofa
{

namespace component
{


namespace mapping
{



using namespace sofa::core::behavior;
using namespace sofa::defaulttype;

template <class BasicMapping>
class HexahedronCompositeFEMEdgeBasedMapping : public HexahedronCompositeFEMMapping<BasicMapping>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HexahedronCompositeFEMEdgeBasedMapping,BasicMapping), SOFA_TEMPLATE(HexahedronCompositeFEMMapping,BasicMapping));

    typedef HexahedronCompositeFEMMapping<BasicMapping> Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename In::Real InReal;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename OutCoord::value_type Real;
    typedef typename Inherit::Weight Weight;


    HexahedronCompositeFEMEdgeBasedMapping ( In* from, Out* to ): Inherit ( from, to )
    {
    }

    virtual ~HexahedronCompositeFEMEdgeBasedMapping() {}

    virtual void init();

    virtual void apply ( OutVecCoord& out, const InVecCoord& in );



protected :


    static const int EDGES[12][3]; // 2 indices + dir (0=x,1=y,2=z)
    typedef helper::fixed_array<int,3> Edge;// 2 indices + dir (0=x,1=y,2=z)
    helper::vector< Edge > _edges;
    helper::vector<std::map<int,Real> > _weightsEdge; // for each fine nodes -> list of edges with coef
    InCoord _size0;
    helper::vector< std::map< int, Real > > _coarseBarycentricCoord; // barycentric coordinates for each fine points into the coarse elements (coarse nodes idx + weights)
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
