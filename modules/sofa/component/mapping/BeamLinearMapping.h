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
#ifndef SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_H
#define SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_H

#include <sofa/core/behavior/MechanicalMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
class BeamLinearMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BeamLinearMapping,BasicMapping), BasicMapping);
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::DataTypes OutDataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename defaulttype::SparseConstraint<Deriv> OutSparseConstraint;
    typedef typename OutSparseConstraint::const_data_iterator OutConstraintIterator;

    typedef typename In::DataTypes InDataTypes;
    typedef typename In::Deriv InDeriv;
    typedef typename Coord::value_type Real;
    enum { N=OutDataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;

protected:
    sofa::helper::vector<Coord> points;
    //Coord translation;
    //Real orientation[4];
    //Mat rotation;
    sofa::helper::vector<Real> beamLength;
    sofa::helper::vector<Coord> rotatedPoints0;
    sofa::helper::vector<Coord> rotatedPoints1;
public:
    //Data<unsigned> index;
    Data<bool> localCoord;

    BeamLinearMapping(In* from, Out* to)
        : Inherit(from, to)
        //, index(initData(&index,(unsigned)0,"index","input DOF index"))
        , localCoord(initData(&localCoord,true,"localCoord","true if initial coordinates are in the beam local coordinate system (i.e. a point at (10,0,0) is on the DOF number 10, whereas if this is false it is at whatever position on the beam where the distance from the initial DOF is 10)"))
    {
    }

    virtual ~BeamLinearMapping()
    {
    }

    void init();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );

    void draw();

};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
