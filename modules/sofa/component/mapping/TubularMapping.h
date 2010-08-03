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
#ifndef SOFA_COMPONENT_MAPPING_TUBULARMAPPING_H
#define SOFA_COMPONENT_MAPPING_TUBULARMAPPING_H

#include <sofa/core/behavior/MechanicalMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/container/RadiusContainer.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
class TubularMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TubularMapping,BasicMapping), BasicMapping);
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::DataTypes DataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    //typedef typename std::map<unsigned int, Deriv>::const_iterator OutConstraintIterator;

    typedef typename In::Deriv InDeriv;
    typedef typename Coord::value_type Real;
    enum { M=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<M,M,Real> Mat;

    typedef defaulttype::Vec<M,Real> Vec;

    TubularMapping ( In* from, Out* to )
        : Inherit ( from, to )
        , m_nbPointsOnEachCircle( initData(&m_nbPointsOnEachCircle, "nbPointsOnEachCircle", "Discretization of created circles"))
        , m_radius( initData(&m_radius, "radius", "Radius of created circles"))
        , m_peak (initData(&m_peak, 0, "peak", "=0 no peak, =1 peak on the first segment =2 peak on the two first segment, =-1 peak on the last segment"))
        ,radiusContainer(NULL)
    {
    }

    virtual ~TubularMapping()
    {}

    void init();

    virtual void apply ( typename Out::VecCoord& out, const typename In::VecCoord& in );

    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    virtual void applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    //void applyJT ( typename In::VecConst& out, const typename Out::VecConst& in );
    void applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );

    Data<unsigned int> m_nbPointsOnEachCircle; // number of points along the circles around each point of the input object (10 by default)
    Data<double> m_radius; // radius of the circles around each point of the input object (1 by default)
    Data<int> m_peak; // if 1 or 2 creates a peak at the end

    container::RadiusContainer* radiusContainer;
protected:

    VecCoord rotatedPoints;

};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
