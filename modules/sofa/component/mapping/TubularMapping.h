/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_TUBULARMAPPING_H
#define SOFA_COMPONENT_MAPPING_TUBULARMAPPING_H

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <vector>

#include <sofa/component/topology/TriangleSetTopology.h>

namespace sofa
{

namespace component
{

namespace mapping
{



template <class BasicMapping>
class TubularMapping : public BasicMapping, public virtual core::objectmodel::BaseObject
{
public:
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::SparseDeriv InSparseDeriv;
    typedef typename Coord::value_type Real;
    enum { M=Coord::static_size };
    typedef defaulttype::Mat<M,M,Real> Mat;

    typedef defaulttype::Vec<M,Real> Vec;

    TubularMapping ( In* from, Out* to )
        : Inherit ( from, to )
        , m_nbPointsOnEachCircle( initData(&m_nbPointsOnEachCircle, "nbPointsOnEachCircle", "Discretization of created circles"))
        , m_radius( initData(&m_radius, "radius", "Radius of created circles"))
    {
        if(m_nbPointsOnEachCircle == NULL)
        {
            m_nbPointsOnEachCircle = 10; // number of points along the circles around each point of the input object (10 by default)
        }
        if(m_radius == NULL)
        {
            m_radius = 1.0; // radius of the circles around each point of the input object (1 by default)
        }
    }

    virtual ~TubularMapping()
    {}

    void init();

    virtual void apply ( typename Out::VecCoord& out, const typename In::VecCoord& in );

    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    virtual void applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    //void applyJT ( typename In::VecConst& out, const typename Out::VecConst& in );

    Data<unsigned int> m_nbPointsOnEachCircle;
    Data<double> m_radius;

protected:

    VecCoord rotatedPoints;

};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
