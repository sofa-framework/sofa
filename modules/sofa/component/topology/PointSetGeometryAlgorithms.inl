/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_INL

#include <sofa/component/topology/PointSetGeometryAlgorithms.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;

template <class DataTypes>
void PointSetGeometryAlgorithms< DataTypes >::init()
{
    object = this->getContext()->core::objectmodel::BaseContext::get< core::componentmodel::behavior::MechanicalState< DataTypes > >();
    core::componentmodel::topology::GeometryAlgorithms::init();
    this->m_topology = this->getContext()->getMeshTopology();
}

template <class DataTypes>
typename DataTypes::Coord PointSetGeometryAlgorithms<DataTypes>::getPointSetCenter() const
{
    typename DataTypes::Coord center;
    // get current positions
    typename DataTypes::VecCoord& p = *(object->getX());

    const int numVertices = this->m_topology->getNbPoints();
    for(int i=0; i<numVertices; ++i)
    {
        center += p[i];
    }

    center /= numVertices;
    return center;
}

template<class DataTypes>
void  PointSetGeometryAlgorithms<DataTypes>::getEnclosingSphere(typename DataTypes::Coord &center,
        typename DataTypes::Real &radius) const
{
    // get current positions
    typename DataTypes::VecCoord& p = *(object->getX());

    const unsigned int numVertices = this->m_topology->getNbPoints();
    for(unsigned int i=0; i<numVertices; ++i)
    {
        center += p[i];
    }
    center /= numVertices;
    radius = (Real) 0;

    for(unsigned int i=0; i<numVertices; ++i)
    {
        const Coord dp = center-p[i];
        const Real val = dot(dp,dp);
        if(val > radius)
            radius = val;
    }
    radius = (Real)sqrt((double) radius);
}

template<class DataTypes>
void  PointSetGeometryAlgorithms<DataTypes>::getAABB(typename DataTypes::Real bb[6] ) const
{
    Coord minCoord, maxCoord;
    getAABB(minCoord, maxCoord);

    bb[0] = minCoord[0];
    bb[1] = minCoord[1];
    bb[2] = minCoord[2];
    bb[3] = maxCoord[0];
    bb[4] = maxCoord[1];
    bb[5] = maxCoord[2];
}

template<class DataTypes>
void PointSetGeometryAlgorithms<DataTypes>::getAABB(Coord& minCoord, Coord& maxCoord) const
{
    // get current positions
    typename DataTypes::VecCoord& p = *(object->getX());

    minCoord = p[0];
    maxCoord = p[0];

    for(unsigned int i=1; i<p.size(); ++i)
    {
        if(minCoord[0] > p[i][0]) minCoord[0] = p[i][0];
        if(minCoord[1] > p[i][1]) minCoord[1] = p[i][1];
        if(minCoord[2] > p[i][2]) minCoord[2] = p[i][2];

        if(maxCoord[0] < p[i][0]) maxCoord[0] = p[i][0];
        if(maxCoord[1] < p[i][1]) maxCoord[1] = p[i][1];
        if(maxCoord[2] < p[i][2]) maxCoord[2] = p[i][2];
    }
}

template<class DataTypes>
const typename DataTypes::Coord& PointSetGeometryAlgorithms<DataTypes>::getPointPosition(const PointID pointId) const
{
    // get current positions
    const typename DataTypes::VecCoord& p = *(object->getX());

    return p[pointId];
}

template<class DataTypes>
const typename DataTypes::Coord& PointSetGeometryAlgorithms<DataTypes>::getPointRestPosition(const PointID pointId) const
{
    // get rest positions
    const typename DataTypes::VecCoord& p = *(object->getX0());

    return p[pointId];
}

template<class DataTypes>
typename PointSetGeometryAlgorithms<DataTypes>::Angle
PointSetGeometryAlgorithms<DataTypes>::computeAngle(PointID ind_p0, PointID ind_p1, PointID ind_p2) const
{
    const double ZERO = 1e-10;
    const typename DataTypes::VecCoord& p = *(object->getX());
    Coord p0 = p[ind_p0];
    Coord p1 = p[ind_p1];
    Coord p2 = p[ind_p2];
    double t = (p1 - p0)*(p2 - p0);

    if(abs(t) < ZERO)
        return RIGHT;
    if(t > 0.0)
        return ACUTE;
    else
        return OBTUSE;
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif
