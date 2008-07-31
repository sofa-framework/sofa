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
#include <sofa/defaulttype/VecTypes.h>

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
    this->getContext()->get(m_container);
}

template <class DataTypes>
typename DataTypes::Coord PointSetGeometryAlgorithms<DataTypes>::getPointSetCenter() const
{
    typename DataTypes::Coord center;
    // get restPosition
    typename DataTypes::VecCoord& p = *(object->getX0());

    const unsigned int numVertices = m_container->getNbPoints();
    for(unsigned int i=0; i<numVertices; ++i)
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
    // get restPosition
    typename DataTypes::VecCoord& p = *(object->getX0());

    const unsigned int numVertices = m_container->getNbPoints();
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
    // get restPosition
    typename DataTypes::VecCoord& p = *(object->getX0());

    bb[0] = (Real) p[0][0];
    bb[1] = (Real) p[0][1];
    bb[2] = (Real) p[0][2];
    bb[3] = (Real) p[0][0];
    bb[4] = (Real) p[0][1];
    bb[5] = (Real) p[0][2];

    for(unsigned int i=1; i<p.size(); ++i)
    {
        // min
        if(bb[0] > (Real) p[i][0]) bb[0] = (Real) p[i][0];	// x
        if(bb[1] > (Real) p[i][1]) bb[1] = (Real) p[i][1];	// y
        if(bb[2] > (Real) p[i][2]) bb[2] = (Real) p[i][2];	// z

        // max
        if(bb[3] < (Real) p[i][0]) bb[3] = (Real) p[i][0];	// x
        if(bb[4] < (Real) p[i][1]) bb[4] = (Real) p[i][1];	// y
        if(bb[5] < (Real) p[i][2]) bb[5] = (Real) p[i][2];	// z
    }
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif
