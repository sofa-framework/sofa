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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETGEOMETRYALGORITHMS_INL

#include <sofa/component/topology/HexahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/HexahedronSetTopology.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;

template< class DataTypes>
HexahedronSetTopology< DataTypes >* HexahedronSetGeometryAlgorithms< DataTypes >::getHexahedronSetTopology() const
{
    return static_cast<HexahedronSetTopology< DataTypes >* > (this->m_basicTopology);
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeHexahedronVolume( const unsigned int /*i*/) const
{
    //HexahedronSetTopology< DataTypes > *topology = getHexahedronSetTopology();
    //HexahedronSetTopologyContainer * container = topology->getHexahedronSetTopologyContainer();
    //const Hexahedron &t = container->getHexahedron(i);
    //const VecCoord& p = *(this->object->getX());
    Real volume=(Real)(0.0); // todo
    return volume;
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeRestHexahedronVolume( const unsigned int /*i*/) const
{
    //HexahedronSetTopology< DataTypes > *topology = getHexahedronSetTopology();
    //HexahedronSetTopologyContainer * container = topology->getHexahedronSetTopologyContainer();
    //const Hexahedron &t = container->getHexahedron(i);
    //const VecCoord& p = *(this->object->getX0());
    Real volume=(Real)(0.0); // todo
    return volume;
}

template<class DataTypes>
void HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronVolume( BasicArrayInterface<Real> &ai) const
{
    HexahedronSetTopology< DataTypes > *topology = getHexahedronSetTopology();
    HexahedronSetTopologyContainer * container = topology->getHexahedronSetTopologyContainer();
    //const sofa::helper::vector<Hexahedron> &ta=container->getHexahedronArray();
    //const typename DataTypes::VecCoord& p = *(this->object->getX());
    for(unsigned int i=0; i<container->getNumberOfHexahedra(); ++i)
    {
        //const Hexahedron &t=container->getHexahedron(i); //ta[i];
        ai[i]=(Real)(0.0); // todo
    }
}

/// Cross product for 3-elements vectors.
template<typename real>
inline real tripleProduct(const Vec<3,real>& a, const Vec<3,real>& b,const Vec<3,real> &c)
{
    return dot(a,cross(b,c));
}

/// area from 2-elements vectors.
template <typename real>
inline real tripleProduct(const Vec<2,real>& , const Vec<2,real>& , const Vec<2,real> &)
{
    assert(false);
    return (real)0;
}

/// area for 1-elements vectors.
template <typename real>
inline real tripleProduct(const Vec<1,real>& , const Vec<1,real>& , const Vec<1,real> &)
{
    assert(false);
    return (real)0;
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_HexahedronSetTOPOLOGY_INL
