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

#include <sofa/component/topology/HexahedronSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;

template <class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::init()
{
    QuadSetGeometryAlgorithms< DataTypes >::init();
    m_topology = this->getContext()->getMeshTopology();
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeHexahedronVolume( const unsigned int /*i*/) const
{
    //const Hexahedron &t = m_topology->getHexa(i);
    //const VecCoord& p = *(this->object->getX());
    Real volume=(Real)(0.0); /// @TODO : implementation of computeHexahedronVolume
    return volume;
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeRestHexahedronVolume( const unsigned int /*i*/) const
{
    //const Hexahedron &t = m_topology->getHexa(i);
    //const VecCoord& p = *(this->object->getX0());
    Real volume=(Real)(0.0); /// @TODO : implementation of computeRestHexahedronVolume
    return volume;
}

template<class DataTypes>
void HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronVolume( BasicArrayInterface<Real> &ai) const
{
    //const sofa::helper::vector<Hexahedron> &ta=m_topology->getHexas();
    //const typename DataTypes::VecCoord& p = *(this->object->getX());
    for(int i=0; i<m_topology->getNbHexas(); ++i)
    {
        //const Hexahedron &t=m_topology->getHexa(i); //ta[i];
        ai[i]=(Real)(0.0); /// @TODO : implementation of computeHexahedronVolume
    }
}

/// Write the current mesh into a msh file
template <typename DataTypes>
void HexahedronSetGeometryAlgorithms<DataTypes>::writeMSHfile(const char *filename)
{
    std::ofstream myfile;
    myfile.open (filename);

    const typename DataTypes::VecCoord& vect_c = *(this->object->getX());

    const unsigned int numVertices = vect_c.size();

    myfile << "$NOD\n";
    myfile << numVertices <<"\n";

    for(unsigned int i=0; i<numVertices; ++i)
    {
        double x = (double) vect_c[i][0];
        double y = (double) vect_c[i][1];
        double z = (double) vect_c[i][2];

        myfile << i+1 << " " << x << " " << y << " " << z <<"\n";
    }

    myfile << "$ENDNOD\n";
    myfile << "$ELM\n";

    const sofa::helper::vector<Hexahedron> hea = m_topology->getHexas();

    myfile << hea.size() <<"\n";

    for(unsigned int i=0; i<hea.size(); ++i)
    {
        myfile << i+1 << " 5 1 1 8 " << hea[i][4]+1 << " " << hea[i][5]+1 << " "
                << hea[i][1]+1 << " " << hea[i][0]+1 << " "
                << hea[i][7]+1 << " " << hea[i][6]+1 << " "
                << hea[i][2]+1 << " " << hea[i][3]+1 << "\n";
    }

    myfile << "$ENDELM\n";

    myfile.close();
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
