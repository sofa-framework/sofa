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
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_QUADSETGEOMETRYALGORITHMS_INL

#include <sofa/component/topology/QuadSetGeometryAlgorithms.h>
#include <sofa/component/topology/QuadSetTopology.h>

namespace sofa
{
namespace component
{
namespace topology
{
using namespace sofa::defaulttype;

template <class DataTypes>
void QuadSetGeometryAlgorithms<DataTypes>::init()
{
    EdgeSetGeometryAlgorithms<DataTypes>::init();
    this->getContext()->get(m_container);
}

template< class DataTypes>
typename DataTypes::Real QuadSetGeometryAlgorithms< DataTypes >::computeQuadArea( const unsigned int i) const
{
    const Quad &t = m_container->getQuad(i);
    const VecCoord& p = *(this->object->getX());
    Real area = (Real)((areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])
            + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]])) * (Real) 0.5);
    return area;
}

template< class DataTypes>
typename DataTypes::Real QuadSetGeometryAlgorithms< DataTypes >::computeRestQuadArea( const unsigned int i) const
{
    const Quad &t = m_container->getQuad(i);
    const VecCoord& p = *(this->object->getX0());
    Real area = (Real)((areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])
            + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]])) * (Real) 0.5);
    return area;
}

template<class DataTypes>
void QuadSetGeometryAlgorithms<DataTypes>::computeQuadArea( BasicArrayInterface<Real> &ai) const
{
    //const sofa::helper::vector<Quad> &ta=m_container->getQuadArray();
    unsigned int nb_quads = m_container->getNumberOfQuads();
    const typename DataTypes::VecCoord& p = *(this->object->getX());

    for(unsigned int i=0; i<nb_quads; ++i)
    {
        // ta.size()
        const Quad &t = m_container->getQuad(i);  //ta[i];
        ai[i] = (Real)((areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])
                + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]])) * (Real) 0.5);
    }
}

// Computes the normal vector of a quad indexed by ind_q (not normed)
template<class DataTypes>
Vec<3,double> QuadSetGeometryAlgorithms< DataTypes >::computeQuadNormal(const unsigned int ind_q)
{
    // HYP :  The quad indexed by ind_q is planar

    const Quad &q = m_container->getQuad(ind_q);
    const typename DataTypes::VecCoord& vect_c = *(this->object->getX());

    const typename DataTypes::Coord& c0=vect_c[q[0]];
    const typename DataTypes::Coord& c1=vect_c[q[1]];
    const typename DataTypes::Coord& c2=vect_c[q[2]];
    //const typename DataTypes::Coord& c3=vect_c[q[3]];

    Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);
    //Vec<3,Real> p3;
    //p3[0] = (Real) (c3[0]); p3[1] = (Real) (c3[1]); p3[2] = (Real) (c3[2]);

    Vec<3,Real> normal_q=(p1-p0).cross( p2-p0);

    return ((Vec<3,double>) normal_q);
}


// Test if a quad indexed by ind_quad (and incident to the vertex indexed by ind_p) is included or not in the plane defined by (ind_p, plane_vect)
template<class DataTypes>
bool QuadSetGeometryAlgorithms< DataTypes >::is_quad_in_plane(const unsigned int ind_q,
        const unsigned int ind_p,
        const Vec<3,Real>&plane_vect)
{
    const Quad &q = m_container->getQuad(ind_q);

    // HYP : ind_p==q[0] or ind_q==t[1] or ind_q==t[2] or ind_q==q[3]

    const typename DataTypes::VecCoord& vect_c = *(this->object->getX());

    unsigned int ind_1;
    unsigned int ind_2;
    unsigned int ind_3;

    if(ind_p==q[0])
    {
        ind_1=q[1];
        ind_2=q[2];
        ind_3=q[3];
    }
    else if(ind_p==q[1])
    {
        ind_1=q[2];
        ind_2=q[3];
        ind_3=q[0];
    }
    else if(ind_p==q[2])
    {
        ind_1=q[3];
        ind_2=q[0];
        ind_3=q[1];
    }
    else
    {
        // ind_p==q[3]
        ind_1=q[0];
        ind_2=q[1];
        ind_3=q[2];
    }

    const typename DataTypes::Coord& c0 = vect_c[ind_p];
    const typename DataTypes::Coord& c1 = vect_c[ind_1];
    const typename DataTypes::Coord& c2 = vect_c[ind_2];
    const typename DataTypes::Coord& c3 = vect_c[ind_3];

    Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);
    Vec<3,Real> p3;
    p3[0] = (Real) (c3[0]); p3[1] = (Real) (c3[1]); p3[2] = (Real) (c3[2]);

    return((p1-p0)*( plane_vect)>=0.0 && (p2-p0)*( plane_vect)>=0.0 && (p3-p0)*( plane_vect)>=0.0);
}

/// Write the current mesh into a msh file
template <typename DataTypes>
void QuadSetGeometryAlgorithms<DataTypes>::writeMSHfile(const char *filename)
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

    const sofa::helper::vector<Quad>& qa = m_container->getQuadArray();

    myfile << qa.size() <<"\n";

    for(unsigned int i=0; i<qa.size(); ++i)
    {
        myfile << i+1 << " 3 1 1 4 " << qa[i][0]+1 << " " << qa[i][1]+1 << " " << qa[i][2]+1 << " " << qa[i][3]+1 << "\n";
    }

    myfile << "$ENDELM\n";

    myfile.close();
}

/// Cross product for 3-elements vectors.
template< class Real>
Real areaProduct(const Vec<3,Real>& a, const Vec<3,Real>& b)
{
    return Vec<3,Real>(a.y()*b.z() - a.z()*b.y(),
            a.z()*b.x() - a.x()*b.z(),
            a.x()*b.y() - a.y()*b.x()).norm();
}

/// area from 2-elements vectors.
template< class Real>
Real areaProduct(const defaulttype::Vec<2,Real>& a, const defaulttype::Vec<2,Real>& b )
{
    return a[0]*b[1] - a[1]*b[0];
}

/// area for 1-elements vectors.
template< class Real>
Real areaProduct(const defaulttype::Vec<1,Real>& , const defaulttype::Vec<1,Real>&  )
{
    //	assert(false);
    return (Real)0;
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_QUADSETGEOMETRYALGORITHMS_INL
