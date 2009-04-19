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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_INL

#include <sofa/component/topology/EdgeSetGeometryAlgorithms.h>

#include <sofa/component/topology/PointSetGeometryAlgorithms.inl>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;

template< class DataTypes>
typename DataTypes::Real EdgeSetGeometryAlgorithms< DataTypes >::computeEdgeLength( const EdgeID i) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const VecCoord& p = *(this->object->getX());
    const Real length = (DataTypes::getCPos(p[e[0]])-DataTypes::getCPos(p[e[1]])).norm();
    return length;
}

template< class DataTypes>
typename DataTypes::Real EdgeSetGeometryAlgorithms< DataTypes >::computeRestEdgeLength( const EdgeID i) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const VecCoord& p = *(this->object->getX0());
    const Real length = (DataTypes::getCPos(p[e[0]])-DataTypes::getCPos(p[e[1]])).norm();
    return length;
}

template< class DataTypes>
typename DataTypes::Real EdgeSetGeometryAlgorithms< DataTypes >::computeRestSquareEdgeLength( const EdgeID i) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const VecCoord& p = *(this->object->getX0());
    const Real length = (DataTypes::getCPos(p[e[0]])-DataTypes::getCPos(p[e[1]])).norm2();
    return length;
}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeLength( BasicArrayInterface<Real> &ai) const
{
    const sofa::helper::vector<Edge> &ea = this->m_topology->getEdges();
    const typename DataTypes::VecCoord& p = *(this->object->getX());

    for (unsigned int i=0; i<ea.size(); ++i)
    {
        const Edge &e = ea[i];
        ai[i] = (DataTypes::getCPos(p[e[0]])-DataTypes::getCPos(p[e[1]])).norm();
    }
}

template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeAABB(const EdgeID i, CPos& minCoord, CPos& maxCoord) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const typename DataTypes::VecCoord& p = *(this->object->getX());
    const CPos& a = DataTypes::getCPos(p[e[0]]);
    const CPos& b = DataTypes::getCPos(p[e[1]]);
    for (int c=0; c<NC; ++c)
        if (a[c] < b[c]) { minCoord[c] = a[c]; maxCoord[c] = b[c]; }
        else             { minCoord[c] = b[c]; maxCoord[c] = a[c]; }
}

template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::getEdgeVertexCoordinates(const EdgeID i, Coord pnt[2]) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const typename DataTypes::VecCoord& p = *(this->object->getX());

    pnt[0] = p[e[0]];
    pnt[1] = p[e[1]];
}

template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::getRestEdgeVertexCoordinates(const EdgeID i, Coord pnt[2]) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const typename DataTypes::VecCoord& p = *(this->object->getX0());

    pnt[0] = p[e[0]];
    pnt[1] = p[e[1]];
}

template<class DataTypes>
typename DataTypes::Coord EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeCenter(const EdgeID i) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const typename DataTypes::VecCoord& p = *(this->object->getX());

    return (p[e[0]] + p[e[1]]) * (Real) 0.5;
}

template<class DataTypes>
typename DataTypes::Coord EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeDirection(const EdgeID i) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const typename DataTypes::VecCoord& p = *(this->object->getX());
    return (p[e[1]] - p[e[0]]);
}

// test if a point is on the triangle indexed by ind_e
template<class DataTypes>
bool EdgeSetGeometryAlgorithms<DataTypes>::isPointOnEdge(const sofa::defaulttype::Vec<3,double> &pt, const unsigned int ind_e) const
{
    const double ZERO = 1e-12;

    sofa::defaulttype::Vec<3,double> p0 = pt;

    Coord vertices[2];
    getEdgeVertexCoordinates(ind_e, vertices);

    sofa::defaulttype::Vec<3,double> p1; //(vertices[0][0], vertices[0][1], vertices[0][2]);
    sofa::defaulttype::Vec<3,double> p2; //(vertices[1][0], vertices[1][1], vertices[1][2]);
    DataTypes::get(p1[0], p1[1], p1[2], vertices[0]);
    DataTypes::get(p2[0], p2[1], p2[2], vertices[1]);

    sofa::defaulttype::Vec<3,double> v = (p0 - p1).cross(p0 - p2);

    if(v.norm2() < ZERO)
        return true;
    else
        return false;
}

//
template<class DataTypes>
sofa::helper::vector< double > EdgeSetGeometryAlgorithms<DataTypes>::compute2PointsBarycoefs(
    const Vec<3,double> &p,
    unsigned int ind_p1,
    unsigned int ind_p2) const
{
    const double ZERO = 1e-6;

    sofa::helper::vector< double > baryCoefs;

    const typename DataTypes::VecCoord& vect_c = *(this->object->getX());
    const typename DataTypes::Coord& c0 = vect_c[ind_p1];
    const typename DataTypes::Coord& c1 = vect_c[ind_p2];

    Vec<3,double> a; DataTypes::get(a[0], a[1], a[2], c0);
    Vec<3,double> b; DataTypes::get(b[0], b[1], b[2], c1);

    double dis = (b - a).norm();
    double coef_a, coef_b;


    if(dis < ZERO)
    {
        coef_a = 0.5;
        coef_b = 0.5;
    }
    else
    {
        coef_a = (p - b).norm() / dis;
        coef_b = (p - a).norm() / dis;
    }

    baryCoefs.push_back(coef_a);
    baryCoefs.push_back(coef_b);

    return baryCoefs;

}


/// Write the current mesh into a msh file
template <typename DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::writeMSHfile(const char *filename) const
{
    std::ofstream myfile;
    myfile.open (filename);

    const typename DataTypes::VecCoord& vect_c = *(this->object->getX());

    const unsigned int numVertices = vect_c.size();

    myfile << "$NOD\n";
    myfile << numVertices <<"\n";

    for (unsigned int i=0; i<numVertices; ++i)
    {
        double x=0,y=0,z=0; DataTypes::get(x,y,z, vect_c[i]);

        myfile << i+1 << " " << x << " " << y << " " << z <<"\n";
    }

    myfile << "$ENDNOD\n";
    myfile << "$ELM\n";

    const sofa::helper::vector<Edge> &edge = this->m_topology->getEdges();

    myfile << edge.size() <<"\n";

    for (unsigned int i=0; i<edge.size(); ++i)
    {
        myfile << i+1 << " 1 1 1 2 " << edge[i][0]+1 << " " << edge[i][1]+1 <<"\n";
    }

    myfile << "$ENDELM\n";

    myfile.close();
}


template<class Vec>
bool is_point_on_edge(const Vec& p, const Vec& a, const Vec& b)
{
    const double ZERO = 1e-12;
    Vec v = (p - a).cross(p - b);

    if(v.norm2() < ZERO)
        return true;
    else
        return false;
}

template<class Vec>
sofa::helper::vector< double > compute_2points_barycoefs(const Vec& p, const Vec& a, const Vec& b)
{
    const double ZERO = 1e-6;

    sofa::helper::vector< double > baryCoefs;

    double dis = (b - a).norm();
    double coef_a, coef_b;

    if(dis < ZERO)
    {
        coef_a = 0.5;
        coef_b = 0.5;
    }
    else
    {
        coef_a = (p - b).norm() / dis;
        coef_b = (p - a).norm() / dis;
    }

    baryCoefs.push_back(coef_a);
    baryCoefs.push_back(coef_b);

    return baryCoefs;
}


template<class DataTypes>
sofa::helper::vector< double > EdgeSetGeometryAlgorithms<DataTypes>::computePointProjectionOnEdge (const EdgeID edgeIndex, const sofa::defaulttype::Vec<3,double> coord_c)
{

    // Compute projection point coordinate H following the formula : AB*AX = ||AB||.||AH||
    //
    //            X                          - Compute vector orthogonal to (ABX), then vector collinear to (XH)
    //          / .                          - Solve the equation system of straight lines intersection
    //        /   .                          - Compute H real coordinates
    //      /     .                          - Compute H bary coef on AB
    //    /       .
    //   A ------ H -------------B


    Coord coord_AB[2];
    Edge theEdge = this->m_topology->getEdge (edgeIndex);
    getEdgeVertexCoordinates (edgeIndex, coord_AB);

    sofa::defaulttype::Vec<3,double> a; DataTypes::get(a[0], a[1], a[2], coord_AB[0]);
    sofa::defaulttype::Vec<3,double> b; DataTypes::get(b[0], b[1], b[2], coord_AB[1]);
    sofa::defaulttype::Vec<3,double> c = coord_c;
    sofa::defaulttype::Vec<3,double> h;

    sofa::defaulttype::Vec<3,double> AB;
    sofa::defaulttype::Vec<3,double> AC;

    for (unsigned int i = 0; i<3; i++)
    {
        AB[i] = b[i] - a[i];
        AC[i] = c[i] - a[i];
    }

    sofa::defaulttype::Vec<3,double> ortho_ABC = cross (AB, AC);
    sofa::defaulttype::Vec<3,double> coef_XH = cross (ortho_ABC, AB);


    // solving system:
    double coef_lambda = AB[0] - ( AB[1]*coef_XH[0]/coef_XH[1] );
    double lambda = ( c[0] - a[0] + (a[1] - c[1])*coef_XH[0]/coef_XH[1])*1/coef_lambda;
    //double alpha = ( a[1] + lambda * AB[1] - c[1] ) * 1/coef_XH[1];

    for (unsigned int i = 0; i<3; i++)
        h[i] = a[i] + lambda * AB[i];


    sofa::helper::vector< double > barycoord = compute2PointsBarycoefs(h, theEdge[0], theEdge[1]);

    return barycoord;

}


} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_EDGESETGEOMETRYALGORITHMS_INL
