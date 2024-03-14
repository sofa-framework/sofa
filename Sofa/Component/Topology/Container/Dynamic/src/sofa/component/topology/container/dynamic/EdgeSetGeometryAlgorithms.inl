/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Jacobi>

#include <sofa/component/topology/container/dynamic/EdgeSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/MatEigen.h>
#include <sofa/type/Mat_solve_Cholesky.h>
#include <sofa/component/topology/container/dynamic/CommonAlgorithms.h>
#include <sofa/component/topology/container/dynamic/PointSetGeometryAlgorithms.inl>

namespace sofa::component::topology::container::dynamic
{

template< class DataTypes>
NumericalIntegrationDescriptor<typename EdgeSetGeometryAlgorithms< DataTypes >::Real,1> &EdgeSetGeometryAlgorithms< DataTypes >::getEdgeNumericalIntegrationDescriptor()
{
    // initialize the cubature table only if needed.
    if (initializedEdgeCubatureTables==false) {
        initializedEdgeCubatureTables=true;
        defineEdgeCubaturePoints();
    }
    return edgeNumericalIntegration;
}

template< class DataTypes>
void EdgeSetGeometryAlgorithms< DataTypes >::defineEdgeCubaturePoints() {
    typedef typename NumericalIntegrationDescriptor<typename EdgeSetGeometryAlgorithms< DataTypes >::Real,1>::QuadraturePoint QuadraturePoint;
    typedef typename NumericalIntegrationDescriptor<typename EdgeSetGeometryAlgorithms< DataTypes >::Real,1>::BarycentricCoordinatesType BarycentricCoordinatesType;

    // Gauss Legendre method : low  number of integration points for a given order
    // for order > 5 no closed form expression exists and therefore use values from http://www.holoborodko.com/pavel/numerical-methods/numerical-integration/#gauss_quadrature_abscissas_table

    typename NumericalIntegrationDescriptor<typename EdgeSetGeometryAlgorithms< DataTypes >::Real,1>::QuadratureMethod m=NumericalIntegrationDescriptor<typename EdgeSetGeometryAlgorithms< DataTypes >::Real,1>::GAUSS_LEGENDRE_METHOD;
    typename NumericalIntegrationDescriptor<typename EdgeSetGeometryAlgorithms< DataTypes >::Real,1>::QuadraturePointArray qpa;
    BarycentricCoordinatesType v;
    Real div2 = 0.5;

    /// integration with linear accuracy.
    v=BarycentricCoordinatesType(0.5);
    qpa.push_back(QuadraturePoint(v,(Real)1.0));
    edgeNumericalIntegration.addQuadratureMethod(m,1,qpa);
    /// integration with quadratic accuracy.
    qpa.clear();
    
    Real a= div2 +1/(2*sqrt((Real)3));
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v, div2));
    Real b=div2-1/(2*sqrt((Real)3.));
    v=BarycentricCoordinatesType(b);
    qpa.push_back(QuadraturePoint(v, div2));
    edgeNumericalIntegration.addQuadratureMethod(m,2,qpa);
    /// integration with cubic accuracy.
    qpa.clear();
    a=div2*(1-sqrt((Real)3/5));
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,(Real)(5.0/18.0)));
    b=div2*(1+sqrt((Real)3/5));
    v=BarycentricCoordinatesType(b);
    qpa.push_back(QuadraturePoint(v,(Real)(5.0/18.0)));
    v=BarycentricCoordinatesType(div2);
    qpa.push_back(QuadraturePoint(v,(Real)(8.0/18.0)));
    edgeNumericalIntegration.addQuadratureMethod(m,3,qpa);
    /// integration with quartic accuracy.
    qpa.clear();
    a=div2*(1-sqrt((Real)(3+2*sqrt(6.0/5.0))/7));
    v=BarycentricCoordinatesType(a);
    Real a2= 0.25f - (Real)sqrt(5.0/6.0)/12;
    qpa.push_back(QuadraturePoint(v,a2));
    a=div2*(1+sqrt((Real)(3+2*sqrt(6.0/5.0))/7));
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,a2));
    a=div2*(1-sqrt((Real)(3-2*sqrt(6.0/5.0))/7));
    a2= 0.25f + (Real)sqrt(5.0/6.0)/12;
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,a2));
    a=div2*(1+sqrt((Real)(3-2*sqrt(6.0/5.0))/7));
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,a2));
    edgeNumericalIntegration.addQuadratureMethod(m,4,qpa);
    /// integration with quintic accuracy.
    qpa.clear();
    a=div2*(1-sqrt((Real)(5+2*sqrt(10.0/7.0)))/3);
    v=BarycentricCoordinatesType(a);
    a2= (Real)(322-13*sqrt(70.0))/900;
    qpa.push_back(QuadraturePoint(v,a2/2));
    a=div2*(1+sqrt((Real)(5+2*sqrt(10.0/7.0)))/3);
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,a2/2));

    a=div2*(1-sqrt((Real)(5-2*sqrt(10.0/7.0)))/3);
    v=BarycentricCoordinatesType(a);
    a2= (Real)(322+13*sqrt(70.0))/900;
    qpa.push_back(QuadraturePoint(v,a2/2));
    a=div2*(1+sqrt((Real)(5-2*sqrt(10.0/7.0)))/3);
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,a2/2));

    v=BarycentricCoordinatesType(div2);
    qpa.push_back(QuadraturePoint(v,(Real)(512/1800.0)));
    edgeNumericalIntegration.addQuadratureMethod(m,5,qpa);

    /// integration with  accuracy of order 6.
    /// no closed form expression
    // copy values for integration in [-1;1] and translate it for integration in [0;1]
    Real varray[6];
    Real warray[6],warray0;
    size_t nbIPs=3;
//	size_t order=6;
    size_t i;

    qpa.clear();
    varray[0] = static_cast<Real>(0.2386191860831969086305017); warray[0] = static_cast<Real>(0.4679139345726910473898703);
    varray[1] = static_cast<Real>(0.6612093864662645136613996); warray[1] = static_cast<Real>(0.3607615730481386075698335);
    varray[2] = static_cast<Real>(0.9324695142031520278123016); warray[2] = static_cast<Real>(0.1713244923791703450402961);

    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(div2*(1+ varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
        v=BarycentricCoordinatesType(div2*(1- varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
    }
    edgeNumericalIntegration.addQuadratureMethod(m,6,qpa);
    /// integration with  accuracy of order 7.
    qpa.clear();
    warray0 = static_cast<Real>(0.4179591836734693877551020);
    varray[0] = static_cast<Real>(0.4058451513773971669066064);	warray[0] = static_cast<Real>(0.3818300505051189449503698);
    varray[1] = static_cast<Real>(0.7415311855993944398638648);	warray[1] = static_cast<Real>(0.2797053914892766679014678);
    varray[2] = static_cast<Real>(0.9491079123427585245261897);	warray[2] = static_cast<Real>(0.1294849661688696932706114);

    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(div2*(1+ varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
        v=BarycentricCoordinatesType(div2*(1- varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
    }
    v=BarycentricCoordinatesType(div2);
    qpa.push_back(QuadraturePoint(v,warray0/2));
    edgeNumericalIntegration.addQuadratureMethod(m,7,qpa);
    /// integration with  accuracy of order 8.
    qpa.clear();
    varray[0]= static_cast<Real>(0.1834346424956498049394761); warray[0]= static_cast<Real>(0.3626837833783619829651504);
    varray[1]= static_cast<Real>(0.5255324099163289858177390); warray[1]= static_cast<Real>(0.3137066458778872873379622);
    varray[2]= static_cast<Real>(0.7966664774136267395915539); warray[2]= static_cast<Real>(0.2223810344533744705443560);
    varray[3]= static_cast<Real>(0.9602898564975362316835609); warray[3]= static_cast<Real>(0.1012285362903762591525314);
    nbIPs=4;


    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(div2*(1+ varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
        v=BarycentricCoordinatesType(div2*(1- varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
    }
    edgeNumericalIntegration.addQuadratureMethod(m,8,qpa);
    /// integration with  accuracy of order 9
    qpa.clear();
    warray0= static_cast<Real>(0.3302393550012597631645251);
    varray[0]= static_cast<Real>(0.3242534234038089290385380);	warray[0]=static_cast<Real>(0.3123470770400028400686304);
    varray[1]= static_cast<Real>(0.6133714327005903973087020);	warray[1]=static_cast<Real>(0.2606106964029354623187429);
    varray[2]= static_cast<Real>(0.8360311073266357942994298);	warray[2]=static_cast<Real>(0.1806481606948574040584720);
    varray[3]= static_cast<Real>(0.9681602395076260898355762);	warray[3]=static_cast<Real>(0.0812743883615744119718922);


    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(div2*(1+ varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
        v=BarycentricCoordinatesType(div2*(1- varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
    }
    v=BarycentricCoordinatesType(div2);
    qpa.push_back(QuadraturePoint(v,warray0/2));
    edgeNumericalIntegration.addQuadratureMethod(m,9,qpa);

    /// integration with accuracy of order 10.
    qpa.clear();
    varray[0] = static_cast<Real>(0.1488743389816312108848260); warray[0]= static_cast<Real>(0.2955242247147528701738930);
    varray[1] = static_cast<Real>(0.4333953941292471907992659); warray[1]= static_cast<Real>(0.2692667193099963550912269);
    varray[2] = static_cast<Real>(0.6794095682990244062343274); warray[2]= static_cast<Real>(0.2190863625159820439955349);
    varray[3] = static_cast<Real>(0.8650633666889845107320967); warray[3]= static_cast<Real>(0.1494513491505805931457763);
    varray[4] = static_cast<Real>(0.9739065285171717200779640); warray[4]= static_cast<Real>(0.0666713443086881375935688);
    nbIPs=5;

    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(div2*(1+ varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
        v=BarycentricCoordinatesType(div2*(1- varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
    }
    edgeNumericalIntegration.addQuadratureMethod(m,10,qpa);
    /// integration with accuracy of order 11.
    qpa.clear();
    warray0 = static_cast<Real>(0.2729250867779006307144835);
    varray[0] = static_cast<Real>(0.2695431559523449723315320);	warray[0] = static_cast<Real>(0.2628045445102466621806889);
    varray[1] = static_cast<Real>(0.5190961292068118159257257);	warray[1] = static_cast<Real>(0.2331937645919904799185237);
    varray[2] = static_cast<Real>(0.7301520055740493240934163);	warray[2] = static_cast<Real>(0.1862902109277342514260976);
    varray[3] = static_cast<Real>(0.8870625997680952990751578);	warray[3] = static_cast<Real>(0.1255803694649046246346943);
    varray[4] = static_cast<Real>(0.9782286581460569928039380);	warray[4] = static_cast<Real>(0.0556685671161736664827537);

    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(div2*(1+ varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
        v=BarycentricCoordinatesType(div2*(1- varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
    }
    v=BarycentricCoordinatesType(div2);
    qpa.push_back(QuadraturePoint(v,warray0/2));
    edgeNumericalIntegration.addQuadratureMethod(m,11,qpa);
    /// integration with accuracy of order 12.
    varray[0] = static_cast<Real>(0.1252334085114689154724414);	warray[0] = static_cast<Real>(0.2491470458134027850005624);
    varray[1] = static_cast<Real>(0.3678314989981801937526915);	warray[1] = static_cast<Real>(0.2334925365383548087608499);
    varray[2] = static_cast<Real>(0.5873179542866174472967024);	warray[2] = static_cast<Real>(0.2031674267230659217490645);
    varray[3] = static_cast<Real>(0.7699026741943046870368938);	warray[3] = static_cast<Real>(0.1600783285433462263346525);
    varray[4] = static_cast<Real>(0.9041172563704748566784659);	warray[4] = static_cast<Real>(0.1069393259953184309602547);
    varray[5] = static_cast<Real>(0.9815606342467192506905491);	warray[5] = static_cast<Real>(0.0471753363865118271946160);
    nbIPs=6;

    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(div2*(1+ varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
        v=BarycentricCoordinatesType(div2*(1- varray[i]));
        qpa.push_back(QuadraturePoint(v,warray[i]/2));
    }
    edgeNumericalIntegration.addQuadratureMethod(m,10,qpa);
}
template< class DataTypes>
typename DataTypes::Real EdgeSetGeometryAlgorithms< DataTypes >::computeEdgeLength( const EdgeID i) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
    const Real length = (DataTypes::getCPos(p[e[0]])-DataTypes::getCPos(p[e[1]])).norm();
    return length;
}




template< class DataTypes>
typename DataTypes::Real EdgeSetGeometryAlgorithms< DataTypes >::computeRestEdgeLength( const EdgeID i) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());
    const Real length = (DataTypes::getCPos(p[e[0]])-DataTypes::getCPos(p[e[1]])).norm();
    return length;
}

template< class DataTypes>
typename DataTypes::Real EdgeSetGeometryAlgorithms< DataTypes >::computeRestSquareEdgeLength( const EdgeID i) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());
    const Real length = (DataTypes::getCPos(p[e[0]])-DataTypes::getCPos(p[e[1]])).norm2();
    return length;
}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeLength( BasicArrayInterface<Real> &ai) const
{
    const sofa::type::vector<Edge> &ea = this->m_topology->getEdges();
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    for (Index i=0; i<ea.size(); ++i)
    {
        const Edge &e = ea[i];
        ai[i] = (DataTypes::getCPos(p[e[0]])-DataTypes::getCPos(p[e[1]])).norm();
    }
}

template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeAABB(const EdgeID i, CPos& minCoord, CPos& maxCoord) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
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
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    pnt[0] = p[e[0]];
    pnt[1] = p[e[1]];
}

template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::getRestEdgeVertexCoordinates(const EdgeID i, Coord pnt[2]) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());

    pnt[0] = p[e[0]];
    pnt[1] = p[e[1]];
}

template<class DataTypes>
typename DataTypes::Coord EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeCenter(const EdgeID i) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    return (p[e[0]] + p[e[1]]) * (Real) 0.5;
}

template<class DataTypes>
typename DataTypes::Coord EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeDirection(const EdgeID i) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
    return (p[e[1]] - p[e[0]]);
}

template<class DataTypes>
typename DataTypes::Coord EdgeSetGeometryAlgorithms<DataTypes>::computeRestEdgeDirection(const EdgeID i) const
{
    const Edge &e = this->m_topology->getEdge(i);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());
    return (p[e[1]] - p[e[0]]);
}

// test if a point is on the triangle indexed by ind_e
template<class DataTypes>
bool EdgeSetGeometryAlgorithms<DataTypes>::isPointOnEdge(const sofa::type::Vec<3,Real> &pt, const EdgeID ind_e) const
{
    Coord vertices[2];
    getEdgeVertexCoordinates(ind_e, vertices);
    sofa::type::Vec<3, Real> p1(type::NOINIT), p2(type::NOINIT);

    DataTypes::get(p1[0], p1[1], p1[2], vertices[0]);
    DataTypes::get(p2[0], p2[1], p2[2], vertices[1]);
    
    return sofa::geometry::Edge::isPointOnEdge(pt, p1, p2);
}


template<class DataTypes>
auto EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeBarycentricCoordinates(
    const sofa::type::Vec<3, Real> &p,
    PointID ind_p1, PointID ind_p2, bool useRestPosition) const -> sofa::type::vector< SReal >
{
    sofa::core::ConstVecCoordId::MyVecId _vecId = useRestPosition ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();

    const typename DataTypes::VecCoord& vect_c = (this->object->read(_vecId)->getValue());
    const typename DataTypes::Coord& c0 = vect_c[ind_p1];
    const typename DataTypes::Coord& c1 = vect_c[ind_p2];

    sofa::type::Vec<3, Real> a; DataTypes::get(a[0], a[1], a[2], c0);
    sofa::type::Vec<3, Real> b; DataTypes::get(b[0], b[1], b[2], c1);

    sofa::type::Vec<2, Real> coefs = sofa::geometry::Edge::getBarycentricCoordinates(p, a, b);
    sofa::type::vector< SReal > baryCoefs;

    baryCoefs.push_back(coefs[0]);
    baryCoefs.push_back(coefs[1]);

    return baryCoefs;
}


template<class Vec>
bool is_point_on_edge(const Vec& p, const Vec& a, const Vec& b)
{
    const typename Vec::value_type ZERO = 1e-12;
    Vec v = (p - a).cross(p - b);

    if(v.norm2() < ZERO)
        return true;
    else
        return false;
}

template<class DataTypes>
auto EdgeSetGeometryAlgorithms<DataTypes>::computePointProjectionOnEdge (const EdgeID edgeIndex,
        sofa::type::Vec<3, Real> c,
        bool& intersected) -> sofa::type::vector< SReal >
{

    // Compute projection point coordinate H using parametric straight lines equations.
    //
    //            C                          - Compute vector orthogonal to (ABX), then vector collinear to (XH)
    //          / .                          - Solve the equation system of straight lines intersection
    //        /   .                          - Compute H real coordinates
    //      /     .                          - Compute H bary coef on AB
    //    /       .
    //   A ------ H -------------B


    Coord coord_AB, coord_AC;
    Coord coord_edge1[2], coord_edge2[2];

    // Compute Coord of first edge AB:
    Edge theEdge = this->m_topology->getEdge (edgeIndex);
    getEdgeVertexCoordinates (edgeIndex, coord_edge1);
    coord_AB = coord_edge1[1] - coord_edge1[0];

    // Compute Coord of tmp vector AC:
    DataTypes::add (coord_edge2[0], c[0], c[1], c[2]);
    coord_AC = coord_edge2[0] - coord_edge1[0];

    // Compute Coord of second edge XH:

    sofa::type::Vec<3, Real> AB; DataTypes::get(AB[0], AB[1], AB[2], coord_AB);
    sofa::type::Vec<3, Real> AC; DataTypes::get(AC[0], AC[1], AC[2], coord_AC);
    sofa::type::Vec<3, Real> ortho_ABC = cross (AB, AC)*1000;
    sofa::type::Vec<3, Real> coef_CH = cross (ortho_ABC, AB)*1000;

    for (unsigned int i = 0; i<Coord::spatial_dimensions; i++)
        coord_edge2[1][i] = coord_edge2[0][i] + (float)coef_CH[i];

    // Compute Coord of projection point H:
    Coord coord_H = compute2EdgesIntersection ( coord_edge1, coord_edge2, intersected);
    sofa::type::Vec<3, Real> h; DataTypes::get(h[0], h[1], h[2], coord_H);

    return computeEdgeBarycentricCoordinates(h, theEdge[0], theEdge[1]);
}

template<class DataTypes>
bool EdgeSetGeometryAlgorithms<DataTypes>::computeEdgePlaneIntersection (EdgeID edgeID, sofa::type::Vec<3,Real> pointOnPlane, sofa::type::Vec<3,Real> normalOfPlane, sofa::type::Vec<3,Real>& intersection)
{
    const Edge &e = this->m_topology->getEdge(edgeID);
    const VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    sofa::type::Vec<3,Real> p1,p2;
    p1[0]=p[e[0]][0]; p1[1]=p[e[0]][1]; p1[2]=p[e[0]][2];
    p2[0]=p[e[1]][0]; p2[1]=p[e[1]][1]; p2[2]=p[e[1]][2];

    //plane equation
    normalOfPlane.normalize();
    Real d=normalOfPlane*pointOnPlane;
    Real t=(d-normalOfPlane*p1)/(normalOfPlane*(p2-p1));

    if((t<1)&&(t>=0))
    {
        intersection=p1+(p2-p1)*t;
        return true;
    }
    else
        return false;

}

template<class DataTypes>
bool EdgeSetGeometryAlgorithms<DataTypes>::computeRestEdgePlaneIntersection (EdgeID edgeID, sofa::type::Vec<3,Real> pointOnPlane, sofa::type::Vec<3,Real> normalOfPlane, sofa::type::Vec<3,Real>& intersection)
{
    const Edge &e = this->m_topology->getEdge(edgeID);
    const VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());

    sofa::type::Vec<3,Real> p1,p2;
    p1[0]=p[e[0]][0]; p1[1]=p[e[0]][1]; p1[2]=p[e[0]][2];
    p2[0]=p[e[1]][0]; p2[1]=p[e[1]][1]; p2[2]=p[e[1]][2];

    //plane equation
    normalOfPlane.normalize();
    Real d=normalOfPlane*pointOnPlane;
    Real t=(d-normalOfPlane*p1)/(normalOfPlane*(p2-p1));

    if((t<1)&&(t>=0))
    {
        intersection=p1+(p2-p1)*t;
        return true;
    }
    else
        return false;

}

template<class DataTypes>
typename DataTypes::Coord EdgeSetGeometryAlgorithms<DataTypes>::compute2EdgesIntersection (const Coord edge1[2], const Coord edge2[2], bool& intersected)
{

    // Creating director vectors:
    Coord vec1 = edge1[1] - edge1[0];
    Coord vec2 = edge2[1] - edge2[0];
    Coord X;
    for (unsigned int i=0; i<Coord::spatial_dimensions; i++)
        X[i] = 0;

    int ind1 = -1;
    int ind2 = -1;
    constexpr Real epsilon = static_cast<Real>(0.0001);
    Real lambda = 0.0;
    Real alpha = 0.0;

    // Searching vector composante not null:
    for (unsigned int i=0; i<Coord::spatial_dimensions; i++)
    {
        if ( (vec1[i] > epsilon) || (vec1[i] < -epsilon) )
        {
            ind1 = i;

            for (unsigned int j = 0; j<Coord::spatial_dimensions; j++)
                if ( (vec2[j] > epsilon || vec2[j] < -epsilon) && (j != i))
                {
                    ind2 = j;

                    // Solving system:
                    Real coef_lambda = vec1[ind1] - ( vec1[ind2]*vec2[ind1]/vec2[ind2] );

                    if (coef_lambda < epsilon && coef_lambda > -epsilon)
                        break;

                    lambda = ( edge2[0][ind1] - edge1[0][ind1] + (edge1[0][ind2] - edge2[0][ind2])*vec2[ind1]/vec2[ind2]) * 1/coef_lambda;
                    alpha = (edge1[0][ind2] + lambda * vec1[ind2] - edge2[0][ind2]) * 1 /vec2[ind2];
                    break;
                }
        }

        if (lambda != 0.0)
            break;
    }

    if ((ind1 == -1) || (ind2 == -1))
    {
        msg_error() << "Vector director is null." ;
        intersected = false;
        return X;
    }

    // Compute X coords:
    for (unsigned int i = 0; i<Coord::spatial_dimensions; i++)
        X[i] = edge1[0][i] + (float)lambda * vec1[i];

    intersected = true;

    // Check if lambda found is really a solution
    for (unsigned int i = 0; i<Coord::spatial_dimensions; i++)
        if ( (X[i] - edge2[0][i] - alpha * vec2[i]) > 0.1 )
        {
            msg_error() << "Edges don't intersect themself." ;
            intersected = false;
        }

    return X;
}


template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    PointSetGeometryAlgorithms<DataTypes>::draw(vparams);

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    // Draw Edges indices
    if (showEdgeIndices.getValue() && this->m_topology->getNbEdges() != 0)
    {        
        const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
        float scale = this->getIndicesScale();

        //for edges:
        scale = scale/2;

        const sofa::type::vector<Edge>& edgeArray = this->m_topology->getEdges();

        std::vector<type::Vec3> positions;
        for (size_t i = 0; i < edgeArray.size(); i++)
        {

            Edge the_edge = edgeArray[i];
            Coord vertex1 = coords[the_edge[0]];
            Coord vertex2 = coords[the_edge[1]];
            type::Vec3 center;
            center = (DataTypes::getCPos(vertex1) + DataTypes::getCPos(vertex2)) / 2;

            positions.push_back(center);
        }
        vparams->drawTool()->draw3DText_Indices(positions, scale, _drawColor.getValue());
    }


    // Draw edges
    if (d_drawEdges.getValue() && this->m_topology->getNbEdges() != 0)
    {
        const sofa::type::vector<Edge> &edgeArray = this->m_topology->getEdges();

        const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());

        std::vector<type::Vec3> positions;
        positions.reserve(edgeArray.size()*2u);
        for (size_t i = 0; i<edgeArray.size(); i++)
        {
            const Edge& e = edgeArray[i];
            positions.push_back(type::Vec3(DataTypes::getCPos(coords[e[0]])));
            positions.push_back(type::Vec3(DataTypes::getCPos(coords[e[1]])));
        }
        vparams->drawTool()->drawLines(positions, 1.0f, _drawColor.getValue());
        vparams->drawTool()->drawPoints(positions, 4.0f, _drawColor.getValue());
    }


}



template< class DataTypes>
void EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights( type::vector<sofa::Index>& numEdges, type::vector<Edge>& vertexEdges, type::vector<Vec3d>& weights ) const
{
    const VecCoord& pos =(this->object->read(core::ConstVecCoordId::position())->getValue()); // point positions

    sofa::type::vector<sofa::type::Vec<3, Real> > edgeVec;                  // 3D edges

    numEdges.clear();
    vertexEdges.clear();
    weights.clear();

    const SeqEdges& edges = this->m_topology->getEdges();

    for(PointID pointId=0; pointId<pos.size(); pointId++ )
    {
        EdgesAroundVertex ve = this->m_topology->getEdgesAroundVertex(pointId);
        edgeVec.resize(ve.size());
        numEdges.push_back(sofa::Size(ve.size()));            // number of edges attached to this point
        sofa::type::Mat<3, 3, Real> EEt, L;

        // Solve E.W = I , where each column of E is an adjacent edge vector, W are the desired weights, and I is the 3x3 identity
        // Each row of W corresponds to an edge, and encode the contribution of the edge to the basis vectors x,y,z
        // To solve this underconstrained system, we assume that W = Et.U , where Et is the transpose of E and U is 3x3
        // We solve (E.Et).U = I , then we compute W = Et.U
        // todo: weight the edges according to their lengths

        // compute E.Et
        for(Size e=0; e<ve.size(); e++ )
        {
            Edge edge = edges[ve[e]];
            vertexEdges.push_back(edge);              // concatenate
            const CPos& p0 = DataTypes::getCPos(pos[edge[0]]);
            const CPos& p1 = DataTypes::getCPos(pos[edge[1]]);
            edgeVec[e] = p1 - p0;
            // each edge vector adds e.et to the matrix
            for(unsigned j=0; j<3; j++)
                for(unsigned k=0; k<3; k++)
                    EEt[j][k] += edgeVec[e][k]*edgeVec[e][j];
        }

        // decompose E.Et for system solution
        if( cholDcmp(L,EEt) ) // Cholesky decomposition of the covariance matrix succeeds, we use it to solve the systems
        {
            const size_t n = weights.size();     // start index for this vertex
            weights.resize( n + ve.size() ); // concatenate all the W of the nodes
            sofa::type::Vec<3, Real> a,u;

            // axis x
            a = { 1,0,0 };
            cholBksb(u,L,a); // solve EEt.u=x using the Cholesky decomposition
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][0] = u * edgeVec[i];
            }

            // axis y
            a = { 0,1,0 };
            cholBksb(u,L,a); // solve EEt.u=y using the Cholesky decomposition
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][1] = u * edgeVec[i];
            }

            // axis z
            a = { 0,0,1 };
            cholBksb(u,L,a); // solve EEt.u=z using the Cholesky decomposition
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][2] = u * edgeVec[i];
            }
        }
        else
        {
            const size_t n = weights.size();     // start index for this vertex
            weights.resize( n + ve.size() ); // concatenate all the W of the nodes
            sofa::type::Vec<3, Real> a,u;

            typedef Eigen::Matrix<Real,3,3> EigenM33;
            EigenM33 emat = helper::eigenMat(EEt);
            Eigen::JacobiSVD<EigenM33> jacobi(emat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix<Real,3,1> solution;

            // axis x
            a = { 1,0,0 };
            solution = jacobi.solve( helper::eigenVec(a) );
            // least-squares solve EEt.u=x
            for(int i=0; i<3; i++)
                u[i] = solution(i);
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][0] = u * edgeVec[i];
            }

            // axis y
            a = { 0,1,0 };
            solution = jacobi.solve(helper::eigenVec(a) );
            // least-squares solve EEt.u=y
            for(int i=0; i<3; i++)
                u[i] = solution(i);
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][1] = u * edgeVec[i];
            }

            // axis z
            a = { 0,0,1 };
            solution = jacobi.solve(helper::eigenVec(a) );
            // least-squares solve EEt.u=z
            for(int i=0; i<3; i++)
                u[i] = solution(i);
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][2] = u * edgeVec[i];
            }
        }

    }
}


template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::initPointAdded(PointID index, const core::topology::PointAncestorElem &ancestorElem
        , const type::vector< VecCoord* >& coordVecs, const type::vector< VecDeriv* >& derivVecs)
{
    using namespace sofa::core::topology;

    if (ancestorElem.type != geometry::ElementType::EDGE)
    {
        PointSetGeometryAlgorithms< DataTypes >::initPointAdded(index, ancestorElem, coordVecs, derivVecs);
    }
    else
    {
        const Edge &e = this->m_topology->getEdge(ancestorElem.index);

        for (unsigned int i = 0; i < coordVecs.size(); i++)
        {
            VecCoord &curVecCoord = *coordVecs[i];
            Coord& curCoord = curVecCoord[index];

            const Coord &c0 = curVecCoord[e[0]];
            const Coord &c1 = curVecCoord[e[1]];

            curCoord = c0 + (c1-c0) * ancestorElem.localCoords[0];
        }
    }
}

template<class DataTypes>
bool EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeSegmentIntersection(EdgeID edgeID,
    const sofa::type::Vec<3, Real>& a,
    const sofa::type::Vec<3, Real>& b,
    Real &baryCoef)
{
    bool is_intersect = false;
    
    const Edge& e = this->m_topology->getEdge(edgeID);
    const VecCoord& p = (this->object->read(core::ConstVecCoordId::position())->getValue());
    const typename DataTypes::Coord& c0 = p[e[0]];
    const typename DataTypes::Coord& c1 = p[e[1]];
    
    sofa::type::Vec<3, Real> p0{ c0[0],c0[1],c0[2] };
    sofa::type::Vec<3, Real> p1{ c1[0],c1[1],c1[2] };
    sofa::type::Vec<3, Real> pa{ a[0],a[1],a[2] };
    sofa::type::Vec<3, Real> pb{ b[0],b[1],b[2] };
  
    sofa::type::Vec<3, Real> v_0a = p0 - pa;
    sofa::type::Vec<3, Real> v_ba = pb - pa;
    sofa::type::Vec<3, Real> v_10 = p1 - p0;
  
    Real d0aba, dba10, d0a10, dbaba, d1010;

    d0aba = v_0a * v_ba;
    dba10 = v_ba * v_ba;
    d0a10 = v_0a * v_10;
    dbaba = v_ba * v_ba;
    d1010 = v_10 * v_10;

    Real deno, num;
    deno = d1010 * dbaba - dba10 * dba10;
    
    if (abs(deno) > std::numeric_limits<Real>::epsilon())
    {
        num = d0aba * dba10 - d0a10 * dbaba;

        baryCoef = num / deno;
        is_intersect = true;
    }
    return is_intersect;
}

template <class DataTypes>
bool EdgeSetGeometryAlgorithms<DataTypes>::mustComputeBBox() const
{
    return ( (this->m_topology->getNbEdges() != 0 && (d_drawEdges.getValue() || showEdgeIndices.getValue())) || Inherit1::mustComputeBBox() );
}


template <class DataTypes>
sofa::type::vector< SReal > EdgeSetGeometryAlgorithms<DataTypes>::compute2PointsBarycoefs(const sofa::type::Vec<3, Real> &p, PointID ind_p1, PointID ind_p2) const
{
    return computeEdgeBarycentricCoordinates(p, ind_p1, ind_p2);
}


template <class DataTypes>
sofa::type::vector< SReal > EdgeSetGeometryAlgorithms<DataTypes>::computeRest2PointsBarycoefs(const sofa::type::Vec<3, Real> &p, PointID ind_p1, PointID ind_p2) const
{
    return computeEdgeBarycentricCoordinates(p, ind_p1, ind_p2, true);
}


} //namespace sofa::component::topology::container::dynamic
