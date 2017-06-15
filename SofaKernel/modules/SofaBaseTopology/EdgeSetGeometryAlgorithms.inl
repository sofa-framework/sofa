/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_INL

#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/MatEigen.h>
#include <sofa/defaulttype/Mat_solve_Cholesky.h>
#include <SofaBaseTopology/CommonAlgorithms.h>
#include <SofaBaseTopology/PointSetGeometryAlgorithms.inl>

namespace sofa
{

namespace component
{

namespace topology
{

/*template<class DataTypes>
    void EdgeSetGeometryAlgorithms< DataTypes>::reinit()
    {
       P
    }
*/
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

    /// integration with linear accuracy.
    v=BarycentricCoordinatesType(0.5);
    qpa.push_back(QuadraturePoint(v,(Real)1.0));
    edgeNumericalIntegration.addQuadratureMethod(m,1,qpa);
    /// integration with quadratic accuracy.
    qpa.clear();
    Real a=0.5+1/(2*sqrt(3.));
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,(Real)0.5));
    Real b=0.5-1/(2*sqrt(3.));
    v=BarycentricCoordinatesType(b);
    qpa.push_back(QuadraturePoint(v,(Real)0.5));
    edgeNumericalIntegration.addQuadratureMethod(m,2,qpa);
    /// integration with cubic accuracy.
    qpa.clear();
    a=0.5*(1-sqrt((Real)3/5.0));
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,(Real)(5.0/18.0)));
    b=0.5*(1+sqrt((Real)3/5.0));
    v=BarycentricCoordinatesType(b);
    qpa.push_back(QuadraturePoint(v,(Real)(5.0/18.0)));
    v=BarycentricCoordinatesType(0.5);
    qpa.push_back(QuadraturePoint(v,(Real)(8.0/18.0)));
    edgeNumericalIntegration.addQuadratureMethod(m,3,qpa);
    /// integration with quartic accuracy.
    qpa.clear();
    a=0.5*(1-sqrt((Real)(3+2*sqrt(6.0/5.0))/7));
    v=BarycentricCoordinatesType(a);
    Real a2=0.25-sqrt(5.0/6.0)/12;
    qpa.push_back(QuadraturePoint(v,a2));
    a=0.5*(1+sqrt((Real)(3+2*sqrt(6.0/5.0))/7));
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,a2));
    a=0.5*(1-sqrt((Real)(3-2*sqrt(6.0/5.0))/7));
    a2=0.25+sqrt(5.0/6.0)/12;
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,a2));
    a=0.5*(1+sqrt((Real)(3-2*sqrt(6.0/5.0))/7));
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,a2));
    edgeNumericalIntegration.addQuadratureMethod(m,4,qpa);
    /// integration with quintic accuracy.
    qpa.clear();
    a=0.5*(1-sqrt((Real)(5+2*sqrt(10.0/7.0)))/3);
    v=BarycentricCoordinatesType(a);
    a2=(322-13*sqrt(70.0))/900;
    qpa.push_back(QuadraturePoint(v,a2/2));
    a=0.5*(1+sqrt((Real)(5+2*sqrt(10.0/7.0)))/3);
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,a2/2));

    a=0.5*(1-sqrt((Real)(5-2*sqrt(10.0/7.0)))/3);
    v=BarycentricCoordinatesType(a);
    a2=(322+13*sqrt(70.0))/900;
    qpa.push_back(QuadraturePoint(v,a2/2));
    a=0.5*(1+sqrt((Real)(5-2*sqrt(10.0/7.0)))/3);
    v=BarycentricCoordinatesType(a);
    qpa.push_back(QuadraturePoint(v,a2/2));

    v=BarycentricCoordinatesType(0.5);
    qpa.push_back(QuadraturePoint(v,(Real)(512/1800.0)));
    edgeNumericalIntegration.addQuadratureMethod(m,5,qpa);

    /// integration with  accuracy of order 6.
    /// no closed form expression
    // copy values for integration in [-1;1] and translate it for integration in [0;1]
    double varray[6];
    double warray[6],warray0;
    size_t nbIPs=3;
//	size_t order=6;
    size_t i;

    qpa.clear();
    varray[0]=0.2386191860831969086305017; warray[0]=0.4679139345726910473898703;
    varray[1]=0.6612093864662645136613996;	warray[1]=0.3607615730481386075698335;
    varray[2]=0.9324695142031520278123016; warray[2]=0.1713244923791703450402961;

    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(0.5*(1+varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
        v=BarycentricCoordinatesType(0.5*(1-varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
    }
    edgeNumericalIntegration.addQuadratureMethod(m,6,qpa);
    /// integration with  accuracy of order 7.
    qpa.clear();
    warray0=0.4179591836734693877551020;
    varray[0]=0.4058451513773971669066064;	warray[0]=0.3818300505051189449503698;
    varray[1]=0.7415311855993944398638648;	warray[1]=0.2797053914892766679014678;
    varray[2]=0.9491079123427585245261897;	warray[2]=0.1294849661688696932706114;

    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(0.5*(1+varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
        v=BarycentricCoordinatesType(0.5*(1-varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
    }
    v=BarycentricCoordinatesType(0.5);
    qpa.push_back(QuadraturePoint(v,(Real)warray0/2));
    edgeNumericalIntegration.addQuadratureMethod(m,7,qpa);
    /// integration with  accuracy of order 8.
    qpa.clear();
    varray[0]=0.1834346424956498049394761; warray[0]=	0.3626837833783619829651504;
    varray[1]=0.5255324099163289858177390; warray[1]=	0.3137066458778872873379622;
    varray[2]=0.7966664774136267395915539; warray[2]=	0.2223810344533744705443560;
    varray[3]=0.9602898564975362316835609; warray[3]=	0.1012285362903762591525314;
    nbIPs=4;


    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(0.5*(1+varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
        v=BarycentricCoordinatesType(0.5*(1-varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
    }
    edgeNumericalIntegration.addQuadratureMethod(m,8,qpa);
    /// integration with  accuracy of order 9
    qpa.clear();
    warray0=	0.3302393550012597631645251;
    varray[0]=0.3242534234038089290385380;	warray[0]=0.3123470770400028400686304;
    varray[1]=0.6133714327005903973087020;	warray[1]=	0.2606106964029354623187429;
    varray[2]=0.8360311073266357942994298;	warray[2]=0.1806481606948574040584720;
    varray[3]=0.9681602395076260898355762;	warray[3]=	0.0812743883615744119718922;


    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(0.5*(1+varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
        v=BarycentricCoordinatesType(0.5*(1-varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
    }
    v=BarycentricCoordinatesType(0.5);
    qpa.push_back(QuadraturePoint(v,(Real)warray0/2));
    edgeNumericalIntegration.addQuadratureMethod(m,9,qpa);

    /// integration with accuracy of order 10.
    qpa.clear();
    varray[0]=0.1488743389816312108848260; warray[0]=	0.2955242247147528701738930;
    varray[1]=0.4333953941292471907992659; warray[1]=	0.2692667193099963550912269;
    varray[2]=0.6794095682990244062343274; warray[2]=	0.2190863625159820439955349;
    varray[3]=0.8650633666889845107320967; warray[3]=	0.1494513491505805931457763;
    varray[4]=0.9739065285171717200779640; warray[4]=	0.0666713443086881375935688;
    nbIPs=5;

    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(0.5*(1+varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
        v=BarycentricCoordinatesType(0.5*(1-varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
    }
    edgeNumericalIntegration.addQuadratureMethod(m,10,qpa);
    /// integration with accuracy of order 11.
    qpa.clear();
    warray0=0.2729250867779006307144835;
    varray[0]=0.2695431559523449723315320;	warray[0]=	0.2628045445102466621806889;
    varray[1]=0.5190961292068118159257257;	warray[1]=0.2331937645919904799185237;
    varray[2]=0.7301520055740493240934163;	warray[2]=	0.1862902109277342514260976;
    varray[3]=0.8870625997680952990751578;	warray[3]=	0.1255803694649046246346943;
    varray[4]=0.9782286581460569928039380;	warray[4]=	0.0556685671161736664827537;

    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(0.5*(1+varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
        v=BarycentricCoordinatesType(0.5*(1-varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
    }
    v=BarycentricCoordinatesType(0.5);
    qpa.push_back(QuadraturePoint(v,(Real)warray0/2));
    edgeNumericalIntegration.addQuadratureMethod(m,11,qpa);
    /// integration with accuracy of order 12.
    varray[0]=0.1252334085114689154724414;	warray[0]=	0.2491470458134027850005624;
    varray[1]=0.3678314989981801937526915;	warray[1]=	0.2334925365383548087608499;
    varray[2]=0.5873179542866174472967024;	warray[2]=	0.2031674267230659217490645;
    varray[3]=0.7699026741943046870368938;	warray[3]=	0.1600783285433462263346525;
    varray[4]=0.9041172563704748566784659;	warray[4]=	0.1069393259953184309602547;
    varray[5]=0.9815606342467192506905491;	warray[5]=	0.0471753363865118271946160;
    nbIPs=6;

    for (i=0;i<nbIPs;++i) {
        v=BarycentricCoordinatesType(0.5*(1+varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
        v=BarycentricCoordinatesType(0.5*(1-varray[i]));
        qpa.push_back(QuadraturePoint(v,(Real)warray[i]/2));
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
    const sofa::helper::vector<Edge> &ea = this->m_topology->getEdges();
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

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
    const sofa::defaulttype::Vec<3,double> &p,
    unsigned int ind_p1,
    unsigned int ind_p2) const
{
    const double ZERO = 1e-6;

    sofa::helper::vector< double > baryCoefs;

    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::ConstVecCoordId::position())->getValue());
    const typename DataTypes::Coord& c0 = vect_c[ind_p1];
    const typename DataTypes::Coord& c1 = vect_c[ind_p2];

    sofa::defaulttype::Vec<3,double> a; DataTypes::get(a[0], a[1], a[2], c0);
    sofa::defaulttype::Vec<3,double> b; DataTypes::get(b[0], b[1], b[2], c1);

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

    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::ConstVecCoordId::position())->getValue());

    const size_t numVertices = vect_c.size();

    myfile << "$NOD\n";
    myfile << numVertices <<"\n";

    for (size_t i=0; i<numVertices; ++i)
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

template<class DataTypes>
sofa::helper::vector< double > EdgeSetGeometryAlgorithms<DataTypes>::computeRest2PointsBarycoefs(
    const sofa::defaulttype::Vec<3,double> &p,
    unsigned int ind_p1,
    unsigned int ind_p2) const
{
    const double ZERO = 1e-6;

    sofa::helper::vector< double > baryCoefs;

    const typename DataTypes::VecCoord& vect_c = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());
    const typename DataTypes::Coord& c0 = vect_c[ind_p1];
    const typename DataTypes::Coord& c1 = vect_c[ind_p2];

    sofa::defaulttype::Vec<3,double> a; DataTypes::get(a[0], a[1], a[2], c0);
    sofa::defaulttype::Vec<3,double> b; DataTypes::get(b[0], b[1], b[2], c1);

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
sofa::helper::vector< double > EdgeSetGeometryAlgorithms<DataTypes>::computePointProjectionOnEdge (const EdgeID edgeIndex,
        sofa::defaulttype::Vec<3,double> c,
        bool& intersected)
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

    sofa::defaulttype::Vec<3,double> AB; DataTypes::get(AB[0], AB[1], AB[2], coord_AB);
    sofa::defaulttype::Vec<3,double> AC; DataTypes::get(AC[0], AC[1], AC[2], coord_AC);
    sofa::defaulttype::Vec<3,double> ortho_ABC = cross (AB, AC)*1000;
    sofa::defaulttype::Vec<3,double> coef_CH = cross (ortho_ABC, AB)*1000;

    for (unsigned int i = 0; i<Coord::spatial_dimensions; i++)
        coord_edge2[1][i] = coord_edge2[0][i] + (float)coef_CH[i];

    // Compute Coord of projection point H:
    Coord coord_H = compute2EdgesIntersection ( coord_edge1, coord_edge2, intersected);
    sofa::defaulttype::Vec<3,double> h; DataTypes::get(h[0], h[1], h[2], coord_H);

    sofa::helper::vector< double > barycoord = compute2PointsBarycoefs(h, theEdge[0], theEdge[1]);
    return barycoord;

}

template<class DataTypes>
bool EdgeSetGeometryAlgorithms<DataTypes>::computeEdgePlaneIntersection (EdgeID edgeID, sofa::defaulttype::Vec<3,Real> pointOnPlane, sofa::defaulttype::Vec<3,Real> normalOfPlane, sofa::defaulttype::Vec<3,Real>& intersection)
{
    const Edge &e = this->m_topology->getEdge(edgeID);
    const VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    sofa::defaulttype::Vec<3,Real> p1,p2;
    p1[0]=p[e[0]][0]; p1[1]=p[e[0]][1]; p1[2]=p[e[0]][2];
    p2[0]=p[e[1]][0]; p2[1]=p[e[1]][1]; p2[2]=p[e[1]][2];

    //plane equation
    normalOfPlane.normalize();
    double d=normalOfPlane*pointOnPlane;
    double t=(d-normalOfPlane*p1)/(normalOfPlane*(p2-p1));

    if((t<1)&&(t>=0))
    {
        intersection=p1+(p2-p1)*t;
        return true;
    }
    else
        return false;

}

template<class DataTypes>
bool EdgeSetGeometryAlgorithms<DataTypes>::computeRestEdgePlaneIntersection (EdgeID edgeID, sofa::defaulttype::Vec<3,Real> pointOnPlane, sofa::defaulttype::Vec<3,Real> normalOfPlane, sofa::defaulttype::Vec<3,Real>& intersection)
{
    const Edge &e = this->m_topology->getEdge(edgeID);
    const VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());

    sofa::defaulttype::Vec<3,Real> p1,p2;
    p1[0]=p[e[0]][0]; p1[1]=p[e[0]][1]; p1[2]=p[e[0]][2];
    p2[0]=p[e[1]][0]; p2[1]=p[e[1]][1]; p2[2]=p[e[1]][2];

    //plane equation
    normalOfPlane.normalize();
    double d=normalOfPlane*pointOnPlane;
    double t=(d-normalOfPlane*p1)/(normalOfPlane*(p2-p1));

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
    double epsilon = 0.0001;
    double lambda = 0.0;
    double alpha = 0.0;

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
                    double coef_lambda = vec1[ind1] - ( vec1[ind2]*vec2[ind1]/vec2[ind2] );

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
        std::cout << "Error: EdgeSetGeometryAlgorithms::compute2EdgeIntersection, vector director is null." << std::endl;
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
            std::cout << "Error: EdgeSetGeometryAlgorithms::compute2EdgeIntersection, edges don't intersect themself." << std::endl;
            intersected = false;
        }

    return X;
}


template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    PointSetGeometryAlgorithms<DataTypes>::draw(vparams);

    // Draw Edges indices
    if (showEdgeIndices.getValue())
    {
        const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
        float scale = this->getIndicesScale();

        //for edges:
        scale = scale/2;

        const sofa::helper::vector <Edge>& edgeArray = this->m_topology->getEdges();

        helper::vector<defaulttype::Vector3> positions;
        for (unsigned int i = 0; i < edgeArray.size(); i++)
        {

            Edge the_edge = edgeArray[i];
            Coord vertex1 = coords[the_edge[0]];
            Coord vertex2 = coords[the_edge[1]];
            defaulttype::Vector3 center;
            center = (DataTypes::getCPos(vertex1) + DataTypes::getCPos(vertex2)) / 2;

            positions.push_back(center);
        }
        vparams->drawTool()->draw3DText_Indices(positions, scale, _drawColor.getValue());
    }


    // Draw edges
    if (_draw.getValue())
    {
        const sofa::helper::vector<Edge> &edgeArray = this->m_topology->getEdges();

        if (!edgeArray.empty())
        {
            const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());

            std::vector<defaulttype::Vector3> positions;
            positions.reserve(edgeArray.size()*2u);
            for (unsigned int i = 0; i<edgeArray.size(); i++)
            {
                const Edge& e = edgeArray[i];
                positions.push_back(defaulttype::Vector3(DataTypes::getCPos(coords[e[0]])));
                positions.push_back(defaulttype::Vector3(DataTypes::getCPos(coords[e[1]])));
            }
            vparams->drawTool()->drawLines(positions,1.0f, _drawColor.getValue());
            vparams->drawTool()->drawPoints(positions, 4.0f, _drawColor.getValue());
        }
    }

}



template< class DataTypes>
void EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights( helper::vector<unsigned>& numEdges, helper::vector<Edge>& vertexEdges, helper::vector<Vec3d>& weights ) const
{
    const VecCoord& pos =(this->object->read(core::ConstVecCoordId::position())->getValue()); // point positions

    sofa::helper::vector<defaulttype::Vector3> edgeVec;                  // 3D edges

    numEdges.clear();
    vertexEdges.clear();
    weights.clear();

    const SeqEdges& edges = this->m_topology->getEdges();

    for(unsigned pointId=0; pointId<pos.size(); pointId++ )
    {
        //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, point " << pointId << endl;
        EdgesAroundVertex ve = this->m_topology->getEdgesAroundVertex(pointId);
        edgeVec.resize(ve.size());
        numEdges.push_back((unsigned)ve.size());            // number of edges attached to this point
        sofa::defaulttype::Matrix3 EEt,L;

        // Solve E.W = I , where each column of E is an adjacent edge vector, W are the desired weights, and I is the 3x3 identity
        // Each row of W corresponds to an edge, and encode the contribution of the edge to the basis vectors x,y,z
        // To solve this underconstrained system, we assume that W = Et.U , where Et is the transpose of E and U is 3x3
        // We solve (E.Et).U = I , then we compute W = Et.U
        // todo: weight the edges according to their lengths

        // compute E.Et
        for(unsigned e=0; e<ve.size(); e++ )
        {
            Edge edge = edges[ve[e]];
            vertexEdges.push_back(edge);              // concatenate
            CPos p0 = DataTypes::getCPos(pos[edge[0]]);
            CPos p1 = DataTypes::getCPos(pos[edge[1]]);
            edgeVec[e] = p1 - p0;
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights debug: edge "<< edge << ", edgeVec = " << edgeVec[e] << endl;
            // each edge vector adds e.et to the matrix
            for(unsigned j=0; j<3; j++)
                for(unsigned k=0; k<3; k++)
                    EEt[j][k] += edgeVec[e][k]*edgeVec[e][j];
        }

        // decompose E.Et for system solution
        if( cholDcmp(L,EEt) ) // Cholesky decomposition of the covariance matrix succeeds, we use it to solve the systems
        {
            size_t n = weights.size();     // start index for this vertex
            weights.resize( n + ve.size() ); // concatenate all the W of the nodes
            defaulttype::Vector3 a,u;

            // axis x
            a=defaulttype::Vector3(1,0,0);
            cholBksb(u,L,a); // solve EEt.u=x using the Cholesky decomposition
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, ux = " << u << endl;
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][0] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to x = " << weights[n+i][0] << endl;
            }

            // axis y
            a=defaulttype::Vector3(0,1,0);
            cholBksb(u,L,a); // solve EEt.u=y using the Cholesky decomposition
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, uy = " << u << endl;
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][1] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to y = " << weights[n+i][1] << endl;
            }

            // axis z
            a=defaulttype::Vector3(0,0,1);
            cholBksb(u,L,a); // solve EEt.u=z using the Cholesky decomposition
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, uz = " << u << endl;
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][2] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to z = " << weights[n+i][2] << endl;
            }
        }
        else
        {
            size_t n = weights.size();     // start index for this vertex
            weights.resize( n + ve.size() ); // concatenate all the W of the nodes
            defaulttype::Vector3 a,u;

            typedef Eigen::Matrix<SReal,3,3> EigenM33;
            EigenM33 emat = eigenMat(EEt);
//            Eigen::JacobiSVD<EigenM33> jacobi(emat, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::JacobiSVD<EigenM33> jacobi(emat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix<SReal,3,1> solution;

            // axis x
            a=defaulttype::Vector3(1,0,0);
            solution = jacobi.solve( eigenVec(a) );
            // least-squares solve EEt.u=x
            for(int i=0; i<3; i++)
                u[i] = solution(i);
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, ux = " << u << endl;
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][0] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to x = " << weights[n+i][0] << endl;
            }

            // axis y
            a=defaulttype::Vector3(0,1,0);
            solution = jacobi.solve( eigenVec(a) );
            // least-squares solve EEt.u=y
            for(int i=0; i<3; i++)
                u[i] = solution(i);
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, uy = " << u << endl;
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][1] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to y = " << weights[n+i][1] << endl;
            }

            // axis z
            a=defaulttype::Vector3(0,0,1);
            solution = jacobi.solve( eigenVec(a) );
            // least-squares solve EEt.u=z
            for(int i=0; i<3; i++)
                u[i] = solution(i);
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, uz = " << u << endl;
            for(size_t i=0; i<ve.size(); i++ )
            {
                weights[n+i][2] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to z = " << weights[n+i][2] << endl;
            }
        }

    }
}


template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::initPointAdded(unsigned int index, const core::topology::PointAncestorElem &ancestorElem
        , const helper::vector< VecCoord* >& coordVecs, const helper::vector< VecDeriv* >& derivVecs)
{
    using namespace sofa::core::topology;

    if (ancestorElem.type != EDGE)
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


} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_EDGESETGEOMETRYALGORITHMS_INL
