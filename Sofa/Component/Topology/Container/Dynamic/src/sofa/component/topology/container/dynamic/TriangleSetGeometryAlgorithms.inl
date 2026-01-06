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
#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.h>

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/topology/container/dynamic/CommonAlgorithms.h>
#include <fstream>


namespace sofa::component::topology::container::dynamic
{
const size_t permutation3[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
template< class DataTypes>
NumericalIntegrationDescriptor<typename TriangleSetGeometryAlgorithms< DataTypes >::Real,3> &TriangleSetGeometryAlgorithms< DataTypes >::getTriangleNumericalIntegrationDescriptor()
{
    // initialize the cubature table only if needed.
    if (initializedCubatureTables==false) {
        initializedCubatureTables=true;
        defineTetrahedronCubaturePoints();
    }
    return triangleNumericalIntegration;
}

template< class DataTypes>
void TriangleSetGeometryAlgorithms< DataTypes >::defineTetrahedronCubaturePoints() {
    typedef typename NumericalIntegrationDescriptor<typename TriangleSetGeometryAlgorithms< DataTypes >::Real,3>::QuadraturePoint QuadraturePoint;
    typedef typename NumericalIntegrationDescriptor<typename TriangleSetGeometryAlgorithms< DataTypes >::Real,3>::BarycentricCoordinatesType BarycentricCoordinatesType;
    // Gauss method
    typename NumericalIntegrationDescriptor<typename TriangleSetGeometryAlgorithms< DataTypes >::Real,3>::QuadratureMethod m=NumericalIntegrationDescriptor<typename TriangleSetGeometryAlgorithms< DataTypes >::Real,3>::GAUSS_SIMPLEX_METHOD;
    typename NumericalIntegrationDescriptor<typename TriangleSetGeometryAlgorithms< DataTypes >::Real,3>::QuadraturePointArray qpa;
    BarycentricCoordinatesType v;
    /// integration with linear accuracy.
    v=BarycentricCoordinatesType(1/(Real)3.0,1/(Real)3.0,1/(Real)3.0);
    qpa.push_back(QuadraturePoint(v,1/(Real)2));
    triangleNumericalIntegration.addQuadratureMethod(m,1,qpa);
    /// integration with quadratic accuracy.
    qpa.clear();
    Real a=1/(Real)6;
    Real b=(Real) (2.0/3.0);
    sofa::Index i;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,1/(Real)6));
    }
    triangleNumericalIntegration.addQuadratureMethod(m,2,qpa);
    /// integration with cubic accuracy.
    qpa.clear();
    v=BarycentricCoordinatesType(1/(Real)3.0,1/(Real)3.0,1/(Real)3.0);
    qpa.push_back(QuadraturePoint(v,(Real) -9/32));
    a=(Real)1/5;
    b=(Real)3/5;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,(Real)25/(Real)96));
    }
    triangleNumericalIntegration.addQuadratureMethod(m,3,qpa);
    /// integration with quadric accuracy with 6 points
    qpa.clear();
    a=(Real)0.445948490915965;
    b=(Real)1-2*a;
    Real c1= (Real)0.111690794839005;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    a=(Real)0.091576213509771;
    b=(Real)1-2*a;
    Real c2= (Real)0.054975871827661;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c2));
    }
    triangleNumericalIntegration.addQuadratureMethod(m,4,qpa);
    /// integration with quintic accuracy and 7 points
    qpa.clear();
    v=BarycentricCoordinatesType(1/(Real)3.0,1/(Real)3.0,1/(Real)3.0);
    qpa.push_back(QuadraturePoint(v,9/(Real)80));
    a=(Real)(6+sqrt(15))/21;
    b=(Real)1-2*a;
    c1=(Real)(155+sqrt(15))/2400;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    a=(Real)4/7-a;
    b=(Real)1-2*a;
     c2=(Real)31/240 -c1;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c2));
    }
    triangleNumericalIntegration.addQuadratureMethod(m,5,qpa);
    /// integration with order 6 accuracy and 12 points
     qpa.clear();
    a=(Real) 0.063089104491502;
    b=(Real)1-2*a;
    c1=(Real)0.025422453185103;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    a=(Real) 0.249286745170910;
    b=(Real)1-2*a;
    c1=(Real)0.058393137863189;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    Real aa[3];
    aa[0]=(Real)0.310352451033785;
    aa[1]=(Real)0.053145049844816;
    aa[2]=(Real)(1.0-aa[0]-aa[1]);
    c2=(Real)0.041425537809187;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(aa[permutation3[i][0]],aa[permutation3[i][1]],aa[permutation3[i][2]]);
        qpa.push_back(QuadraturePoint(v,c2));
    }
    triangleNumericalIntegration.addQuadratureMethod(m,6,qpa);
    /// integration with order 7 accuracy and 13 points
     qpa.clear();
     v=BarycentricCoordinatesType(1/(Real)3.0,1/(Real)3.0,1/(Real)3.0);
    qpa.push_back(QuadraturePoint(v,(Real)-0.0747850222338));
    a=(Real)0.0651301029022;
    b=(Real)1-2*a;
    c1=(Real)0.0266736178044;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    a=(Real) 0.2603459660790;
    b=(Real)1-2*a;
    c1=(Real)0.0878076287166;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }

    aa[0]=(Real)0.3128654960049;
    aa[1]=(Real)0.6384441885698;
    aa[2]=(Real)(1.0-aa[0]-aa[1]);
    c2=(Real)0.0385568804451;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(aa[permutation3[i][0]],aa[permutation3[i][1]],aa[permutation3[i][2]]);
        qpa.push_back(QuadraturePoint(v,c2));
    }
    triangleNumericalIntegration.addQuadratureMethod(m,7,qpa);
/// integration with order 8 accuracy and 16 points
    qpa.clear();
    v=BarycentricCoordinatesType(1/(Real)3.0,1/(Real)3.0,1/(Real)3.0);
    c1=(Real)0.1443156076777871682510911104890646/2.0;
    qpa.push_back(QuadraturePoint(v,(Real)c1));
    a=(Real)0.1705693077517602066222935014914645;
    b=(Real)1-2*a;
    c1=(Real)0.1032173705347182502817915502921290/2.0;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    a=(Real)0.0505472283170309754584235505965989;
    b=(Real)1-2*a;
    c1=(Real)0.0324584976231980803109259283417806/2.0;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    a=(Real)0.4592925882927231560288155144941693;
    b=(Real)1-2*a;
    c1=(Real)0.0950916342672846247938961043885843/2.0;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    aa[0]=(Real)0.2631128296346381134217857862846436;
    aa[1]=(Real)0.0083947774099576053372138345392944;
    aa[2]=(Real)(1.0-aa[0]-aa[1]);
    c2=(Real)0.0272303141744349942648446900739089/2.0;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(aa[permutation3[i][0]],aa[permutation3[i][1]],aa[permutation3[i][2]]);
        qpa.push_back(QuadraturePoint(v,c2));
    }
    triangleNumericalIntegration.addQuadratureMethod(m,8,qpa);
    /// integration with order 10 accuracy and 25 points from https://github.com/libMesh/libmesh/blob/master/src/quadrature/quadrature_gauss_2D.C
    qpa.clear();
    v=BarycentricCoordinatesType(1/(Real)3.0,1/(Real)3.0,1/(Real)3.0);
    c1=(Real)4.5408995191376790047643297550014267e-02L;
    qpa.push_back(QuadraturePoint(v,(Real)c1));
    a=(Real)4.8557763338365737736750753220812615e-01L;
    b=(Real)1-2*a;
    c1=(Real)1.8362978878233352358503035945683300e-02L;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    a=(Real)1.0948157548503705479545863134052284e-01L;
    b=(Real)1-2*a;
    c1=(Real)2.2660529717763967391302822369298659e-02L;
    for (i=0;i<3;++i) {
        v=BarycentricCoordinatesType(a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    aa[0]=(Real)3.0793983876412095016515502293063162e-01L;
    aa[1]=(Real)5.5035294182099909507816172659300821e-01L;
    aa[2]=(Real)(1.0-aa[0]-aa[1]);
    c2=(Real)3.6378958422710054302157588309680344e-02L;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(aa[permutation3[i][0]],aa[permutation3[i][1]],aa[permutation3[i][2]]);
        qpa.push_back(QuadraturePoint(v,c2));
    }
    aa[0]=(Real)2.4667256063990269391727646541117681e-01L;
    aa[1]=(Real)7.2832390459741092000873505358107866e-01L;
    aa[2]=(Real)(1.0-aa[0]-aa[1]);
    c2=(Real)1.4163621265528742418368530791049552e-02L;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(aa[permutation3[i][0]],aa[permutation3[i][1]],aa[permutation3[i][2]]);
        qpa.push_back(QuadraturePoint(v,c2));
    }
    aa[0]=(Real)6.6803251012200265773540212762024737e-02L;
    aa[1]=(Real)9.2365593358750027664630697761508843e-01L;
    aa[2]=(Real)(1.0-aa[0]-aa[1]);
    c2=(Real)4.7108334818664117299637354834434138e-03L;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(aa[permutation3[i][0]],aa[permutation3[i][1]],aa[permutation3[i][2]]);
        qpa.push_back(QuadraturePoint(v,c2));
    }
    triangleNumericalIntegration.addQuadratureMethod(m,10,qpa);

}


template<class DataTypes>
void TriangleSetGeometryAlgorithms< DataTypes >::init()
{
    EdgeSetGeometryAlgorithms<DataTypes>::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);

    TriangleSetGeometryAlgorithms< DataTypes >::reinit();
}

template<class DataTypes>
void TriangleSetGeometryAlgorithms< DataTypes >::reinit()
{
    EdgeSetGeometryAlgorithms<DataTypes>::reinit();

    if (p_recomputeTrianglesOrientation.getValue())
        this->reorderTrianglesOrientationFromNormals();
}


template< class DataTypes>
void TriangleSetGeometryAlgorithms< DataTypes >::computeTriangleAABB(const TriangleID i, Coord& minCoord, Coord& maxCoord) const
{
    const Triangle &t = this->m_topology->getTriangle(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());

    for(PointID i=0; i<3; ++i)
    {
        minCoord[i] = std::min(p[t[0]][i], std::min(p[t[1]][i], p[t[2]][i]));
        maxCoord[i] = std::max(p[t[0]][i], std::max(p[t[1]][i], p[t[2]][i]));
    }
}

template<class DataTypes>
typename DataTypes::Coord TriangleSetGeometryAlgorithms<DataTypes>::computeTriangleCenter(const TriangleID i) const
{
    const Triangle &t = this->m_topology->getTriangle(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());

    return (p[t[0]] + p[t[1]] + p[t[2]]) / (Real) 3.0;
}

template<class DataTypes>
typename DataTypes::Coord TriangleSetGeometryAlgorithms<DataTypes>::computeRestTriangleCenter(const TriangleID i) const
{
    const Triangle &t = this->m_topology->getTriangle(i);
    const typename DataTypes::VecCoord& p = (this->object->read(core::vec_id::read_access::restPosition)->getValue());

    return (p[t[0]] + p[t[1]] + p[t[2]]) / (Real) 3.0;
}

template<class DataTypes>
void TriangleSetGeometryAlgorithms<DataTypes>::computeTriangleCircumcenterBaryCoefs(sofa::type::Vec<3,Real> &baryCoord,
                                                                                    const TriangleID i) const
{
    const Triangle &t = this->m_topology->getTriangle(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());
    Real a2, b2, c2; // square lengths of the 3 edges
    a2 = (p[t[1]]-p[t[0]]).norm2();
    b2 = (p[t[2]]-p[t[1]]).norm2();
    c2 = (p[t[0]]-p[t[2]]).norm2();

    Real n = a2*(-a2+b2+c2) + b2*(a2-b2+c2) + c2*(a2+b2-c2);

    baryCoord[2] = a2*(-a2+b2+c2) / n;
    baryCoord[0] = b2*(a2-b2+c2) / n;
    baryCoord[1] = c2*(a2+b2-c2) / n;

    // barycentric coordinates are defined as
    //baryCoord = sofa::type::Vec<3,Real>(a2*(-a2+b2+c2) / n, b2*(a2-b2+c2) / n, c2*(a2+b2-c2) / n);
}

template<class DataTypes>
typename DataTypes::Coord TriangleSetGeometryAlgorithms<DataTypes>::computeTriangleCircumcenter(const TriangleID i) const
{
    const Triangle &t = this->m_topology->getTriangle(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());

    sofa::type::Vec<3,Real> barycentricCoords;
    computeTriangleCircumcenterBaryCoefs(barycentricCoords, i);

    return (p[t[0]]*barycentricCoords[0] + p[t[1]]*barycentricCoords[1] + p[t[2]]*barycentricCoords[2]);
}

template< class DataTypes>
void TriangleSetGeometryAlgorithms< DataTypes >::getTriangleVertexCoordinates(const TriangleID i, Coord pnt[3]) const
{
    const Triangle &t = this->m_topology->getTriangle(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());

    for(PointID i=0; i<3; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
void TriangleSetGeometryAlgorithms< DataTypes >::getRestTriangleVertexCoordinates(const TriangleID i, Coord pnt[3]) const
{
    const Triangle &t = this->m_topology->getTriangle(i);
    const typename DataTypes::VecCoord& p = (this->object->read(core::vec_id::read_access::restPosition)->getValue());

    for(PointID i=0; i<3; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
typename DataTypes::Real TriangleSetGeometryAlgorithms< DataTypes >::computeTriangleArea( const TriangleID i) const
{
    const Triangle &t = this->m_topology->getTriangle(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());
    Real area = (Real)(areaProduct(p[t[1]]-p[t[0]], p[t[2]]-p[t[0]]) * 0.5);
    return area;
}

template< class DataTypes>
typename DataTypes::Real TriangleSetGeometryAlgorithms< DataTypes >::computeRestTriangleArea( const TriangleID i) const
{
    const Triangle &t = this->m_topology->getTriangle(i);
    const typename DataTypes::VecCoord& p = (this->object->read(core::vec_id::read_access::restPosition)->getValue());
    Real area = (Real) (areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]]) * 0.5);
    return area;
}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void TriangleSetGeometryAlgorithms<DataTypes>::computeTriangleArea( BasicArrayInterface<Real> &ai) const
{
    const sofa::type::vector<Triangle> &ta = this->m_topology->getTriangles();
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());

    for (size_t i=0; i<ta.size(); ++i)
    {
        const Triangle &t=ta[i];
        Coord vec1 = p[t[1]] - p[t[0]];
        Coord vec2 = p[t[2]] - p[t[0]];
        ai[(int)i]=(Real)(areaProduct(vec1, vec2) * 0.5);
    }
}

// Computes the point defined by 2 indices of vertex and 1 barycentric coordinate
template<class DataTypes>
auto TriangleSetGeometryAlgorithms< DataTypes >::computeBaryEdgePoint(PointID p0, PointID p1, Real coord_p) const -> sofa::type::Vec<3, Real>
{
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

    const sofa::type::Vec<3,Real> c0 = type::toVecN<3, Real>(vect_c[p0]);
    const sofa::type::Vec<3,Real> c1 = type::toVecN<3, Real>(vect_c[p1]);
    return c0*(1-coord_p) + c1*coord_p;
}

template<class DataTypes>
auto TriangleSetGeometryAlgorithms< DataTypes >::computeBaryTrianglePoint(PointID p0, PointID p1, PointID p2, sofa::type::Vec<3,Real>& coord_p) const -> sofa::type::Vec<3, Real>
{
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

    sofa::type::Vec<3,Real> c0 = type::toVecN<3, Real>(vect_c[p0]);
    sofa::type::Vec<3,Real> c1 = type::toVecN<3, Real>(vect_c[p1]);
    sofa::type::Vec<3,Real> c2 = type::toVecN<3, Real>(vect_c[p2]);
    return c0*coord_p[0] + c1*coord_p[1] + c2*coord_p[2];
}


// Computes the opposite point to ind_p
template<class DataTypes>
auto TriangleSetGeometryAlgorithms< DataTypes >::getOppositePoint(PointID ind_p,
        const Edge& indices,
        Real coord_p) const -> sofa::type::Vec<3, Real>
{
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

    const typename DataTypes::Coord& c1 = vect_c[indices[0]];
    const typename DataTypes::Coord& c2 = vect_c[indices[1]];

    sofa::type::Vec<3,Real> p;

    if(ind_p == indices[0])
    {
        p = { c2[0], c2[1], c2[2] };
    }
    else if (ind_p == indices[1])
    {
        p = { c1[0], c1[1], c1[2] };
    }
    else
    {
        p[0]= (Real) ((1.0-coord_p)*c1[0] + coord_p*c2[0]);
        p[1]= (Real) ((1.0-coord_p)*c1[1] + coord_p*c2[1]);
        p[2]= (Real) ((1.0-coord_p)*c1[2] + coord_p*c2[2]);
    }

    return p;
}

// Computes the normal vector of a triangle indexed by ind_t (not normed)
template<class DataTypes>
auto TriangleSetGeometryAlgorithms< DataTypes >::computeTriangleNormal(const TriangleID ind_t) const -> sofa::type::Vec<3, Real>
{
    const Triangle &t = this->m_topology->getTriangle(ind_t);
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

    sofa::type::Vec<3, Real> p0; DataTypes::get(p0[0], p0[1], p0[2], vect_c[t[0]]);
    sofa::type::Vec<3, Real> p1; DataTypes::get(p1[0], p1[1], p1[2], vect_c[t[1]]);
    sofa::type::Vec<3, Real> p2; DataTypes::get(p2[0], p2[1], p2[2], vect_c[t[2]]);
    
    return sofa::geometry::Triangle::normal(p0, p1, p2);
}

// barycentric coefficients of point p in triangle (a,b,c) indexed by ind_t
template<class DataTypes>
auto TriangleSetGeometryAlgorithms< DataTypes >::computeTriangleBarycoefs(
    const TriangleID ind_t,
    const sofa::type::Vec<3,Real> &p) const -> sofa::type::vector<SReal>
{
    const Triangle &t=this->m_topology->getTriangle(ind_t);
    return compute3PointsBarycoefs(p, t[0], t[1], t[2],false);
}

template<class DataTypes>
auto TriangleSetGeometryAlgorithms< DataTypes >::computeTriangleBarycentricCoordinates(const TriangleID ind_t, const sofa::type::Vec<3, Real>& p, bool useRestPosition)  const -> sofa::type::Vec<3, Real>
{
    sofa::core::ConstVecCoordId::MyVecId _vecId = useRestPosition ? core::vec_id::read_access::restPosition : core::vec_id::read_access::position;
    const Triangle& t = this->m_topology->getTriangle(ind_t);
    const typename DataTypes::VecCoord& vect_c = (this->object->read(_vecId)->getValue());

    sofa::type::Vec<3, Real> p0; DataTypes::get(p0[0], p0[1], p0[2], vect_c[t[0]]);
    sofa::type::Vec<3, Real> p1; DataTypes::get(p1[0], p1[1], p1[2], vect_c[t[1]]);
    sofa::type::Vec<3, Real> p2; DataTypes::get(p2[0], p2[1], p2[2], vect_c[t[2]]);

    return sofa::geometry::Triangle::getBarycentricCoordinates(p, p0, p1, p2);
}


// barycentric coefficients of point p in initial triangle (a,b,c) indexed by ind_t
template<class DataTypes>
auto TriangleSetGeometryAlgorithms< DataTypes >::computeRestTriangleBarycoefs(
    const TriangleID ind_t,
    const sofa::type::Vec<3, Real>& p) const -> sofa::type::vector<SReal>
{
    const Triangle& t = this->m_topology->getTriangle(ind_t);
    return compute3PointsBarycoefs(p, t[0], t[1], t[2], true);
}

// barycentric coefficients of point p in triangle whose vertices are indexed by (ind_p1,ind_p2,ind_p3)
template<class DataTypes>
auto TriangleSetGeometryAlgorithms< DataTypes >::compute3PointsBarycoefs(
    const sofa::type::Vec<3, Real> &p,
    PointID ind_p1,
    PointID ind_p2,
    PointID ind_p3,
    bool bRest) const -> sofa::type::vector<SReal>
{
    const Real ZERO = 1e-12;
    sofa::type::vector< SReal > baryCoefs;

    const typename DataTypes::VecCoord& vect_c = (bRest ? (this->object->read(core::vec_id::read_access::restPosition)->getValue()) : (this->object->read(core::vec_id::read_access::position)->getValue()));

    const typename DataTypes::Coord& c0 = vect_c[ind_p1];
    const typename DataTypes::Coord& c1 = vect_c[ind_p2];
    const typename DataTypes::Coord& c2 = vect_c[ind_p3];

    sofa::type::Vec<3,Real> a;
    a[0] = (Real) (c0[0]);
    a[1] = (Real) (c0[1]);
    a[2] = (Real) (c0[2]);
    sofa::type::Vec<3,Real> b;
    b[0] = (Real) (c1[0]);
    b[1] = (Real) (c1[1]);
    b[2] = (Real) (c1[2]);
    sofa::type::Vec<3,Real> c;
    c[0] = (Real) (c2[0]);
    c[1] = (Real) (c2[1]);
    c[2] = (Real) (c2[2]);

    sofa::type::Vec<3,Real> M = (sofa::type::Vec<3,Real>) (b-a).cross(c-a);
    Real norm2_M = M*(M);

    Real coef_a, coef_b, coef_c;

    //if(norm2_M==0.0) // triangle (a,b,c) is flat
    if(norm2_M < ZERO) // triangle (a,b,c) is flat
    {
        coef_a = (Real) (1.0/3.0);
        coef_b = (Real) (1.0/3.0);
        coef_c = (Real) (1.0 - (coef_a + coef_b));

    }
    else
    {
        sofa::type::Vec<3,Real> N =  M/norm2_M;

        coef_a = N*((b-p).cross(c-p));
        coef_b = N*((c-p).cross(a-p));
        coef_c = (Real) (1.0 - (coef_a + coef_b)); //N*((a-p).cross(b-p));
    }

    baryCoefs.push_back(coef_a);
    baryCoefs.push_back(coef_b);
    baryCoefs.push_back(coef_c);

    return baryCoefs;
}

// Find the two closest points from two triangles (each of the point belonging to one triangle)
template<class DataTypes>
void TriangleSetGeometryAlgorithms< DataTypes >::computeClosestIndexPair(const TriangleID ind_ta, const TriangleID ind_tb,
        PointID &ind1, PointID &ind2) const
{
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

    const Triangle &ta=this->m_topology->getTriangle(ind_ta);
    const Triangle &tb=this->m_topology->getTriangle(ind_tb);

    Real min_value=(Real) 0.0;
    bool is_init = false;

    for(unsigned int i=0; i<3; i++)
    {
        const typename DataTypes::Coord& ca=vect_c[ta[i]];
        sofa::type::Vec<3,Real> pa;
        pa[0] = (Real) (ca[0]);
        pa[1] = (Real) (ca[1]);
        pa[2] = (Real) (ca[2]);

        for(unsigned int j=0; j!=i && j<3; j++)
        {
            const typename DataTypes::Coord& cb=vect_c[tb[i]];
            sofa::type::Vec<3,Real> pb;
            pb[0] = (Real) (cb[0]);
            pb[1] = (Real) (cb[1]);
            pb[2] = (Real) (cb[2]);

            Real norm_v_normal = (pa-pb)*(pa-pb);
            if(!is_init)
            {
                min_value = norm_v_normal;
                ind1 = ta[i];
                ind2 = tb[j];
                is_init = true;
            }
            else
            {
                if(norm_v_normal<min_value)
                {
                    min_value = norm_v_normal;
                    ind1 = ta[i];
                    ind2 = tb[j];
                }
            }
        }
    }

    return;
}

// test if a point is inside the triangle indexed by ind_t
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::isPointInsideTriangle(const TriangleID ind_t,
        bool is_tested,
        const sofa::type::Vec<3,Real>& p,
        TriangleID &ind_t_test,
        bool bRest) const
{
    const Real ZERO = -1e-12;
    const typename DataTypes::VecCoord& vect_c = bRest
        ? (this->object->read(core::vec_id::read_access::restPosition)->getValue())
        :(this->object->read(core::vec_id::read_access::position)->getValue());
    const Triangle &t=this->m_topology->getTriangle(ind_t);

    const typename DataTypes::Coord& c0=vect_c[t[0]];
    const typename DataTypes::Coord& c1=vect_c[t[1]];
    const typename DataTypes::Coord& c2=vect_c[t[2]];

    sofa::type::Vec<3,Real> ptest = p;

    sofa::type::Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]);
    p0[1] = (Real) (c0[1]);
    p0[2] = (Real) (c0[2]);
    sofa::type::Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]);
    p1[1] = (Real) (c1[1]);
    p1[2] = (Real) (c1[2]);
    sofa::type::Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]);
    p2[1] = (Real) (c2[1]);
    p2[2] = (Real) (c2[2]);

    sofa::type::Vec<3,Real> v_normal = (p2-p0).cross(p1-p0);

    Real norm_v_normal = v_normal*(v_normal);
    if(norm_v_normal != 0.0)
    {
        sofa::type::Vec<3,Real> n_01 = (p1-p0).cross(v_normal);
        sofa::type::Vec<3,Real> n_12 = (p2-p1).cross(v_normal);
        sofa::type::Vec<3,Real> n_20 = (p0-p2).cross(v_normal);

        Real v_01 = (Real) ((ptest-p0)*(n_01));
        Real v_12 = (Real) ((ptest-p1)*(n_12));
        Real v_20 = (Real) ((ptest-p2)*(n_20));

        const bool is_inside = (v_01 > ZERO) && (v_12 > ZERO) && (v_20 > ZERO);

        if(is_tested && (!is_inside))
        {
            sofa::type::vector< TriangleID > shell;
            EdgeID ind_edge = 0;

            if(v_01 < 0.0)
            {
                if(v_12 < 0.0) /// vertex 1
                {
                    shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundVertex(t[1]));
                }
                else
                {
                    if(v_20 < 0.0) /// vertex 0
                    {
                        shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundVertex(t[0]));

                    }
                    else // v_01 < 0.0
                    {
                        ind_edge=this->m_topology->getEdgeIndex(t[0],t[1]);
                        shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundEdge(ind_edge));
                    }
                }
            }
            else
            {
                if(v_12 < 0.0)
                {
                    if(v_20 < 0.0) /// vertex 2
                    {
                        shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundVertex(t[2]));

                    }
                    else // v_12 < 0.0
                    {
                        ind_edge=this->m_topology->getEdgeIndex(t[1],t[2]);
                        shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundEdge(ind_edge));
                    }
                }
                else // v_20 < 0.0
                {
                    ind_edge=this->m_topology->getEdgeIndex(t[2],t[0]);
                    shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundEdge(ind_edge));
                }
            }

            size_t i =0;
            bool is_in_next_triangle=false;
            TriangleID ind_triangle=0;
            TriangleID ind_t_false_init;
            TriangleID &ind_t_false = ind_t_false_init;

            if(shell.size()>1)
            {
                while(i < shell.size() && !is_in_next_triangle)
                {
                    ind_triangle=shell[i];

                    if(ind_triangle != ind_t)
                    {
                        is_in_next_triangle = isPointInTriangle(ind_triangle, false, p, ind_t_false);
                    }
                    i++;
                }

                if(is_in_next_triangle)
                {
                    ind_t_test=ind_triangle;
                    //msg_info() << "correct to triangle indexed by " << ind_t_test;
                }
                else // not found
                {
                    //msg_info() << "not found !!! ";
                    ind_t_test=ind_t;
                }
            }
            else
            {
                ind_t_test=ind_t;
            }
        }
        return is_inside;

    }
    else // triangle is flat
    {
        //msg_info() << "INFO_print : triangle is flat";
        return false;
    }
}

// test if a point is in the triangle indexed by ind_t
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::isPointInTriangle(const TriangleID ind_t,
        bool is_tested,
        const sofa::type::Vec<3,Real>& p,
        TriangleID &ind_t_test) const
{
    const Real ZERO = 1e-12;
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());
    const Triangle &t=this->m_topology->getTriangle(ind_t);

    const typename DataTypes::Coord& c0=vect_c[t[0]];
    const typename DataTypes::Coord& c1=vect_c[t[1]];
    const typename DataTypes::Coord& c2=vect_c[t[2]];

    sofa::type::Vec<3,Real> ptest = p;

    sofa::type::Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]);
    p0[1] = (Real) (c0[1]);
    p0[2] = (Real) (c0[2]);
    sofa::type::Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]);
    p1[1] = (Real) (c1[1]);
    p1[2] = (Real) (c1[2]);
    sofa::type::Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]);
    p2[1] = (Real) (c2[1]);
    p2[2] = (Real) (c2[2]);

    sofa::type::Vec<3,Real> v_normal = (p2-p0).cross(p1-p0);

    Real norm_v_normal = v_normal*(v_normal);
    //if(norm_v_normal != 0.0)
    if(norm_v_normal > ZERO)
    {
        sofa::type::Vec<3,Real> n_01 = (p1-p0).cross(v_normal);
        sofa::type::Vec<3,Real> n_12 = (p2-p1).cross(v_normal);
        sofa::type::Vec<3,Real> n_20 = (p0-p2).cross(v_normal);

        Real v_01 = (Real) ((ptest-p0)*(n_01));
        Real v_12 = (Real) ((ptest-p1)*(n_12));
        Real v_20 = (Real) ((ptest-p2)*(n_20));

        //bool is_inside = (v_01 > 0.0) && (v_12 > 0.0) && (v_20 > 0.0);
        const bool is_inside = (v_01 > -ZERO) && (v_12 > -ZERO) && (v_20 >= -ZERO);

        if(is_tested && (!is_inside))
        {
            sofa::type::vector< TriangleID > shell;
            EdgeID ind_edge = 0;

            //if(v_01 < 0.0)
            if(v_01 < -ZERO)
            {
                //if(v_12 < 0.0) /// vertex 1
                if(v_12 < -ZERO) /// vertex 1
                {
                    shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundVertex(t[1]));
                }
                else
                {
                    //if(v_20 < 0.0) /// vertex 0
                    if(v_20 < -ZERO) /// vertex 0
                    {
                        shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundVertex(t[0]));

                    }
                    else // v_01 < 0.0
                    {
                        ind_edge=this->m_topology->getEdgeIndex(t[0],t[1]);
                        shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundEdge(ind_edge));
                    }
                }
            }
            else
            {
                //if(v_12 < 0.0)
                if(v_12 < -ZERO)
                {
                    //if(v_20 < 0.0) /// vertex 2
                    if(v_20 < -ZERO) /// vertex 2
                    {
                        shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundVertex(t[2]));

                    }
                    else // v_12 < 0.0
                    {
                        ind_edge=this->m_topology->getEdgeIndex(t[1],t[2]);
                        shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundEdge(ind_edge));
                    }
                }
                else // v_20 < 0.0
                {
                    ind_edge=this->m_topology->getEdgeIndex(t[2],t[0]);
                    shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundEdge(ind_edge));
                }
            }

            size_t i =0;
            bool is_in_next_triangle=false;
            TriangleID ind_triangle=0;
            TriangleID ind_t_false_init;
            TriangleID &ind_t_false = ind_t_false_init;

            if(shell.size()>1)
            {
                while(i < shell.size() && !is_in_next_triangle)
                {
                    ind_triangle=shell[i];

                    if(ind_triangle != ind_t)
                    {
                        is_in_next_triangle = isPointInTriangle(ind_triangle, false, p, ind_t_false);
                    }
                    i++;
                }

                if(is_in_next_triangle)
                {
                    ind_t_test=ind_triangle;
                    //msg_info() << "correct to triangle indexed by " << ind_t_test;
                }
                else // not found
                {
                    //msg_info() << "not found !!! ";
                    ind_t_test=ind_t;
                }
            }
            else
            {
                ind_t_test=ind_t;
            }
        }
        return is_inside;

    }
    else // triangle is flat
    {
        //msg_info() << "INFO_print : triangle is flat";
        return false;
    }
}

// Tests how to triangularize a quad whose vertices are defined by (p_q1, p_q2, ind_q3, ind_q4) according to the Delaunay criterion
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::isQuadDeulaunayOriented(const typename DataTypes::Coord& p_q1,
        const typename DataTypes::Coord& p_q2,
        QuadID ind_q3,
        QuadID ind_q4)
{
    sofa::type::vector< Real > baryCoefs;

    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

    const typename DataTypes::Coord& c3 = vect_c[ind_q3];
    const typename DataTypes::Coord& c4 = vect_c[ind_q4];

    return isQuadDeulaunayOriented(p_q1, p_q2, c3, c4);
}

/** \brief Tests how to triangularize a quad whose vertices are defined by (p1, p2, p3, p4) according to the Delaunay criterion
 *
 */
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::isQuadDeulaunayOriented(const typename DataTypes::Coord& p1,
        const typename DataTypes::Coord& p2,
        const typename DataTypes::Coord& p3,
        const typename DataTypes::Coord& p4)
{
    /* use formula with angles
     * A----B     p1----p2
     *   \   \     \     \
     *    D----C    p4----p3
     * if the sum of opposites angles (not on the common edge) is < 180deg, the triangles meet the Delaunay condition
     */
    sofa::type::Vec<3, Real> AB = { p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] };
    sofa::type::Vec<3, Real> AD = { p4[0] - p1[0], p4[1] - p1[1], p4[2] - p1[2] };
    
    sofa::type::Vec<3, Real> CB = { p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2] };
    sofa::type::Vec<3, Real> CD = { p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2] };

    AB.normalize();
    AD.normalize();
    Real alpha = acos(dot(AB, AD));

    CB.normalize();
    CD.normalize();
    Real beta = acos(CB * CD);
    const bool isDelau = (alpha + beta <= M_PI);

    return isDelau;
}


template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::isDiagonalsIntersectionInQuad (const typename DataTypes::Coord triangle1[3],const  typename DataTypes::Coord triangle2[3])
{

    Coord CommonEdge[2], oppositeVertices[2];
    unsigned int cpt = 0;
    bool test = false;

    for (unsigned int i = 0; i<3; i++)
    {
        test = false;
        for (unsigned int j = 0; j<3; j++)
            if(triangle1[i] == triangle2[j])
            {
                test = true;
                break;
            }

        if(test)
        {
            CommonEdge[cpt] = triangle1[i];
            cpt++;
        }
        else
            oppositeVertices[0] = triangle1[i];
    }


    for (unsigned int i = 0; i<3; i++)
    {
        test = false;
        for (unsigned int j = 0; j<2; j++)
            if (triangle2[i] == CommonEdge[j])
            {
                test = true;
                break;
            }

        if (!test)
        {
            oppositeVertices[1] = triangle2[i];
            break;
        }
    }

    bool intersected = false;

    Coord inter = this->compute2EdgesIntersection (CommonEdge, oppositeVertices, intersected);

    if (intersected)
    {
        sofa::type::Vec<3,Real> A; DataTypes::get(A[0], A[1], A[2], CommonEdge[0]);
        sofa::type::Vec<3,Real> B; DataTypes::get(B[0], B[1], B[2], CommonEdge[1]);

        sofa::type::Vec<3,Real> C; DataTypes::get(C[0], C[1], C[2], oppositeVertices[0]);
        sofa::type::Vec<3,Real> D; DataTypes::get(D[0], D[1], D[2], oppositeVertices[1]);

        sofa::type::Vec<3,Real> X; DataTypes::get(X[0], X[1], X[2], inter);

        Real ABAX = (A - B)*(A - X);
        Real CDCX = (C - D)*(C - X);

        if ( (ABAX < 0) || (CDCX < 0) || ((A - X).norm2() > (A - B).norm2()) || ((C - X).norm2() > (C - D).norm2()) )
            return false;
        else
            return true;
    }

    return false;
}


// Test if a triangle indexed by ind_triangle (and incident to the vertex indexed by ind_p) is included or not in the plane defined by (ind_p, plane_vect)
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::isTriangleInPlane(const TriangleID ind_t,
        const PointID ind_p,
        const sofa::type::Vec<3,Real>&plane_vect) const
{
    const Triangle &t=this->m_topology->getTriangle(ind_t);

    // HYP : ind_p==t[0] or ind_p==t[1] or ind_p==t[2]

    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

    PointID ind_1;
    PointID ind_2;

    if(ind_p==t[0])
    {
        ind_1=t[1];
        ind_2=t[2];
    }
    else
    {
        if(ind_p==t[1])
        {
            ind_1=t[2];
            ind_2=t[0];
        }
        else // ind_p==t[2]
        {
            ind_1=t[0];
            ind_2=t[1];
        }
    }

    const typename DataTypes::Coord& c0=vect_c[ind_p];
    const typename DataTypes::Coord& c1=vect_c[ind_1];
    const typename DataTypes::Coord& c2=vect_c[ind_2];

    sofa::type::Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]);
    p0[1] = (Real) (c0[1]);
    p0[2] = (Real) (c0[2]);
    sofa::type::Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]);
    p1[1] = (Real) (c1[1]);
    p1[2] = (Real) (c1[2]);
    sofa::type::Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]);
    p2[1] = (Real) (c2[1]);
    p2[2] = (Real) (c2[2]);

    return((p1-p0)*( plane_vect)>=0.0 && (p1-p0)*( plane_vect)>=0.0);
}

// Prepares the duplication of a vertex
template<class DataTypes>
void TriangleSetGeometryAlgorithms< DataTypes >::prepareVertexDuplication(const PointID ind_p,
        const TriangleID ind_t_from,
        const TriangleID ind_t_to,
        const Edge& indices_from,
        const Real &coord_from,
        const Edge& indices_to,
        const Real &coord_to,
        sofa::type::vector< TriangleID > &triangles_list_1,
        sofa::type::vector< TriangleID > &triangles_list_2) const
{
    //HYP : if coord_from or coord_to == 0.0 or 1.0, ind_p is distinct from ind_from and from ind_to

    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

    const typename DataTypes::Coord& c_p = vect_c[ind_p];
    sofa::type::Vec<3,Real> point_p;
    point_p[0]= (Real) c_p[0];
    point_p[1]= (Real) c_p[1];
    point_p[2]= (Real) c_p[2];

    sofa::type::Vec<3, Real> point_from = getOppositePoint(ind_p, indices_from, coord_from);
    sofa::type::Vec<3, Real> point_to = getOppositePoint(ind_p, indices_to, coord_to);

    //Vec<3,Real> point_from=(Vec<3,Real>) computeBaryEdgePoint((sofa::type::vector< TriangleID>&) indices_from, coord_from);
    //Vec<3,Real> point_to=(Vec<3,Real>) computeBaryEdgePoint((sofa::type::vector< TriangleID>&) indices_to, coord_to);

    sofa::type::Vec<3,Real> vect_from = point_from - point_p;
    sofa::type::Vec<3,Real> vect_to = point_p - point_to;

    //msg_info() << "INFO_print : vect_from = " << vect_from <<  sendl;
    //msg_info() << "INFO_print : vect_to = " << vect_to <<  sendl;

    sofa::type::Vec<3,Real> normal_from;
    sofa::type::Vec<3,Real> normal_to;

    sofa::type::Vec<3,Real> plane_from;
    sofa::type::Vec<3,Real> plane_to;

    if((coord_from!=0.0) && (coord_from!=1.0))
    {
        normal_from=(sofa::type::Vec<3,Real>) computeTriangleNormal(ind_t_from);
        plane_from=vect_from.cross( normal_from); // inverse ??
    }
    else
    {
        // HYP : only 2 edges maximum are adjacent to the same triangle (otherwise : compute the one which minimizes the normed dotProduct and which gives the positive cross)

        EdgeID ind_edge;

        if(coord_from==0.0)
        {
            ind_edge=this->m_topology->getEdgeIndex(indices_from[0], ind_p);
        }
        else // coord_from==1.0
        {
            ind_edge=this->m_topology->getEdgeIndex(indices_from[1], ind_p);
        }

        if (this->m_topology->getNbEdges()>0)
        {
            sofa::type::vector< TriangleID > shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundEdge(ind_edge));
            TriangleID ind_triangle=shell[0];
            size_t i=0;
            bool is_in_next_triangle=false;

            if(shell.size()>1)
            {
                while(i < shell.size() || !is_in_next_triangle)
                {
                    if(shell[i] != ind_t_from)
                    {
                        ind_triangle=shell[i];
                        is_in_next_triangle=true;
                    }
                    i++;
                }
            }
            else
            {
                return;
            }

            if(is_in_next_triangle)
            {
                sofa::type::Vec<3,Real> normal_from_1=(sofa::type::Vec<3,Real>) computeTriangleNormal(ind_triangle);
                sofa::type::Vec<3,Real> normal_from_2=(sofa::type::Vec<3,Real>) computeTriangleNormal(ind_t_from);

                normal_from=(normal_from_1+normal_from_2)/2.0;
                plane_from=vect_from.cross( normal_from);
            }
            else
            {
                return;
            }
        }
    }

    if((coord_to!=0.0) && (coord_to!=1.0))
    {
        normal_to=(sofa::type::Vec<3,Real>) computeTriangleNormal(ind_t_to);

        plane_to=vect_to.cross( normal_to);
    }
    else
    {
        // HYP : only 2 edges maximum are adjacent to the same triangle (otherwise : compute the one which minimizes the normed dotProduct and which gives the positive cross)

        EdgeID ind_edge;

        if(coord_to==0.0)
        {
            ind_edge=this->m_topology->getEdgeIndex(indices_to[0], ind_p);
        }
        else // coord_to==1.0
        {
            ind_edge=this->m_topology->getEdgeIndex(indices_to[1], ind_p);
        }

        if (this->m_topology->getNbEdges()>0)
        {
            sofa::type::vector< TriangleID > shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundEdge(ind_edge));
            TriangleID ind_triangle=shell[0];
            size_t i=0;
            bool is_in_next_triangle=false;

            if(shell.size()>1)
            {
                while(i < shell.size() || !is_in_next_triangle)
                {
                    if(shell[i] != ind_t_to)
                    {
                        ind_triangle=shell[i];
                        is_in_next_triangle=true;
                    }
                    i++;
                }
            }
            else
            {
                return;
            }

            if(is_in_next_triangle)
            {
                sofa::type::Vec<3,Real> normal_to_1=(sofa::type::Vec<3,Real>) computeTriangleNormal(ind_triangle);
                sofa::type::Vec<3,Real> normal_to_2=(sofa::type::Vec<3,Real>) computeTriangleNormal(ind_t_to);

                normal_to=(normal_to_1+normal_to_2)/2.0;
                plane_to=vect_to.cross( normal_to);
            }
            else
            {
                return;
            }
        }
    }

    if (this->m_topology->getNbPoints()>0)
    {
        sofa::type::vector< TriangleID > shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundVertex(ind_p));
        TriangleID ind_triangle=shell[0];
        size_t i=0;

        bool is_in_plane_from;
        bool is_in_plane_to;

        if(shell.size()>1)
        {
            sofa::type::Vec<3,Real> normal_test = plane_from.cross( plane_to);
            Real value_test =   normal_test*(normal_from+normal_to);

            if(value_test<=0.0)
            {
                while(i < shell.size())
                {
                    ind_triangle=shell[i];

                    is_in_plane_from=isTriangleInPlane(ind_triangle,ind_p, (const sofa::type::Vec<3,Real>&) plane_from);
                    is_in_plane_to=isTriangleInPlane(ind_triangle,ind_p, (const sofa::type::Vec<3,Real>&) plane_to);

                    if((ind_triangle != ind_t_from) && (ind_triangle != ind_t_to))
                    {
                        if(is_in_plane_from || is_in_plane_to)
                        {
                            triangles_list_1.push_back(ind_triangle);
                        }
                        else
                        {
                            triangles_list_2.push_back(ind_triangle);
                        }
                    }
                    i++;
                }
            }
            else // value_test>0.0
            {
                while(i < shell.size())
                {
                    ind_triangle=shell[i];

                    is_in_plane_from=isTriangleInPlane(ind_triangle,ind_p, (const sofa::type::Vec<3,Real>&) plane_from);
                    is_in_plane_to=isTriangleInPlane(ind_triangle,ind_p, (const sofa::type::Vec<3,Real>&) plane_to);

                    if((ind_triangle != ind_t_from) && (ind_triangle != ind_t_to))
                    {
                        if(is_in_plane_from && is_in_plane_to)
                        {
                            triangles_list_1.push_back(ind_triangle);
                        }
                        else
                        {
                            triangles_list_2.push_back(ind_triangle);
                        }
                    }
                    i++;
                }
            }
        }
        else
        {
            return;
        }
    }
    else
    {
        return;
    }
}


template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::computeSegmentTriangleIntersectionInPlane(
    const sofa::type::Vec<3, Real>& ptA,
    const sofa::type::Vec<3, Real>& ptB,
    const TriangleID triId,
    sofa::type::vector<EdgeID>& intersectedEdges,
    sofa::type::vector<Real>& baryCoefs) const
{
    // Get coordinates of each vertex of the triangle
    const typename DataTypes::VecCoord& coords = (this->object->read(core::vec_id::read_access::position)->getValue());
    const Triangle& tri = this->m_topology->getTriangle(triId);

    const typename DataTypes::Coord& c0 = coords[tri[0]];
    const typename DataTypes::Coord& c1 = coords[tri[1]];
    const typename DataTypes::Coord& c2 = coords[tri[2]];
    type::fixed_array<Vec3, 3> triP = { Vec3(c0[0], c0[1], c0[2]), Vec3(c1[0], c1[1], c1[2]), Vec3(c2[0], c2[1], c2[2]) };

    // Project A and B into triangle plan
    Vec3 v_normal = (triP[2] - triP[0]).cross(triP[1] - triP[0]);
    v_normal.normalize();
    const Vec3 pa_proj = ptA - v_normal * dot(ptA - triP[0], v_normal);
    const Vec3 pb_proj = ptB - v_normal * dot(ptB - triP[0], v_normal);

    // check intersection between AB and each edge of the triangle
    const sofa::type::fixed_array<EdgeID, 3>& edgesInTri = this->m_topology->getEdgesInTriangle(triId);
    for (const EdgeID& edgeId : edgesInTri)
    {
        const Edge& edge = this->m_topology->getEdge(edgeId);
        Edge localIds;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++) 
            {
                if (edge[i] == tri[j])
                {
                    localIds[i] = j;
                    break;
                }
            }
        }

        type::Vec2 baryCoords(type::NOINIT);
        bool res = geometry::Edge::intersectionWithEdge(triP[localIds[0]], triP[localIds[1]], pa_proj, pb_proj, baryCoords);
        if (res)
        {
            intersectedEdges.push_back(edgeId);
            baryCoefs.push_back(baryCoords[0]);
        }
    }

    return !intersectedEdges.empty();
}


// Computes the intersection of the segment from point a to point b and the triangle indexed by t
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::computeSegmentTriangleIntersection(bool is_entered,
        const sofa::type::Vec<3,Real>& a,
        const sofa::type::Vec<3,Real>& b,
        const TriangleID ind_t,
        sofa::type::vector<TriangleID> &indices,
        Real &baryCoef, Real& coord_kmin) const
{
    // HYP : point a is in triangle indexed by t
    // is_entered == true => indices.size() == 2
    TriangleID ind_first=0;
    TriangleID ind_second=0;

    if(indices.size()>1)
    {
        ind_first=indices[0];
        ind_second=indices[1];
    }

    indices.clear();

    bool is_validated = false;
    bool is_intersected = false;

    const Triangle &t=this->m_topology->getTriangle(ind_t);
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

    bool is_full_01=(is_entered && ((t[0] == ind_first && t[1] == ind_second) || (t[1] == ind_first && t[0] == ind_second)));
    bool is_full_12=(is_entered && ((t[1] == ind_first && t[2] == ind_second) || (t[2] == ind_first && t[1] == ind_second)));
    bool is_full_20=(is_entered && ((t[2] == ind_first && t[0] == ind_second) || (t[0] == ind_first && t[2] == ind_second)));

    const typename DataTypes::Coord& c0=vect_c[t[0]];
    const typename DataTypes::Coord& c1=vect_c[t[1]];
    const typename DataTypes::Coord& c2=vect_c[t[2]];

    sofa::type::Vec<3, Real> p0 = { c0[0], c0[1], c0[2] };
    sofa::type::Vec<3,Real> p1 = { c1[0], c1[1], c1[2] };
    sofa::type::Vec<3,Real> p2 = { c2[0], c2[1], c2[2] };

    sofa::type::Vec<3,Real> pa = { a[0], a[1], a[2] };
    sofa::type::Vec<3,Real> pb = { b[0], b[1], b[2] };

    sofa::type::Vec<3,Real> v_normal = (p2-p0).cross(p1-p0);
    Real norm_v_normal = v_normal.norm(); // WARN : square root COST

    if(norm_v_normal != 0.0)
    {

        v_normal/=norm_v_normal;

        sofa::type::Vec<3,Real> v_ab = pb-pa;
        sofa::type::Vec<3,Real> v_ab_proj = v_ab - v_normal * dot(v_ab,v_normal); // projection (same values if incision in the plan)
        sofa::type::Vec<3,Real> pb_proj = v_ab_proj + pa;

        sofa::type::Vec<3,Real> v_01 = p1-p0;
        sofa::type::Vec<3,Real> v_12 = p2-p1;
        sofa::type::Vec<3,Real> v_20 = p0-p2;

        sofa::type::Vec<3,Real> n_proj =v_ab_proj.cross(v_normal);

        sofa::type::Vec<3,Real> n_01 = v_01.cross(v_normal);
        sofa::type::Vec<3,Real> n_12 = v_12.cross(v_normal);
        sofa::type::Vec<3,Real> n_20 = v_20.cross(v_normal);

        Real norm2_v_ab_proj = v_ab_proj*(v_ab_proj); //dot product WARNING

        if(norm2_v_ab_proj != 0.0) // pb_proj != pa
        {
            Real coord_t=0.0;
            Real coord_k=0.0;

            Real is_initialized=false;
            coord_kmin=0.0;

            Real coord_test1;
            Real coord_test2;

            Real s_t;
            Real s_k;

            if(!is_full_01)
            {
                /// Test of edge (p0,p1) :
                s_t = (p0-p1)*n_proj;
                s_k = (pa-pb_proj)*n_01;
                if(s_t==0.0) // (pa,pb_proj) and (p0,p1) are parallel
                {
                    if((p0-pa)*(n_proj)==0.0) // (pa,pb_proj) and (p0,p1) are on the same line
                    {
                        coord_test1 = (pa-p0)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa
                        coord_test2 = (pa-p1)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa

                        if(coord_test1>=0)
                        {
                            coord_k=coord_test1;
                            coord_t=0.0;
                        }
                        else
                        {
                            coord_k=coord_test2;
                            coord_t=1.0;
                        }

                        is_intersected = (coord_k > 0.0 && (coord_t >= 0.0 && coord_t <= 1.0));

                    }
                    else  // (pa,pb_proj) and (p0,p1) are parallel and disjoint
                    {
                        is_intersected=false;
                    }
                }
                else // s_t != 0.0 and s_k != 0.0
                {
                    coord_k=Real((pa-p0)*(n_01))*1.0/Real(s_k);
                    coord_t=Real((p0-pa)*(n_proj))*1.0/Real(s_t);

                    is_intersected = ((coord_k > 0.0) && (coord_t >= 0.0 && coord_t <= 1.0));
                }

                if(is_intersected)
                {
                    if((!is_initialized) || (coord_k > coord_kmin))
                    {
                        indices.clear();
                        indices.push_back(t[0]);
                        indices.push_back(t[1]);
                        baryCoef=coord_t;
                        coord_kmin=coord_k;
                    }

                    is_initialized=true;
                }

                is_validated = is_validated || is_initialized;
            }



            if(!is_full_12)
            {
                /// Test of edge (p1,p2) :

                s_t = (p1-p2)*(n_proj);
                s_k = (pa-pb_proj)*(n_12);

                // s_t == 0.0 iff s_k == 0.0

                if(s_t==0.0) // (pa,pb_proj) and (p1,p2) are parallel
                {
                    if((p1-pa)*(n_proj)==0.0) // (pa,pb_proj) and (p1,p2) are on the same line
                    {
                        coord_test1 = (pa-p1)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa
                        coord_test2 = (pa-p2)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa

                        if(coord_test1>=0)
                        {
                            coord_k=coord_test1;
                            coord_t=0.0;
                        }
                        else
                        {
                            coord_k=coord_test2;
                            coord_t=1.0;
                        }

                        is_intersected = (coord_k > 0.0 && (coord_t >= 0.0 && coord_t <= 1.0));
                    }
                    else // (pa,pb_proj) and (p1,p2) are parallel and disjoint
                    {
                        is_intersected=false;
                    }

                }
                else   // s_t != 0.0 and s_k != 0.0
                {

                    coord_k=Real((pa-p1)*(n_12))*1.0/Real(s_k);
                    coord_t=Real((p1-pa)*(n_proj))*1.0/Real(s_t);

                    is_intersected = ((coord_k > 0.0) && (coord_t >= 0.0 && coord_t <= 1.0));
                }

                if(is_intersected)
                {
                    if((!is_initialized) || (coord_k > coord_kmin))
                    {
                        indices.clear();
                        indices.push_back(t[1]);
                        indices.push_back(t[2]);
                        baryCoef=coord_t;
                        coord_kmin=coord_k;
                    }

                    is_initialized=true;
                }

                is_validated = is_validated || is_initialized;
            }



            if(!is_full_20)
            {
                /// Test of edge (p2,p0) :

                s_t = (p2-p0)*(n_proj);
                s_k = (pa-pb_proj)*(n_20);

                // s_t == 0.0 iff s_k == 0.0

                if(s_t==0.0) // (pa,pb_proj) and (p2,p0) are parallel
                {
                    if((p2-pa)*(n_proj)==0.0) // (pa,pb_proj) and (p2,p0) are on the same line
                    {
                        coord_test1 = (pa-p2)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa
                        coord_test2 = (pa-p0)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa

                        if(coord_test1>=0)
                        {
                            coord_k=coord_test1;
                            coord_t=0.0;
                        }
                        else
                        {
                            coord_k=coord_test2;
                            coord_t=1.0;
                        }

                        is_intersected = (coord_k > 0.0 && (coord_t >= 0.0 && coord_t <= 1.0));

                    }
                    else // (pa,pb_proj) and (p2,p0) are parallel and disjoint
                    {
                        is_intersected = false;
                    }
                }
                else // s_t != 0.0 and s_k != 0.0
                {
                    coord_k=Real((pa-p2)*(n_20))*1.0/Real(s_k);
                    coord_t=Real((p2-pa)*(n_proj))*1.0/Real(s_t);

                    is_intersected = ((coord_k > 0.0) && (coord_t >= 0.0 && coord_t <= 1.0));
                }

                if(is_intersected)
                {
                    if((!is_initialized) || (coord_k > coord_kmin))
                    {
                        indices.clear();
                        indices.push_back(t[2]);
                        indices.push_back(t[0]);
                        baryCoef=coord_t;
                        coord_kmin=coord_k;
                    }

                    is_initialized = true;
                }
                is_validated = is_validated || is_initialized;
            }


        }
        else
        {
            is_validated = false; // points a and b are projected to the same point on triangle t
        }
    }
    else
    {
        is_validated = false; // triangle t is flat
    }

    return is_validated;
}

// Computes the intersection of the segment from point a to point b and the triangle indexed by t
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::computeIntersectionsLineTriangle(bool is_entered,
    const sofa::type::Vec<3, Real>& a,
    const sofa::type::Vec<3, Real>& b,
    const TriangleID ind_t,
    sofa::type::vector<TriangleID>& indices,
    sofa::type::vector<Real>& vecBaryCoef,
    sofa::type::vector<Real>& vecCoordKmin) const
{
    Real EPS = 1e-10;
    Real baryCoef;
    Real coord_kmin;
    // HYP : point a is in triangle indexed by t
    // is_entered == true => indices.size() == 2
    TriangleID ind_first = 0;
    TriangleID ind_second = 0;

    if (indices.size() > 1)
    {
        ind_first = indices[0];
        ind_second = indices[1];
    }

    indices.clear();
    vecBaryCoef.clear();

    bool is_validated = false;

    const Triangle& t = this->m_topology->getTriangle(ind_t);
    const typename DataTypes::VecCoord& vect_c = (this->object->read(core::vec_id::read_access::position)->getValue());

    bool is_full_01 = (is_entered && ((t[0] == ind_first && t[1] == ind_second) || (t[1] == ind_first && t[0] == ind_second)));
    bool is_full_12 = (is_entered && ((t[1] == ind_first && t[2] == ind_second) || (t[2] == ind_first && t[1] == ind_second)));
    bool is_full_20 = (is_entered && ((t[2] == ind_first && t[0] == ind_second) || (t[0] == ind_first && t[2] == ind_second)));

    const typename DataTypes::Coord& c0 = vect_c[t[0]];
    const typename DataTypes::Coord& c1 = vect_c[t[1]];
    const typename DataTypes::Coord& c2 = vect_c[t[2]];

    sofa::type::Vec<3, Real> p0{ c0[0],c0[1],c0[2] };
    sofa::type::Vec<3, Real> p1{ c1[0],c1[1],c1[2] };
    sofa::type::Vec<3, Real> p2{ c2[0],c2[1],c2[2] };
    sofa::type::Vec<3, Real> pa{ a[0],a[1],a[2] };
    sofa::type::Vec<3, Real> pb{ b[0],b[1],b[2] };

    sofa::type::Vec<3, Real> v_normal = (p2 - p0).cross(p1 - p0);
    Real norm_v_normal = v_normal.norm(); // WARN : square root COST

    if (norm_v_normal >=EPS)
    {

        v_normal /= norm_v_normal;

        sofa::type::Vec<3, Real> v_ab = pb - pa;
        sofa::type::Vec<3, Real> v_ab_proj = v_ab - v_normal * dot(v_ab, v_normal); // projection (same values if incision in the plan)
        sofa::type::Vec<3, Real> pb_proj = v_ab_proj + pa;

        sofa::type::Vec<3, Real> v_01 = p1 - p0;
        sofa::type::Vec<3, Real> v_12 = p2 - p1;
        sofa::type::Vec<3, Real> v_20 = p0 - p2;

        sofa::type::Vec<3, Real> n_proj = v_ab_proj.cross(v_normal);

        sofa::type::Vec<3, Real> n_01 = v_01.cross(v_normal);
        sofa::type::Vec<3, Real> n_12 = v_12.cross(v_normal);
        sofa::type::Vec<3, Real> n_20 = v_20.cross(v_normal);

        Real norm2_v_ab_proj = v_ab_proj * (v_ab_proj); //dot product WARNING

        if (norm2_v_ab_proj >= EPS) // pb_proj != pa
        {
            Real coord_t = 0.0;
            Real coord_k = 0.0;

            coord_kmin = 0.0;

            Real coord_test1;
            Real coord_test2;

            Real s_t;
            Real s_k;

            if (!is_full_01)
            {
                coord_t = 0.0;
                coord_k = 0.0;
                bool is_intersected_01 = false;
                bool is_initialized_01 = false;
                /// Test of edge (p0,p1) :
                s_t = (p0 - p1) * n_proj;
                s_k = (pa - pb_proj) * n_01;
                if (s_t == 0.0) // (pa,pb_proj) and (p0,p1) are parallel
                {
                    if ((p0 - pa) * (n_proj) == 0.0) // (pa,pb_proj) and (p0,p1) are on the same line
                    {
                        coord_test1 = (pa - p0) * (pa - pb_proj) / norm2_v_ab_proj; // HYP : pb_proj != pa
                        coord_test2 = (pa - p1) * (pa - pb_proj) / norm2_v_ab_proj; // HYP : pb_proj != pa

                        if (coord_test1 >= 0)
                        {
                            coord_k = coord_test1;
                            coord_t = 0.0;
                        }
                        else
                        {
                            coord_k = coord_test2;
                            coord_t = 1.0;
                        }

                        is_intersected_01 = (coord_k > 0.0 && (coord_t >= 0.0 && coord_t <= 1.0));

                    }
                    else  // (pa,pb_proj) and (p0,p1) are parallel and disjoint
                    {
                        is_intersected_01 = false;
                    }
                }
                else // s_t != 0.0 and s_k != 0.0
                {
                    coord_k = Real((pa - p0) * (n_01)) * 1.0 / Real(s_k);
                    coord_t = Real((p0 - pa) * (n_proj)) * 1.0 / Real(s_t);

                    is_intersected_01 = ((coord_k > 0.0) && (coord_t >= 0.0 && coord_t <= 1.0));
                }

                if (is_intersected_01)
                {
                    if ((!is_initialized_01) || (coord_k > coord_kmin))
                    {
                        indices.push_back(t[0]);
                        indices.push_back(t[1]);
                        baryCoef = coord_t;
                        coord_kmin = coord_k;
                        vecBaryCoef.push_back(baryCoef);
                        vecCoordKmin.push_back(coord_kmin);
                    }

                    is_initialized_01 = true;
                }

                is_validated = is_validated || is_initialized_01;
            }



            if (!is_full_12)
            {
                coord_t = 0.0;
                coord_k = 0.0;
                bool is_intersected_12 = false;
                bool is_initialized_12 = false;
                /// Test of edge (p1,p2) :

                s_t = (p1 - p2) * (n_proj);
                s_k = (pa - pb_proj) * (n_12);

                // s_t == 0.0 iff s_k == 0.0

                if (s_t == 0.0) // (pa,pb_proj) and (p1,p2) are parallel
                {
                    if ((p1 - pa) * (n_proj) == 0.0) // (pa,pb_proj) and (p1,p2) are on the same line
                    {
                        coord_test1 = (pa - p1) * (pa - pb_proj) / norm2_v_ab_proj; // HYP : pb_proj != pa
                        coord_test2 = (pa - p2) * (pa - pb_proj) / norm2_v_ab_proj; // HYP : pb_proj != pa

                        if (coord_test1 >= 0)
                        {
                            coord_k = coord_test1;
                            coord_t = 0.0;
                        }
                        else
                        {
                            coord_k = coord_test2;
                            coord_t = 1.0;
                        }

                        is_intersected_12 = (coord_k > 0.0 && (coord_t >= 0.0 && coord_t <= 1.0));
                    }
                    else // (pa,pb_proj) and (p1,p2) are parallel and disjoint
                    {
                        is_intersected_12 = false;
                    }

                }
                else   // s_t != 0.0 and s_k != 0.0
                {

                    coord_k = Real((pa - p1) * (n_12)) * 1.0 / Real(s_k);
                    coord_t = Real((p1 - pa) * (n_proj)) * 1.0 / Real(s_t);

                    is_intersected_12 = ((coord_k > 0.0) && (coord_t >= 0.0 && coord_t <= 1.0));
                }

                if (is_intersected_12)
                {
                    if ((!is_initialized_12) || (coord_k > coord_kmin))
                    {
                        indices.push_back(t[1]);
                        indices.push_back(t[2]);
                        baryCoef = coord_t;
                        coord_kmin = coord_k;
                        vecBaryCoef.push_back(baryCoef);
                        vecCoordKmin.push_back(coord_kmin);
                    }

                    is_initialized_12 = true;
                }

                is_validated = is_validated || is_initialized_12;
            }



            if (!is_full_20)
            {
                coord_t = 0.0;
                coord_k = 0.0;
                bool is_intersected_20 = false;
                bool is_initialized_20 = false;
                /// Test of edge (p2,p0) :

                s_t = (p2 - p0) * (n_proj);
                s_k = (pa - pb_proj) * (n_20);

                // s_t == 0.0 iff s_k == 0.0

                if (s_t == 0.0) // (pa,pb_proj) and (p2,p0) are parallel
                {
                    if ((p2 - pa) * (n_proj) == 0.0) // (pa,pb_proj) and (p2,p0) are on the same line
                    {
                        coord_test1 = (pa - p2) * (pa - pb_proj) / norm2_v_ab_proj; // HYP : pb_proj != pa
                        coord_test2 = (pa - p0) * (pa - pb_proj) / norm2_v_ab_proj; // HYP : pb_proj != pa

                        if (coord_test1 >= 0)
                        {
                            coord_k = coord_test1;
                            coord_t = 0.0;
                        }
                        else
                        {
                            coord_k = coord_test2;
                            coord_t = 1.0;
                        }

                        is_intersected_20 = (coord_k > 0.0 && (coord_t >= 0.0 && coord_t <= 1.0));

                    }
                    else // (pa,pb_proj) and (p2,p0) are parallel and disjoint
                    {
                        is_intersected_20 = false;
                    }
                }
                else // s_t != 0.0 and s_k != 0.0
                {
                    coord_k = Real((pa - p2) * (n_20)) * 1.0 / Real(s_k);
                    coord_t = Real((p2 - pa) * (n_proj)) * 1.0 / Real(s_t);

                    is_intersected_20 = ((coord_k > 0.0) && (coord_t >= 0.0 && coord_t <= 1.0));
                }

                if (is_intersected_20)
                {
                    if ((!is_initialized_20) || (coord_k > coord_kmin))
                    {
                        indices.push_back(t[2]);
                        indices.push_back(t[0]);
                        baryCoef = coord_t;
                        coord_kmin = coord_k;
                        vecBaryCoef.push_back(baryCoef);
                        vecCoordKmin.push_back(coord_kmin);
                    }

                    is_initialized_20 = true;
                }
                is_validated = is_validated || is_initialized_20;
            }
        }
        else
        {
            is_validated = false; // points a and b are projected to the same point on triangle t
        }
    }
    else
    {
        is_validated = false; // triangle t is flat
    }

    return is_validated;
}


template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::computeSegmentTriangulationIntersections(
    const sofa::type::Vec<3, Real>& ptA,
    const sofa::type::Vec<3, Real>& ptB,
    const TriangleID ind_ta, const TriangleID ind_tb,
    sofa::type::vector< TriangleID >& triangles_list,
    sofa::type::vector< EdgeID >& edges_list,
    sofa::type::vector< Real >& coords_list) const
{
    if (ind_ta == ind_tb)
        return false;

    const typename DataTypes::VecCoord& coords = (this->object->read(core::vec_id::read_access::position)->getValue());
    sofa::type::Vec<3, Real> current_point = ptA;
    TriangleID current_triID = ind_ta;
    EdgeID current_edgeID = sofa::InvalidID;
    Real current_bary = 0;

    for (;;)
    {
        // Get the edges of a the current_triID [AB] that are intersected by Segment [AB]
        sofa::type::vector<EdgeID> intersectedEdges;
        sofa::type::vector<Real> baryCoefs;
        bool is_intersected = computeSegmentTriangleIntersectionInPlane(current_point, ptB, current_triID, intersectedEdges, baryCoefs);

        if (!is_intersected)
        {
            msg_error() << "No intersection can be found in method computeIncisionPath between input segment A: " << ptA << " - B: " << ptB << " and triangle: " << current_triID;
            return false;
        }

        // Handle start and end points       
        if (intersectedEdges.size() == 1) // Only one edge intersected, beginning or end
        {
            if (current_edgeID == intersectedEdges[0]) // reach end
            {
                triangles_list.push_back(current_triID);
                break;
            }

            // Beginning: new edge intersected
            current_edgeID = intersectedEdges[0];
            current_bary = baryCoefs[0];
        }
        else if (current_edgeID == sofa::InvalidID) // special case if cut start directly on an edge or a vertex, add it and init the loop with this edge
        {
            // To find the good direction at start, check intersection with middle of the triangle and ptB
            sofa::type::vector<EdgeID> tmp_intersectedEdges;
            sofa::type::vector<Real> tmp_baryCoefs;

            const typename DataTypes::Coord cG = computeTriangleCenter(ind_ta);
            const sofa::type::Vec<3, Real> pG = type::toVec3(cG);

            computeSegmentTriangleIntersectionInPlane(pG, ptB, current_triID, tmp_intersectedEdges, tmp_baryCoefs);
            if (tmp_intersectedEdges.size() != 1) // only one edge should be intersected to find the next edge in cut direction
            {
                msg_error() << "Impossible to determine cut direction at start due to snapping. Between input segment A: " << ptA << " - B: " << ptB << " and triangle: " << current_triID;
                return false;
            }

            sofa::Size nbrInter = intersectedEdges.size();
            // find next edge in correct initial intersection
            sofa::Index curLocalId = sofa::InvalidID;
            for (sofa::Index j = 0; j < nbrInter; j++)
            {
                if (tmp_intersectedEdges[0] == intersectedEdges[j]) {
                    curLocalId = j;
                    break;
                }
            }
            sofa::Index nextLocalId = (curLocalId + 1) % nbrInter;

            edges_list.push_back(intersectedEdges[nextLocalId]);
            coords_list.push_back(baryCoefs[nextLocalId]);
            current_edgeID = intersectedEdges[nextLocalId];
        }


        if (intersectedEdges.size() == 2) // triangle fully traversed, look for the next edge
        {
            if (current_edgeID == intersectedEdges[0])
            {
                current_edgeID = intersectedEdges[1];
                current_bary = baryCoefs[1];
            }
            else if (current_edgeID == intersectedEdges[1])
            {
                current_edgeID = intersectedEdges[0];
                current_bary = baryCoefs[0];
            }
            else
            {
                msg_error() << "Previous edge id: " << current_edgeID << " can't be found in the intersection between input segment A: " << ptA << " - B: " << ptB << " and triangle: " << current_triID;
                return false;
            }
        }
        else if (intersectedEdges.size() == 3) // triangle fully traversed and going in/out through a vertex
        {
            // double check that one edge is the previous one and the 2 others have a baryCoef equal to 0 or 1
            sofa::Index curLocalId = sofa::InvalidID;
            sofa::Index localNoSnapId = sofa::InvalidID;
            sofa::Size nbrV = 0;
            for (sofa::Index j = 0; j < 3; j++)
            {
                if (current_edgeID == intersectedEdges[j]) {
                    curLocalId = j;
                }

                if (baryCoefs[j] == 0 || baryCoefs[j] == 1)
                    nbrV++;
                else
                    localNoSnapId = j;
            }

            if (curLocalId == sofa::InvalidID)
            {
                msg_error() << "Previous edge id: " << current_edgeID << " can't be found in the intersectionbetween input segment A: " << ptA << " - B: " << ptB << " and triangle: " << current_triID;
                return false;
            }

            if (nbrV != 2)
            {
                msg_error() << "3 intersections have been found in method computeIncisionPath between input segment A: " << ptA << " - B: " << ptB << " and triangle: " << current_triID << ". But the intersection is not going through a vertex. This is not possible!";
                return false;
            }

            if (curLocalId == localNoSnapId) // means current edge is not cut at a vertex. we go out at the opposite vertex.
            {
                // in case of going through a vertex, arbitrary take the next edge in the list
                current_edgeID = intersectedEdges[(curLocalId + 1) % 3];
                current_bary = baryCoefs[(curLocalId + 1) % 3];
            }
            else // we are going inside this triangle from a vertex. We need to go out on the opposite edge
            {
                current_edgeID = intersectedEdges[localNoSnapId];
                current_bary = baryCoefs[localNoSnapId];
            }
        }
        else if (intersectedEdges.size() > 3)
        {
            msg_error() << "More than 3 intersections have been found in method computeIncisionPath between input segment A: " << ptA << " - B: " << ptB << " and triangle: " << current_triID << ". This is not possible!";
            return false;
        }

        // Add current triangle into the list of intersected triangles
        triangles_list.push_back(current_triID);

        // Add current edge and barycoef to the intersected lists
        edges_list.push_back(current_edgeID);
        coords_list.push_back(current_bary);

        if (current_triID == ind_tb) // reach end
            break;

        // Update start interaction point
        const Edge& edge = this->m_topology->getEdge(current_edgeID);

        const typename DataTypes::Coord& c0 = coords[edge[0]];
        const typename DataTypes::Coord& c1 = coords[edge[1]];
        sofa::type::Vec<3, Real> p0 = { c0[0], c0[1], c0[2] };
        sofa::type::Vec<3, Real> p1 = { c1[0], c1[1], c1[2] };

        // update pA with the intersection point on the new edge 
        sofa::type::Vec<3, Real> newIntersection = p0 * current_bary + p1 * (1.0 - current_bary);
        current_point = current_point + (newIntersection - current_point) * 0.8; // add a small threshold to make sure point is out of next triangle


        // search for next triangle to be intersected
        sofa::type::vector< TriangleID > triAE = this->m_topology->getTrianglesAroundEdge(current_edgeID);
        if (triAE.size() == 1)
        {
            break;
        }
        else if (triAE.size() == 2)
        {
            if (triAE[0] == current_triID)
                current_triID = triAE[1];
            else
                current_triID = triAE[0];
        }
        else
        {
            msg_error() << "More than 2 triangles found around edge: " << current_edgeID << ". Non - Manifold triangulation is not supported by computeIncisionPath method";
            return false;
        }
    }

    return !coords_list.empty();
}


template<class DataTypes>
type::vector< std::shared_ptr<PointToAdd> > TriangleSetGeometryAlgorithms< DataTypes >::computeIncisionPath(const sofa::type::Vec<3, Real>& ptA, const sofa::type::Vec<3, Real>& ptB,
    const TriangleID ind_ta, const TriangleID ind_tb, Real snapThreshold, Real snapThresholdBorder) const
{
    // Get points coordinates
    const typename DataTypes::VecCoord& vect_c = (this->object->read(core::vec_id::read_access::position)->getValue());

    // 1. Get access to buffers and format Cut Path data
    const auto& triangles = m_container->getTriangles();
    const auto& edges = m_container->getEdges();
    const auto& triAEdges = m_container->getTrianglesAroundEdgeArray();
    const auto& edgesInTri = m_container->getEdgesInTriangleArray();
    auto nbrPoints = PointID(m_container->getNbPoints());

    type::fixed_array < Vec3, 2> pathPts = { ptA , ptB };
    type::fixed_array< TriangleID, 2> triIds = { ind_ta , ind_tb };
    type::fixed_array< Triangle, 2> theTris = { triangles[triIds[0]], triangles[triIds[1]] };
    type::fixed_array < Vec3, 2> _coefsTris;
    _coefsTris[0] = computeTriangleBarycentricCoordinates(ind_ta, ptA);
    _coefsTris[1] = computeTriangleBarycentricCoordinates(ind_tb, ptB);
    const sofa::type::Vec3 cutPath = ptB - ptA;

    // 2. Check if snapping is needed at first and last points. Snapping on point is more important than snapping on edge   
    type::fixed_array< PointID, 2> snapVertexStatus = { InvalidID , InvalidID };
    type::fixed_array< PointID, 2> snapEdgeStatus = { InvalidID , InvalidID }; 

    // check possible snap on cut bounds based on barycentric coordinates
    for (unsigned int i = 0; i < 2; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            if (_coefsTris[i][j] > snapThresholdBorder) // snap to point at start
            {
                snapVertexStatus[i] = j;
            }

            if (_coefsTris[i][j] < (1_sreal - snapThresholdBorder)) // otherwise snap to edge at start
            {
                snapEdgeStatus[i] = j;// edgesInTri[triIds[0]][(i + 3) % 3];
            }
        }
    }

    // Apply snapping and compute new start/end points
    type::fixed_array < sofa::geometry::ElementType, 2> _elemBorders;
    type::fixed_array < bool, 2> _borderSplit;
    for (unsigned int i = 0; i < 2; ++i)
    {
        if (snapVertexStatus[i] != InvalidID)
        {
            // snap Vertex is prioritary
            const PointID localVId = snapVertexStatus[i];
            const PointID vId = theTris[i][localVId];           

            pathPts[i] = type::toVec3(vect_c[vId]);
            _elemBorders[i] = sofa::geometry::ElementType::POINT;

            // check if point need to be subdivided at start: yes if on border of mesh, otherwise false.
            int nextTriId = -1;
            if (i == 0) // cut path start
            {
                nextTriId = this->getTriangleInDirection(vId, -cutPath);
            }
            else // cut path end
            {
                nextTriId = this->getTriangleInDirection(vId, cutPath);
            }

            if (nextTriId != -1) // means there is a triangle on the other side of the point. Point should not be splitted
                _borderSplit[i] = false;
            else // on the border or in middle of a T junction, need to split
                _borderSplit[i] = true;

            //std::cout << "Snap Vertex needed here: " << vId << " with split: " << _borderSplit[i] << std::endl;
        }
        else if (snapEdgeStatus[i] != InvalidID) // snap edge
        {
            PointID localVId = snapEdgeStatus[i];
            const EdgeID edgeId = edgesInTri[triIds[i]][(localVId + 3) % 3];
            const Edge& edge = edges[edgeId];
            
            type::Vec2 newCoefs;
            if (edge[0] == theTris[i][(localVId + 1) % 3])
            {
                newCoefs[0] = _coefsTris[i][(localVId + 1) % 3];
                newCoefs[1] = _coefsTris[i][(localVId + 2) % 3];
            }
            else
            {
                newCoefs[0] = _coefsTris[i][(localVId + 2) % 3];
                newCoefs[1] = _coefsTris[i][(localVId + 1) % 3];
            }
            SReal sum = newCoefs[0] + newCoefs[1];
            newCoefs[0] = newCoefs[0] / sum;
            newCoefs[1] = newCoefs[1] / sum;

            pathPts[i] = type::toVec3(vect_c[edge[0]] * newCoefs[0] + vect_c[edge[1]] * newCoefs[1]);

            _elemBorders[i] = sofa::geometry::ElementType::EDGE;

            // check if point need to be subdivided at start: yes if on border of mesh, otherwise false.
            if (triAEdges[edgeId].size() == 1) // only one edge. means on border
            {
                _borderSplit[i] = true;
            }
            else 
            {
                _borderSplit[i] = false;
            }

        }
        else
        {
            _elemBorders[i] = sofa::geometry::ElementType::TRIANGLE;
            _borderSplit[i] = false;
        }
    }

    // 3. compute incision path through triangles and edges
    sofa::type::vector< TriangleID > triangles_list;
    sofa::type::vector< EdgeID > edges_list;
    sofa::type::vector< Real > coords_list;
    type::vector< std::shared_ptr<PointToAdd> > _pointsToAdd;

    bool validPath = computeSegmentTriangulationIntersections(pathPts[0], pathPts[1], ind_ta, ind_tb, triangles_list, edges_list, coords_list);
    if (!validPath)
        return _pointsToAdd;

    // 4. post processing the list of intersected edges if snapping is requested
    std::set <PointID> psnap;
    for (unsigned int i = 0; i < edges_list.size(); ++i)
    {
        const Edge& edge = edges[edges_list[i]];
        if (coords_list[i] > snapThreshold)
            psnap.insert(edge[0]);
        else if (1.0 - coords_list[i] > snapThreshold)
            psnap.insert(edge[1]);
    }

    for (unsigned int i = 0; i < edges_list.size(); ++i)
    {
        const Edge& edge = edges[edges_list[i]];
        if (psnap.find(edge[0]) != psnap.end())
            coords_list[i] = 1.0;
        else if (psnap.find(edge[1]) != psnap.end())
            coords_list[i] = 0.0;
    }

     
    // check first point here:
    if (_elemBorders[0] == sofa::geometry::ElementType::TRIANGLE)
    {
        type::vector<SReal> _coefs = { _coefsTris[0][0], _coefsTris[0][1], _coefsTris[0][2] };
        type::vector<PointID> _ancestors = { theTris[0][0] , theTris[0][1], theTris[0][2] };
        PointID uniqID = getUniqueId(theTris[0][0], theTris[0][1], theTris[0][2]);

        std::shared_ptr<PointToAdd> PTA = std::make_shared<PointToAdd>(uniqID, nbrPoints, _ancestors, _coefs);
        PTA->m_ancestorType = sofa::geometry::ElementType::TRIANGLE;
        PTA->m_ownerId = triIds[0];
        _pointsToAdd.push_back(PTA);
        nbrPoints = nbrPoints + PTA->getNbrNewPoint();
    }

    // create PointToAdd from edges
    for (unsigned int i = 0; i < edges_list.size(); ++i)
    {
        const Edge& edge = edges[edges_list[i]];
        type::vector<SReal> _coefs = { coords_list[i], 1.0 - coords_list[i] };
        type::vector<PointID> _ancestors = { edge[0], edge[1] };

        PointID uniqID = getUniqueId(edge[0], edge[1]);
        std::shared_ptr<PointToAdd> PTA = std::make_shared<PointToAdd>(uniqID, nbrPoints, _ancestors, _coefs);
        if (coords_list[i] > 1.0 - EQUALITY_THRESHOLD) // snap on this point
        {
            PTA->m_ancestorType = sofa::geometry::ElementType::POINT;
            PTA->m_ownerId = edge[0];
        }
        else if (coords_list[i] < EQUALITY_THRESHOLD) // snap on the other point
        {
            PTA->m_ancestorType = sofa::geometry::ElementType::POINT;
            PTA->m_ownerId = edge[1];
        }
        else 
        {
            PTA->m_ancestorType = sofa::geometry::ElementType::EDGE;
            PTA->m_ownerId = edges_list[i];
        }

        // check if split on border is needed
        if (i == 0 && _elemBorders[0] != sofa::geometry::ElementType::TRIANGLE && _borderSplit[0] == false)
            PTA->updatePointIDForDuplication(false);
        else if (i == edges_list.size()-1 && _elemBorders[1] != sofa::geometry::ElementType::TRIANGLE && _borderSplit[1] == false)
            PTA->updatePointIDForDuplication(false);
        else
            PTA->updatePointIDForDuplication(true);
        
        // Adding new PTA to the vector
        if (PTA->m_ancestorType == sofa::geometry::ElementType::POINT) // check to add it only once
        {
            bool found = false;
            for (const auto& ptAdded : _pointsToAdd)
            {
                if (ptAdded->m_ancestorType == sofa::geometry::ElementType::POINT && ptAdded->m_ownerId == PTA->m_ownerId) // already registered
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                _pointsToAdd.push_back(PTA);
                nbrPoints = nbrPoints + PTA->getNbrNewPoint();
            }
        }
        else
        {
            _pointsToAdd.push_back(PTA);
            nbrPoints = nbrPoints + PTA->getNbrNewPoint();
        }
    }

    // check last point here:
    if (_elemBorders[1] == sofa::geometry::ElementType::TRIANGLE)
    {
        type::vector<SReal> _coefs = { _coefsTris[1][0], _coefsTris[1][1], _coefsTris[1][2] };
        type::vector<PointID> _ancestors = { theTris[1][0] , theTris[1][1], theTris[1][2] };
        PointID uniqID = getUniqueId(theTris[1][0], theTris[1][1], theTris[1][2]);

        std::shared_ptr<PointToAdd> PTA = std::make_shared<PointToAdd>(uniqID, nbrPoints, _ancestors, _coefs);
        PTA->m_ancestorType = sofa::geometry::ElementType::TRIANGLE;
        PTA->m_ownerId = triIds[1];
        _pointsToAdd.push_back(PTA);
    }

    return _pointsToAdd;
}


// Computes the list of points (edge,coord) intersected by the segment from point a to point b
// and the triangular mesh
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::computeIntersectedPointsList(const PointID last_point,
        const sofa::type::Vec<3,Real>& a,
        const sofa::type::Vec<3,Real>& b,
        TriangleID& ind_ta,
        TriangleID& ind_tb,
        sofa::type::vector< TriangleID > &triangles_list,
        sofa::type::vector< EdgeID> &edges_list,
        sofa::type::vector< Real >& coords_list,
        bool& is_on_boundary) const
{

    bool is_validated=true;
    bool is_intersected=true;

    sofa::type::Vec<3,Real> c_t_test = a;

    is_on_boundary = false;

    sofa::type::vector<PointID> indices;

    Real coord_t=0.0;
    Real coord_k=0.0;
    Real coord_k_test=0.0;
    Real dist_min=0.0;

    sofa::type::Vec<3,Real> p_current=a;

    TriangleID ind_t_current=ind_ta;
    EdgeID ind_edge;
    PointID ind_index;
    TriangleID ind_triangle = ind_ta;
    is_intersected=computeSegmentTriangleIntersection(false, p_current, b, ind_t_current, indices, coord_t, coord_k);


    // In case the ind_t is not the good one.
    if ( (!is_intersected || indices[0] == last_point || indices[1] == last_point) && (last_point != sofa::InvalidID))
    {

        const sofa::type::vector< TriangleID >& shell = this->m_topology->getTrianglesAroundVertex (last_point);

        for (size_t i = 0; i<shell.size(); i++)
        {
            if (shell [i] != ind_t_current)
                is_intersected=computeSegmentTriangleIntersection(false, p_current, b, shell[i], indices, coord_t, coord_k);

            if (is_intersected && indices[0] != last_point && indices[1] != last_point)
            {
                ind_t_current = shell[i];
                ind_ta = ind_t_current;
                break;
            }
        }
    }

    if (ind_ta == ind_tb)
    {
        msg_warning() << "TriangleSetTopology.inl : Cut is not reached because inputs elements are the same element." ;
        return false;
    }


    dmsg_info() << "*********************************" << msgendl
                << "* computeIntersectedPointsList * " << msgendl
                << "ind_t_current: " << ind_t_current << msgendl
                << "p_current: " << p_current << msgendl
                << "coord_t: " << coord_t << msgendl
                << "coord_k: " << coord_k << msgendl
                << "indices: " << indices << msgendl
                << "last_point: " << last_point << msgendl
                << "a: " << a << msgendl
                << "b: " << b << msgendl
                << "is_intersected: "<< is_intersected << msgendl
                << "*********************************" ;
    
    coord_k_test=coord_k;
    dist_min=(b-a)*(b-a);

    while((coord_k_test<1.0 && is_validated) && is_intersected)
    {
        ind_edge=this->m_topology->getEdgeIndex(indices[0],indices[1]);
        edges_list.push_back(ind_edge);
        triangles_list.push_back(ind_t_current);
        if (this->m_topology->getEdge(ind_edge)[0] == indices[0])
            coords_list.push_back(coord_t);
        else
            coords_list.push_back(1.0-coord_t);

        const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

        sofa::type::Vec<3,Real> c_t_current; // WARNING : conversion from 'Real' to 'float', possible loss of data ! // typename DataTypes::Coord
        c_t_current[0]=(Real) ((1.0-coord_t)*((Real) (vect_c[indices[0]][0]))+coord_t*((Real) (vect_c[indices[1]][0])));
        c_t_current[1]=(Real) ((1.0-coord_t)*((Real) (vect_c[indices[0]][1]))+coord_t*((Real) (vect_c[indices[1]][1])));
        c_t_current[2]=(Real) ((1.0-coord_t)*((Real) (vect_c[indices[0]][2]))+coord_t*((Real) (vect_c[indices[1]][2])));

        p_current=c_t_current;

        sofa::type::Vec<3,Real> p_t_aux;
        p_t_aux[0] = (Real) (c_t_current[0]);
        p_t_aux[1] = (Real) (c_t_current[1]);
        p_t_aux[2] = (Real) (c_t_current[2]);




        if(coord_t==0.0 || coord_t==1.0) // current point indexed by ind_t_current is on a vertex
        {
            if(coord_t==0.0)
            {
                ind_index=indices[0];
            }
            else // coord_t==1.0
            {
                ind_index=indices[1];
            }

            if (this->m_topology->getNbPoints() >0)
            {
                sofa::type::vector< TriangleID > shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundVertex(ind_index));
                ind_triangle=shell[0];
                TriangleID i=0;
                bool is_test_init=false;

                TriangleID ind_from = ind_t_current;

                if(shell.size()>1) // at leat one neighbor triangle which is not indexed by ind_t_current
                {
                    is_on_boundary=false;

                    while(i < shell.size())
                    {
                        if(shell[i] != ind_from)
                        {
                            ind_triangle=shell[i];

                            const Triangle &t=this->m_topology->getTriangle(ind_triangle);

                            const typename DataTypes::Coord& c0=vect_c[t[0]];
                            const typename DataTypes::Coord& c1=vect_c[t[1]];
                            const typename DataTypes::Coord& c2=vect_c[t[2]];

                            sofa::type::Vec<3,Real> p0_aux;
                            p0_aux[0] = (Real) (c0[0]);
                            p0_aux[1] = (Real) (c0[1]);
                            p0_aux[2] = (Real) (c0[2]);
                            sofa::type::Vec<3,Real> p1_aux;
                            p1_aux[0] = (Real) (c1[0]);
                            p1_aux[1] = (Real) (c1[1]);
                            p1_aux[2] = (Real) (c1[2]);
                            sofa::type::Vec<3,Real> p2_aux;
                            p2_aux[0] = (Real) (c2[0]);
                            p2_aux[1] = (Real) (c2[1]);
                            p2_aux[2] = (Real) (c2[2]);

                            is_intersected=computeSegmentTriangleIntersection(true, p_current, b, ind_triangle, indices, coord_t, coord_k);

                            if(is_intersected)
                            {
                                sofa::type::Vec<3,Real> c_t_test; // WARNING : conversion from 'Real' to 'float', possible loss of data ! // typename DataTypes::Coord
                                c_t_test[0]=(Real) ((1.0-coord_t)*((Real) (vect_c[indices[0]][0]))+coord_t*((Real) (vect_c[indices[1]][0])));
                                c_t_test[1]=(Real) ((1.0-coord_t)*((Real) (vect_c[indices[0]][1]))+coord_t*((Real) (vect_c[indices[1]][1])));
                                c_t_test[2]=(Real) ((1.0-coord_t)*((Real) (vect_c[indices[0]][2]))+coord_t*((Real) (vect_c[indices[1]][2])));

                                Real dist_test=(b-c_t_test)*(b-c_t_test);

                                if(is_test_init)
                                {
                                    if(dist_test<dist_min && coord_k<=1) //dist_test<dist_min
                                    {
                                        coord_k_test=coord_k;
                                        dist_min=dist_test;
                                        ind_t_current=ind_triangle;
                                    }
                                }
                                else
                                {
                                    is_test_init=true;
                                    coord_k_test=coord_k;
                                    dist_min=dist_test;
                                    ind_t_current=ind_triangle;
                                }
                            }
                        }

                        i=i+1;
                    }

                    is_intersected=is_test_init;

                }
                else
                {
                    is_on_boundary=true;
                    is_validated=false;
                }
            }
            else
            {
                is_validated=false;
            }
        }
        else // current point indexed by ind_t_current is on an edge, but not on a vertex
        {
            ind_edge=this->m_topology->getEdgeIndex(indices[0],indices[1]);

            if (this->m_topology->getNbEdges()>0)
            {
                sofa::type::vector< TriangleID > shell =(sofa::type::vector< TriangleID >) (this->m_topology->getTrianglesAroundEdge(ind_edge));

                ind_triangle=shell[0];
                TriangleID i=0;

                bool is_test_init=false;

                TriangleID ind_from = ind_t_current;

                if(shell.size()>0) // at leat one neighbor triangle which is not indexed by ind_t_current
                {
                    is_on_boundary=false;

                    while(i < shell.size())
                    {
                        if(shell[i] != ind_from)
                        {
                            ind_triangle=shell[i];

                            const Triangle &t=this->m_topology->getTriangle(ind_triangle);

                            const typename DataTypes::Coord& c0=vect_c[t[0]];
                            const typename DataTypes::Coord& c1=vect_c[t[1]];
                            const typename DataTypes::Coord& c2=vect_c[t[2]];

                            sofa::type::Vec<3,Real> p0_aux;
                            p0_aux[0] = (Real) (c0[0]);
                            p0_aux[1] = (Real) (c0[1]);
                            p0_aux[2] = (Real) (c0[2]);
                            sofa::type::Vec<3,Real> p1_aux;
                            p1_aux[0] = (Real) (c1[0]);
                            p1_aux[1] = (Real) (c1[1]);
                            p1_aux[2] = (Real) (c1[2]);
                            sofa::type::Vec<3,Real> p2_aux;
                            p2_aux[0] = (Real) (c2[0]);
                            p2_aux[1] = (Real) (c2[1]);
                            p2_aux[2] = (Real) (c2[2]);

                            is_intersected=computeSegmentTriangleIntersection(true, p_current, b, ind_triangle, indices, coord_t, coord_k);

                            if(is_intersected)
                            {
                                //Vec<3,Real> c_t_test; // WARNING : conversion from 'Real' to 'float', possible loss of data ! // typename DataTypes::Coord
                                c_t_test[0]=(Real) ((1.0-coord_t)*((Real) (vect_c[indices[0]][0]))+coord_t*((Real) (vect_c[indices[1]][0])));
                                c_t_test[1]=(Real) ((1.0-coord_t)*((Real) (vect_c[indices[0]][1]))+coord_t*((Real) (vect_c[indices[1]][1])));
                                c_t_test[2]=(Real) ((1.0-coord_t)*((Real) (vect_c[indices[0]][2]))+coord_t*((Real) (vect_c[indices[1]][2])));

                                Real dist_test=(b-c_t_test)*(b-c_t_test);

                                if(is_test_init)
                                {
                                    if(dist_test<dist_min && coord_k<=1) //dist_test<dist_min
                                    {
                                        coord_k_test=coord_k;
                                        dist_min=dist_test;
                                        ind_t_current=ind_triangle;
                                    }
                                }
                                else
                                {
                                    is_test_init=true;
                                    coord_k_test=coord_k;
                                    dist_min=dist_test;
                                    ind_t_current=ind_triangle;
                                }
                            }
                        }
                        i=i+1;
                    }
                    is_intersected=is_test_init;
                }
                else
                {
                    is_on_boundary=true;
                    is_validated=false;
                }
            }
            else
            {
                is_validated=false;
            }
        }
    }

    if (ind_tb == sofa::InvalidID)
        ind_tb = ind_triangle;

    bool is_reached = (ind_tb==ind_triangle && coord_k_test>=1.0);

    if(is_reached)
    {
        dmsg_info() << "TriangleSetTopology.inl : Cut is reached" ;
    }

    if(is_on_boundary)
    {
        dmsg_info() << "TriangleSetTopology.inl : Cut meets a mesh boundary" ;
    }

    if(!is_reached && !is_on_boundary)
    {
        dmsg_info() << "INFO_print - TriangleSetTopology.inl : Cut is not reached" ;
    }

    return (is_reached && is_validated && is_intersected); // b is in triangle indexed by ind_t_current
}



template <typename DataTypes>
bool TriangleSetGeometryAlgorithms<DataTypes>::computeIntersectedObjectsList (const PointID last_point, const Vec3& pointA, const Vec3& pointB,
    TriangleID& ind_triA, TriangleID& ind_triB,
    sofa::type::vector< sofa::geometry::ElementType >& intersected_topoElements,
    sofa::type::vector< ElemID >& intersected_indices,
    sofa::type::vector< Vec3 >& intersected_barycoefs) const
{
    //// QUICK FIX TO USE THE NEW PATH DECLARATION (WITH ONLY EDGES COMING FROM PREVIOUS FUNCTION)
    //// ** TODO: create the real function handle different objects intersection **
    // QUICK FIX for fracture: border points (a and b) can be a point.

    // Output declarations
    sofa::type::vector<TriangleID> triangles_list;
    sofa::type::vector<EdgeID> edges_list;
    sofa::type::vector< Real > edge_barycoefs_list;
    bool is_on_boundary = false;
    // using old function:
    bool pathOK = this->computeIntersectedPointsList(last_point, pointA, pointB, ind_triA, ind_triB, triangles_list, edges_list, edge_barycoefs_list, is_on_boundary);
    dmsg_info() << "*********************************" << msgendl
                << "* computeIntersectedObjectsList *" << msgendl
                << "last_point: " << last_point << msgendl
                << "pointA: " << pointA << msgendl
                << "pointB: " << pointB << msgendl
                << "triangles_list: "<< triangles_list << msgendl
                << "edges_list: "<< edges_list << msgendl
                << "edge_barycoefs_list: "<< edge_barycoefs_list << msgendl
                << "*********************************" ;

    if (!pathOK)
        return false;

    // creating new declaration path:
    sofa::type::Vec<3,Real> baryCoords;

    // 1 - First point a (for the moment: always a point in a triangle)
    if (last_point != sofa::InvalidID)
    {
        intersected_topoElements.push_back (sofa::geometry::ElementType::POINT);
        intersected_indices.push_back (last_point);
        const typename DataTypes::VecCoord& realC =(this->object->read(core::vec_id::read_access::position)->getValue());
        for (unsigned int i = 0; i<3; i++)
            baryCoords[i]=realC[last_point][i];
    }
    else
    {
        auto coefs_a = computeTriangleBarycoefs (ind_triA, pointA);
        intersected_topoElements.push_back (sofa::geometry::ElementType::TRIANGLE);
        intersected_indices.push_back (ind_triA);
        for (unsigned int i = 0; i<3; i++)
            baryCoords[i]=coefs_a[i];
    }
    intersected_barycoefs.push_back (baryCoords);


    // 2 - All edges intersected (only edges for now)
    for (size_t i = 0; i< edges_list.size(); i++)
    {
        intersected_topoElements.push_back (sofa::geometry::ElementType::EDGE);
        intersected_indices.push_back (edges_list[i]);

        baryCoords[0] = edge_barycoefs_list[i];
        baryCoords[1] = 0.0; // or 1 - edge_barycoefs_list[i] ??
        baryCoords[2] = 0.0;

        intersected_barycoefs.push_back (baryCoords);
    }

    // 3 - Last point b (for the moment: always a point in a triangle)
    auto coefs_b = computeTriangleBarycoefs (ind_triB, pointB);
    bool isOnPoint = false;
    for (unsigned int i = 0; i<3; i++)
        if (coefs_b[i] > 0.9999 )
        {
            intersected_topoElements.push_back (sofa::geometry::ElementType::POINT);
            intersected_indices.push_back (this->m_topology->getTriangle (ind_triB)[i]);
            isOnPoint = true;
            break;
        }

    if (!isOnPoint)
    {
        intersected_topoElements.push_back (sofa::geometry::ElementType::TRIANGLE);
        intersected_indices.push_back (ind_triB);
    }
    for (unsigned int i = 0; i<3; i++)
        baryCoords[i]=coefs_b[i];

    intersected_barycoefs.push_back (baryCoords);
 

    return true;
}


/// Get the triangle in a given direction from a point.
template <typename DataTypes>
int TriangleSetGeometryAlgorithms<DataTypes>::getTriangleInDirection(PointID p, const sofa::type::Vec<3,Real>& dir) const
{
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());
    const sofa::type::vector<TriangleID> &shell=this->m_topology->getTrianglesAroundVertex(p);
    sofa::type::Vec<3,Real> dtest = dir;
    for (size_t i=0; i<shell.size(); ++i)
    {
        unsigned int ind_t = shell[i];
        const Triangle &t=this->m_topology->getTriangle(ind_t);

        const typename DataTypes::Coord& c0=vect_c[t[0]];
        const typename DataTypes::Coord& c1=vect_c[t[1]];
        const typename DataTypes::Coord& c2=vect_c[t[2]];

        sofa::type::Vec<3,Real> p0;
        p0[0] = (Real) (c0[0]);
        p0[1] = (Real) (c0[1]);
        p0[2] = (Real) (c0[2]);
        sofa::type::Vec<3,Real> p1;
        p1[0] = (Real) (c1[0]);
        p1[1] = (Real) (c1[1]);
        p1[2] = (Real) (c1[2]);
        sofa::type::Vec<3,Real> p2;
        p2[0] = (Real) (c2[0]);
        p2[1] = (Real) (c2[1]);
        p2[2] = (Real) (c2[2]);

        sofa::type::Vec<3,Real> e1, e2;
        if (t[0] == p) { e1 = p1-p0; e2 = p2-p0; }
        else if (t[1] == p) { e1 = p2-p1; e2 = p0-p1; }
        else { e1 = p0-p2; e2 = p1-p2; }

        sofa::type::Vec<3,Real> v_normal = (e2).cross(e1);

        if(v_normal.norm2() > 1e-20)
        {
            sofa::type::Vec<3,Real> n_01 = e1.cross(v_normal);
            sofa::type::Vec<3,Real> n_02 = e2.cross(v_normal);

            Real v_01 = (Real) ((dtest)*(n_01));
            Real v_02 = (Real) ((dtest)*(n_02));

            const bool is_inside = (v_01 >= 0.0) && (v_02 < 0.0);
            if (is_inside) return ind_t;
        }
    }
    return -1;
}


template <typename DataTypes>
void TriangleSetGeometryAlgorithms<DataTypes>::reorderTrianglesOrientationFromNormals()
{
    sofa::type::Vec<3,Real> firstNormal = computeTriangleNormal(0);

    if (p_flipNormals.getValue())
        firstNormal = -firstNormal;

    sofa::type::vector<TriangleID> _neighTri = this->m_topology->getElementAroundElement(0);
    sofa::type::vector<TriangleID> _neighTri2, buffK, buffKK;
    size_t cpt_secu = 0, max = this->m_topology->getNbTriangles();
    sofa::type::Vec<3,Real> triNormal;
    bool pair = true;


    while (!_neighTri.empty() && cpt_secu < max)
    {
        for (size_t i=0; i<_neighTri.size(); ++i)
        {
            TriangleID triId = _neighTri[i];
            triNormal = this->computeTriangleNormal(triId);
            Real prod = (firstNormal*triNormal)/(firstNormal.norm()*triNormal.norm());
            if (prod < 0.15) //change orientation
                this->m_topology->reOrientateTriangle(triId);
        }

        _neighTri2 = this->m_topology->getElementAroundElements(_neighTri);

        if (pair)
        {
            buffK = _neighTri;
            pair = false;

            _neighTri.clear();
            for (size_t i=0; i<_neighTri2.size(); ++i)
            {
                bool find = false;
                TriangleID id = _neighTri2[i];
                for (size_t j=0; j<buffKK.size(); ++j)
                    if (id == buffKK[j])
                    {
                        find = true;
                        break;
                    }

                if (!find)
                    _neighTri.push_back(id);
            }
        }
        else
        {
            buffKK = _neighTri;
            pair = true;

            _neighTri.clear();
            for (size_t i=0; i<_neighTri2.size(); ++i)
            {
                bool find = false;
                TriangleID id = _neighTri2[i];
                for (size_t j=0; j<buffK.size(); ++j)
                    if (id == buffK[j])
                    {
                        find = true;
                        break;
                    }

                if (!find)
                    _neighTri.push_back(id);
            }
        }


        dmsg_info() << "_neighTri: "<< _neighTri << msgendl
                    << "_neighTri2: "<< _neighTri2 <<msgendl
                    << "buffk: "<< buffK <<msgendl
                    << "buffkk: "<< buffKK <<msgendl ;
        cpt_secu++;
    }

    if(cpt_secu == max)
        msg_warning() << "TriangleSetGeometryAlgorithms: reorder triangle orientation reach security end of loop." ;

    return;
}


template <class DataTypes>
bool TriangleSetGeometryAlgorithms<DataTypes>::mustComputeBBox() const
{
    return ((showTriangleIndices.getValue() || _draw.getValue()) && this->m_topology->getNbTriangles() != 0) || Inherit1::mustComputeBBox();
}

template<class Real>
bool is_point_in_triangle(const sofa::type::Vec<3,Real>& p, const sofa::type::Vec<3,Real>& a, const sofa::type::Vec<3,Real>& b, const sofa::type::Vec<3,Real>& c)
{
    const Real ZERO = 1e-6;

    sofa::type::Vec<3,Real> ptest = p;
    sofa::type::Vec<3,Real> p0 = a;
    sofa::type::Vec<3,Real> p1 = b;
    sofa::type::Vec<3,Real> p2 = c;

    sofa::type::Vec<3,Real> v_normal = (p2-p0).cross(p1-p0);

    Real norm_v_normal = v_normal*(v_normal);
    if(norm_v_normal > ZERO)
    {
        if(fabs((ptest-p0)*(v_normal)) < ZERO) // p is in the plane defined by the triangle (p0,p1,p2)
        {

            sofa::type::Vec<3,Real> n_01 = (p1-p0).cross(v_normal);
            sofa::type::Vec<3,Real> n_12 = (p2-p1).cross(v_normal);
            sofa::type::Vec<3,Real> n_20 = (p0-p2).cross(v_normal);

            return (((ptest-p0)*(n_01) > -ZERO) && ((ptest-p1)*(n_12) > -ZERO) && ((ptest-p2)*(n_20) > -ZERO));

        }
        else // p is not in the plane defined by the triangle (p0,p1,p2)
        {
            return false;
        }

    }
    else // triangle is flat
    {
        return false;
    }
}


/// Test if a point p is in the right halfplane

template<class Real>
bool is_point_in_halfplane(const sofa::type::Vec<3,Real>& p, unsigned int e0, unsigned int e1,
        const sofa::type::Vec<3,Real>& a, const sofa::type::Vec<3,Real>& b, const sofa::type::Vec<3,Real>& c,
        unsigned int ind_p0, unsigned int ind_p1, unsigned int ind_p2)
{
    sofa::type::Vec<3,Real> ptest = p;

    sofa::type::Vec<3,Real> p0 = a;
    sofa::type::Vec<3,Real> p1 = b;
    sofa::type::Vec<3,Real> p2 = c;

    sofa::type::Vec<3,Real> v_normal = (p2-p0).cross(p1-p0);

    Real norm_v_normal = (v_normal)*(v_normal);

    if(norm_v_normal != 0.0)
    {
        if(ind_p0==e0 || ind_p0==e1)
        {
            sofa::type::Vec<3,Real> n_12 = (p2-p1).cross(v_normal);
            return ((ptest-p1)*(n_12) >= 0.0);
        }
        else
        {
            if(ind_p1==e0 || ind_p1==e1)
            {
                sofa::type::Vec<3,Real> n_20 = (p0-p2).cross(v_normal);
                return ((ptest-p2)*(n_20) >= 0.0);
            }
            else
            {
                if(ind_p2==e0 || ind_p2==e1)
                {
                    sofa::type::Vec<3,Real> n_01 = (p1-p0).cross(v_normal);
                    return ((ptest-p0)*(n_01) >= 0.0);
                }
                else
                {
                    return false; // not expected
                }
            }
        }
    }
    else // triangle is flat
    {
        return false;
    }
}


template<class DataTypes>
void TriangleSetGeometryAlgorithms<DataTypes>::initPointAdded(PointID index, const core::topology::PointAncestorElem &ancestorElem
        , const type::vector< VecCoord* >& coordVecs, const type::vector< VecDeriv* >& derivVecs)
{
    using namespace sofa::core::topology;

    if (ancestorElem.type != geometry::ElementType::TRIANGLE)
    {
        EdgeSetGeometryAlgorithms< DataTypes >::initPointAdded(index, ancestorElem, coordVecs, derivVecs);
    }
    else
    {
        const Triangle &t = this->m_topology->getTriangle(ancestorElem.index);

        for (size_t i = 0; i < coordVecs.size(); i++)
        {
            VecCoord &curVecCoord = *coordVecs[i];
            Coord& curCoord = curVecCoord[index];

            const Coord &c0 = curVecCoord[t[0]];
            const Coord &c1 = curVecCoord[t[1]];
            const Coord &c2 = curVecCoord[t[2]];

            // Compute normal (ugly but doesn't require template specialization...)
            type::Vec<3,Real> p0;
            DataTypes::get(p0[0], p0[1], p0[2], c0);
            type::Vec<3,Real> p1;
            DataTypes::get(p1[0], p1[1], p1[2], c1);
            type::Vec<3,Real> p2;
            DataTypes::get(p2[0], p2[1], p2[2], c2);

            type::Vec<3,Real> p0p1 = p1 - p0;
            type::Vec<3,Real> p0p2 = p2 - p0;

            type::Vec<3,Real> n = p0p1.cross(p0p2);
            n.normalize();

            type::Vec<3,Real> newCurCoord = p0 + p0p1 * ancestorElem.localCoords[0] + p0p2 * ancestorElem.localCoords[1] + n * ancestorElem.localCoords[2];
            DataTypes::set(curCoord, newCurCoord[0], newCurCoord[1], newCurCoord[2]);
        }
    }
}



// Move and fix the two closest points of two triangles to their median point
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::Suture2Points(TriangleID ind_ta, TriangleID ind_tb,
    PointID &ind1, PointID &ind2)
{
    // Access the topology
    computeClosestIndexPair(ind_ta, ind_tb, ind1, ind2);

    sofa::type::Vec<3, Real> point_created = computeBaryEdgePoint(ind1, ind2, 0.5);

    sofa::type::vector< Real > x_created;
    x_created.push_back((Real)point_created[0]);
    x_created.push_back((Real)point_created[1]);
    x_created.push_back((Real)point_created[2]);

    auto* state = this->getDOF();

    sofa::helper::WriteAccessor< Data<VecCoord> > x_wA = *state->write(core::vec_id::write_access::position);
    sofa::helper::WriteAccessor< Data<VecDeriv> > v_wA = *state->write(core::vec_id::write_access::velocity);

    DataTypes::set(x_wA[ind1], x_created[0], x_created[1], x_created[2]);
    DataTypes::set(v_wA[ind1], (Real) 0.0, (Real) 0.0, (Real) 0.0);

    DataTypes::set(x_wA[ind2], x_created[0], x_created[1], x_created[2]);
    DataTypes::set(v_wA[ind2], (Real) 0.0, (Real) 0.0, (Real) 0.0);

    return true;
}

// Removes triangles along the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh

template<class DataTypes>
void TriangleSetGeometryAlgorithms< DataTypes >::RemoveAlongTrianglesList(const sofa::type::Vec<3, Real>& a,
    const sofa::type::Vec<3, Real>& b,
    const TriangleID ind_ta,
    const TriangleID ind_tb)
{
    sofa::type::vector< TriangleID > triangles_list;
    sofa::type::vector< EdgeID > edges_list;
    sofa::type::vector< Real > coords_list;

    bool is_intersected = false;

    TriangleID ind_tb_final;

    bool is_on_boundary;

    ind_tb_final = ind_tb;
    TriangleID ind_ta_final = ind_ta;
    is_intersected = computeIntersectedPointsList(sofa::InvalidID, a, b, ind_ta_final, ind_tb_final, triangles_list, edges_list, coords_list, is_on_boundary);

    if (is_intersected)
    {
        m_modifier->removeTriangles(triangles_list, true, true);
    }
}


// Incises along the list of points (ind_edge,coord) intersected by the sequence of input segments (list of input points) and the triangular mesh

template<class DataTypes>
void TriangleSetGeometryAlgorithms< DataTypes >::InciseAlongLinesList(
    const sofa::type::vector< sofa::type::Vec<3, Real> >& input_points,
    const sofa::type::vector< TriangleID > &input_triangles)
{
    // HYP : input_points.size() == input_triangles.size()

    size_t points_size = input_points.size();

    // Initialization for INTERSECTION method
    sofa::type::vector< TriangleID > triangles_list;
    sofa::type::vector< EdgeID > edges_list;
    sofa::type::vector< Real > coords_list;

    TriangleID ind_tb_final;

    bool is_on_boundary;

    const sofa::type::Vec<3, Real> a = input_points[0];
    TriangleID ind_ta = input_triangles[0];

    unsigned int j = 0;
    bool is_validated = true;
    for (j = 0; is_validated && j < points_size - 1; ++j)
    {
        const sofa::type::Vec<3, Real> pa = input_points[j];
        const sofa::type::Vec<3, Real> pb = input_points[j + 1];
        TriangleID ind_tpa = input_triangles[j];
        TriangleID ind_tpb = input_triangles[j + 1];

        bool is_distinct = (pa != pb && ind_tpa != ind_tpb);

        if (is_distinct)
        {
            // Call the method "computeIntersectedPointsList" to get the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh
            ind_tb_final = ind_tpb;
            bool is_intersected = computeIntersectedPointsList(sofa::InvalidID, pa, pb, ind_tpa, ind_tb_final, triangles_list, edges_list, coords_list, is_on_boundary);
            is_validated = is_intersected;
        }
        else
        {
            is_validated = false;
        }
    }

    const sofa::type::Vec<3, Real> b = input_points[j];
    TriangleID ind_tb = input_triangles[j];

    const Triangle &ta = m_container->getTriangle(ind_ta);
    const Triangle &tb = m_container->getTriangle(ind_tb);

    const sofa::Size nb_points = sofa::Size(m_container->getTrianglesAroundVertexArray().size() - 1); 

    const sofa::type::vector<Triangle> &vect_t = m_container->getTriangleArray();
    const sofa::Size nb_triangles = sofa::Size(vect_t.size() - 1);

    // Variables to accumulate the number of elements registered to be created (so as to remember their indices)
    PointID acc_nb_points = (PointID)nb_points;
    TriangleID acc_nb_triangles = (TriangleID)nb_triangles;

    // Variables to accumulate the elements registered to be created or to be removed
    sofa::type::vector< sofa::type::vector< TriangleID > > p_ancestors;
    sofa::type::vector< sofa::type::vector< SReal > > p_baryCoefs;
    sofa::type::vector< Triangle > triangles_to_create;
    sofa::type::vector< TriangleID > trianglesIndexList;
    sofa::type::vector< TriangleID > triangles_to_remove;

    TriangleID ta_to_remove;
    TriangleID tb_to_remove;

    // Initialization for SNAPPING method

    bool is_snap_a0 = false;
    bool is_snap_a1 = false;
    bool is_snap_a2 = false;

    bool is_snap_b0 = false;
    bool is_snap_b1 = false;
    bool is_snap_b2 = false;

    Real epsilon = 0.2; // INFO : epsilon is a threshold in [0,1] to control the snapping of the extremities to the closest vertex

    auto a_baryCoefs =
        computeTriangleBarycoefs(ind_ta, (const sofa::type::Vec<3, Real> &) a);
    snapping_test_triangle(epsilon, a_baryCoefs[0], a_baryCoefs[1], a_baryCoefs[2],
        is_snap_a0, is_snap_a1, is_snap_a2);

    Real is_snapping_a = is_snap_a0 || is_snap_a1 || is_snap_a2;

    auto b_baryCoefs =
        computeTriangleBarycoefs(ind_tb, (const sofa::type::Vec<3, Real> &) b);
    snapping_test_triangle(epsilon, b_baryCoefs[0], b_baryCoefs[1], b_baryCoefs[2],
        is_snap_b0, is_snap_b1, is_snap_b2);

    Real is_snapping_b = is_snap_b0 || is_snap_b1 || is_snap_b2;

    if (is_validated) // intersection successful
    {
        /// force the creation of TrianglesAroundEdgeArray
        m_container->getTrianglesAroundEdgeArray();
        /// force the creation of TrianglesAroundVertexArray
        m_container->getTrianglesAroundVertexArray();

        // Initialization for the indices of the previous intersected edge
        PointID p1_prev = 0;
        PointID p2_prev = 0;

        PointID p1_a = m_container->getEdge(edges_list[0])[0];
        PointID p2_a = m_container->getEdge(edges_list[0])[1];
        PointID p1_b = m_container->getEdge(edges_list[edges_list.size() - 1])[0];
        PointID p2_b = m_container->getEdge(edges_list[edges_list.size() - 1])[1];

        // Plan to remove triangles indexed by ind_ta and ind_tb
        triangles_to_remove.push_back(ind_ta); triangles_to_remove.push_back(ind_tb);

        // Treatment of particular case for first extremity a

        sofa::type::vector< TriangleID > a_first_ancestors;
        sofa::type::vector< SReal > a_first_baryCoefs;

        if (!is_snapping_a)
        {
            /// Register the creation of point a

            a_first_ancestors.push_back(ta[0]);
            a_first_ancestors.push_back(ta[1]);
            a_first_ancestors.push_back(ta[2]);
            p_ancestors.push_back(a_first_ancestors);
            p_baryCoefs.push_back(a_baryCoefs);

            acc_nb_points = acc_nb_points + 1;

            /// Register the creation of triangles incident to point a

            PointID ind_a = (PointID)acc_nb_points; // last point registered to be created

            sofa::type::vector< Triangle > a_triangles;
            Triangle t_a01 = Triangle(ind_a,
                ta[0],
                ta[1]);
            Triangle t_a12 = Triangle(ind_a,
                ta[1],
                ta[2]);
            Triangle t_a20 = Triangle(ind_a,
                ta[2],
                ta[0]);
            triangles_to_create.push_back(t_a01);
            triangles_to_create.push_back(t_a12);
            triangles_to_create.push_back(t_a20);

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles + 1);
            trianglesIndexList.push_back(acc_nb_triangles + 2);
            acc_nb_triangles = acc_nb_triangles + 3;

            /// Register the removal of triangles incident to point a

            if (ta[0] != p1_a && ta[0] != p2_a)
            {
                ta_to_remove = acc_nb_triangles - 1;
            }
            else
            {
                if (ta[1] != p1_a && ta[1] != p2_a)
                {
                    ta_to_remove = acc_nb_triangles;
                }
                else // (ta[2]!=p1_a && ta[2]!=p2_a)
                {
                    ta_to_remove = acc_nb_triangles - 2;
                }
            }
            triangles_to_remove.push_back(ta_to_remove);

            Triangle t_pa1 = Triangle(acc_nb_points + 1,
                ind_a,
                p1_a);
            Triangle t_pa2 = Triangle(acc_nb_points + 2,
                p2_a,
                ind_a);
            triangles_to_create.push_back(t_pa1);
            triangles_to_create.push_back(t_pa2);

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles + 1);
            acc_nb_triangles = acc_nb_triangles + 2;
        }
        else // snapping a to the vertex indexed by ind_a, which is the closest to point a
        {
            // localize the closest vertex
            PointID ind_a;
            PointID p0_a;

            if (ta[0] != p1_a && ta[0] != p2_a)
            {
                p0_a = ta[0];
            }
            else
            {
                if (ta[1] != p1_a && ta[1] != p2_a)
                {
                    p0_a = ta[1];
                }
                else// ta[2]!=p1_a && ta[2]!=p2_a
                {
                    p0_a = ta[2];
                }
            }

            if (is_snap_a0) // is_snap_a1 == false and is_snap_a2 == false
            {
                /// VERTEX 0
                ind_a = ta[0];
            }
            else
            {
                if (is_snap_a1) // is_snap_a0 == false and is_snap_a2 == false
                {
                    /// VERTEX 1
                    ind_a = ta[1];
                }
                else // is_snap_a2 == true and (is_snap_a0 == false and is_snap_a1 == false)
                {
                    /// VERTEX 2
                    ind_a = ta[2];
                }
            }

            /// Register the creation of triangles incident to point indexed by ind_a

            if (ind_a == p1_a)
            {
                Triangle t_pa1 = Triangle(acc_nb_points + 2,
                    p0_a,
                    p1_a);
                Triangle t_pa2 = Triangle(acc_nb_points + 2,
                    p2_a,
                    p0_a);
                triangles_to_create.push_back(t_pa1);
                triangles_to_create.push_back(t_pa2);
            }
            else
            {
                if (ind_a == p2_a)
                {
                    Triangle t_pa1 = Triangle(acc_nb_points + 1,
                        p0_a,
                        p1_a);
                    Triangle t_pa2 = Triangle(acc_nb_points + 1,
                        p2_a,
                        p0_a);
                    triangles_to_create.push_back(t_pa1);
                    triangles_to_create.push_back(t_pa2);
                }
                else
                {
                    Triangle t_pa1 = Triangle(acc_nb_points + 1,
                        ind_a,
                        p1_a);
                    Triangle t_pa2 = Triangle(acc_nb_points + 2,
                        p2_a,
                        ind_a);
                    triangles_to_create.push_back(t_pa1);
                    triangles_to_create.push_back(t_pa2);
                }
            }

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles + 1);
            acc_nb_triangles += 2;
        }

        // Traverse the loop of interected edges

        for (size_t i = 0; i < edges_list.size(); ++i)
        {
            /// Register the creation of the two points (say current "duplicated points") localized on the current interected edge
            PointID p1 = m_container->getEdge(edges_list[i])[0];
            PointID p2 = m_container->getEdge(edges_list[i])[1];

            sofa::type::vector< PointID > p_first_ancestors;
            p_first_ancestors.push_back(p1);
            p_first_ancestors.push_back(p2);
            p_ancestors.push_back(p_first_ancestors);
            p_ancestors.push_back(p_first_ancestors);

            sofa::type::vector< SReal > p_first_baryCoefs;
            p_first_baryCoefs.push_back(1.0 - coords_list[i]);
            p_first_baryCoefs.push_back(coords_list[i]);
            p_baryCoefs.push_back(p_first_baryCoefs);
            p_baryCoefs.push_back(p_first_baryCoefs);

            acc_nb_points = acc_nb_points + 2;

            if (i > 0) // not to treat particular case of first extremitiy
            {
                // SNAPPING TEST

                Real gamma = 0.3;
                bool is_snap_p1;
                bool is_snap_p2;

                snapping_test_edge(gamma, 1.0 - coords_list[i], coords_list[i], is_snap_p1, is_snap_p2);
                Real is_snapping_p = is_snap_p1 || is_snap_p2;

                PointID ind_p;

                if (is_snapping_p && i < edges_list.size() - 1) // not to treat particular case of last extremitiy
                {
                    if (is_snap_p1)
                    {
                        /// VERTEX 0
                        ind_p = p1;
                    }
                    else // is_snap_p2 == true
                    {
                        /// VERTEX 1
                        ind_p = p2;
                    }

                    sofa::type::vector< TriangleID > triangles_list_1;

                    sofa::type::vector< TriangleID > triangles_list_2;

                    prepareVertexDuplication(ind_p, triangles_list[i], triangles_list[i + 1], m_container->getEdge(edges_list[i - 1]), coords_list[i - 1], m_container->getEdge(edges_list[i + 1]), coords_list[i + 1], triangles_list_1, triangles_list_2);
                }

                /// Register the removal of the current triangle

                triangles_to_remove.push_back(triangles_list[i]);

                /// Register the creation of triangles incident to the current "duplicated points" and to the previous "duplicated points"

                PointID p1_created = acc_nb_points - 3;
                PointID p2_created = acc_nb_points - 2;

                PointID p1_to_create = acc_nb_points - 1;
                PointID p2_to_create = acc_nb_points;

                PointID p0_t = m_container->getTriangle(triangles_list[i])[0];
                PointID p1_t = m_container->getTriangle(triangles_list[i])[1];
                PointID p2_t = m_container->getTriangle(triangles_list[i])[2];

                Triangle t_p1 = Triangle(p1_created, p1_prev, p1_to_create);
                Triangle t_p2 = Triangle(p2_created, p2_to_create, p2_prev);

                Triangle t_p3;

                if (p0_t != p1_prev && p0_t != p2_prev)
                {
                    if (p0_t == p1)
                    {
                        t_p3 = Triangle(p0_t, p1_to_create, p1_prev);

                    }
                    else // p0_t==p2
                    {
                        t_p3 = Triangle(p0_t, p2_prev, p2_to_create);
                    }
                }
                else
                {
                    if (p1_t != p1_prev && p1_t != p2_prev)
                    {
                        if (p1_t == p1)
                        {
                            t_p3 = Triangle(p1_t, p1_to_create, p1_prev);
                        }
                        else // p1_t==p2
                        {
                            t_p3 = Triangle(p1_t, p2_prev, p2_to_create);
                        }
                    }
                    else // (p2_t!=p1_prev && p2_t!=p2_prev)
                    {
                        if (p2_t == p1)
                        {
                            t_p3 = Triangle(p2_t, p1_to_create, p1_prev);
                        }
                        else // p2_t==p2
                        {
                            t_p3 = Triangle(p2_t, p2_prev, p2_to_create);
                        }
                    }
                }

                triangles_to_create.push_back(t_p1);
                triangles_to_create.push_back(t_p2);
                triangles_to_create.push_back(t_p3);

                trianglesIndexList.push_back(acc_nb_triangles);
                trianglesIndexList.push_back(acc_nb_triangles + 1);
                trianglesIndexList.push_back(acc_nb_triangles + 2);
                acc_nb_triangles = acc_nb_triangles + 3;
            }

            // Update the previous "duplicated points"
            p1_prev = p1;
            p2_prev = p2;
        }

        // Treatment of particular case for second extremity b
        sofa::type::vector< TriangleID > b_first_ancestors;
        sofa::type::vector< Real > b_first_baryCoefs;

        if (!is_snapping_b)
        {
            /// Register the creation of point b

            b_first_ancestors.push_back(tb[0]);
            b_first_ancestors.push_back(tb[1]);
            b_first_ancestors.push_back(tb[2]);
            p_ancestors.push_back(b_first_ancestors);
            p_baryCoefs.push_back(b_baryCoefs);

            acc_nb_points = acc_nb_points + 1;

            /// Register the creation of triangles incident to point b

            PointID ind_b = acc_nb_points; // last point registered to be created

            sofa::type::vector< Triangle > b_triangles;
            Triangle t_b01 = Triangle(ind_b,
                tb[0],
                tb[1]);
            Triangle t_b12 = Triangle(ind_b,
                tb[1],
                tb[2]);
            Triangle t_b20 = Triangle(ind_b,
                tb[2],
                tb[0]);
            triangles_to_create.push_back(t_b01);
            triangles_to_create.push_back(t_b12);
            triangles_to_create.push_back(t_b20);

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles + 1);
            trianglesIndexList.push_back(acc_nb_triangles + 2);
            acc_nb_triangles = acc_nb_triangles + 3;

            /// Register the removal of triangles incident to point b

            if (tb[0] != p1_b && tb[0] != p2_b)
            {
                tb_to_remove = acc_nb_triangles - 1;
            }
            else
            {
                if (tb[1] != p1_b && tb[1] != p2_b)
                {
                    tb_to_remove = acc_nb_triangles;
                }
                else // (tb[2]!=p1_b && tb[2]!=p2_b)
                {
                    tb_to_remove = acc_nb_triangles - 2;
                }
            }
            triangles_to_remove.push_back(tb_to_remove);

            Triangle t_pb1 = Triangle(acc_nb_points - 2,
                p1_b,
                ind_b);
            Triangle t_pb2 = Triangle(acc_nb_points - 1,
                ind_b,
                p2_b);
            triangles_to_create.push_back(t_pb1);
            triangles_to_create.push_back(t_pb2);

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles + 1);
            acc_nb_triangles = acc_nb_triangles + 2;

        }
        else // snapping b to the vertex indexed by ind_b, which is the closest to point b
        {
            // localize the closest vertex
            PointID ind_b;
            PointID p0_b;

            if (tb[0] != p1_b && tb[0] != p2_b)
            {
                p0_b = tb[0];
            }
            else
            {
                if (tb[1] != p1_b && tb[1] != p2_b)
                {
                    p0_b = tb[1];
                }
                else// tb[2]!=p1_b && tb[2]!=p2_b
                {
                    p0_b = tb[2];
                }
            }

            if (is_snap_b0) // is_snap_b1 == false and is_snap_b2 == false
            {
                /// VERTEX 0
                ind_b = tb[0];
            }
            else
            {
                if (is_snap_b1) // is_snap_b0 == false and is_snap_b2 == false
                {
                    /// VERTEX 1
                    ind_b = tb[1];
                }
                else // is_snap_b2 == true and (is_snap_b0 == false and is_snap_b1 == false)
                {
                    /// VERTEX 2
                    ind_b = tb[2];
                }
            }

            /// Register the creation of triangles incident to point indexed by ind_b

            if (ind_b == p1_b)
            {
                Triangle t_pb1 = Triangle(acc_nb_points, p1_b, p0_b);
                Triangle t_pb2 = Triangle(acc_nb_points, p0_b, p2_b);
                triangles_to_create.push_back(t_pb1);
                triangles_to_create.push_back(t_pb2);

            }
            else
            {
                if (ind_b == p2_b)
                {
                    Triangle t_pb1 = Triangle(acc_nb_points - 1, p1_b, p0_b);
                    Triangle t_pb2 = Triangle(acc_nb_points - 1, p0_b, p2_b);
                    triangles_to_create.push_back(t_pb1);
                    triangles_to_create.push_back(t_pb2);
                }
                else
                {
                    Triangle t_pb1 = Triangle(acc_nb_points - 1, p1_b, ind_b);
                    Triangle t_pb2 = Triangle(acc_nb_points, ind_b, p2_b);
                    triangles_to_create.push_back(t_pb1);
                    triangles_to_create.push_back(t_pb2);
                }
            }

            trianglesIndexList.push_back(acc_nb_triangles);
            trianglesIndexList.push_back(acc_nb_triangles + 1);
            acc_nb_triangles += 2;
        }

        // Create all the points registered to be created
        m_modifier->addPointsProcess(acc_nb_points - nb_points);

        // Warn for the creation of all the points registered to be created
        m_modifier->addPointsWarning(acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

        // Create all the triangles registered to be created
        m_modifier->addTrianglesProcess((const sofa::type::vector< Triangle > &) triangles_to_create); // WARNING called after the creation process by the method "addTrianglesProcess"

        // Warn for the creation of all the triangles registered to be created
        m_modifier->addTrianglesWarning(sofa::Size(triangles_to_create.size()), triangles_to_create, trianglesIndexList);

        // Propagate the topological changes *** not necessary
        //m_modifier->propagateTopologicalChanges();

        // Remove all the triangles registered to be removed
        m_modifier->removeTriangles(triangles_to_remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

        // Propagate the topological changes *** not necessary
        //m_modifier->propagateTopologicalChanges();
    }
}



template<class DataTypes>
int TriangleSetGeometryAlgorithms<DataTypes>::SplitAlongPath(PointID ind_A, Coord& pointA, PointID ind_B, Coord& pointB,
    sofa::type::vector< sofa::geometry::ElementType >& intersected_topoElements,
    sofa::type::vector< ElemID >& intersected_indices,
    sofa::type::vector< Vec3 >& intersected_barycoefs,
    sofa::type::vector< EdgeID >& new_edges, Real epsilonSnapPath, Real epsilonSnapBorder)
{
    //////// STEP 1.a : MODIFY PATH IF SNAP = TRUE (don't change border case here if they are near an edge)
    if (intersected_indices.empty()) return 0;

    sofa::type::vector< sofa::type::vector<Real> > points2Snap;

    //	Real epsilon = 0.25; // to change to an input for snapping

    if (epsilonSnapPath != 0.0)
        SnapAlongPath(intersected_topoElements, intersected_indices, intersected_barycoefs, points2Snap, epsilonSnapPath);

    //STEP 1.b : Modify border case path if snap = true
    if (epsilonSnapBorder != 0.0)
        SnapBorderPath(ind_A, pointA, ind_B, pointB, intersected_topoElements, intersected_indices, intersected_barycoefs, points2Snap, epsilonSnapBorder);

    // Output declarations:
    const size_t nb_points = intersected_indices.size();
    sofa::type::vector< sofa::type::vector< PointID > > p_ancestors; 
    p_ancestors.reserve(nb_points);// WARNING
    sofa::type::vector< sofa::type::vector< SReal > > p_baryCoefs; 
    p_baryCoefs.reserve(nb_points);
    PointID next_point = m_container->getNbPoints();
    TriangleID next_triangle = (TriangleID)m_container->getNbTriangles();
    sofa::type::vector< PointID > new_edge_points; // new points created on each edge
    sofa::type::vector< Triangle > new_triangles;
    sofa::type::vector< TriangleID > new_triangles_id;
    sofa::type::vector< TriangleID > removed_triangles;
    sofa::type::vector< sofa::type::vector< TriangleID > >  triangles_ancestors;
    sofa::type::vector< sofa::type::vector< SReal > >  triangles_barycoefs;


    type::vector< core::topology::PointAncestorElem > srcElems;

    //////// STEP 1 : Create points

    for (size_t i = 0; i < nb_points; i++)
    {

        p_ancestors.resize(p_ancestors.size() + 1);
        sofa::type::vector< PointID >& ancestors = p_ancestors.back();
        p_baryCoefs.resize(p_baryCoefs.size() + 1);
        auto& baryCoefs = p_baryCoefs.back();


        switch (intersected_topoElements[i])
        {

        case geometry::ElementType::POINT:
        {
            // qlq chose a faire?
            new_edge_points.push_back(intersected_indices[i]);

            p_ancestors.resize(p_ancestors.size() - 1);
            p_baryCoefs.resize(p_baryCoefs.size() - 1);

            // For snapping:
            if ((epsilonSnapPath != 0.0) || (!points2Snap.empty()))
                for (size_t j = 0; j < points2Snap.size(); j++)
                    if (points2Snap[j][0] == intersected_indices[i])
                    {
                        if (i == 0 || i == nb_points - 1) //should not append, 0 and nb_points-1 correspond to bordersnap
                        {
                            PointID the_point = intersected_indices[i];
                            const sofa::type::vector<EdgeID>& shell = m_container->getEdgesAroundVertex(the_point);
                            unsigned int cptSnap = 0;

                            for (size_t k = 0; k < shell.size(); k++)
                            {
                                const Edge& the_edge = m_container->getEdge(shell[k]);
                                if (the_edge[0] == the_point)
                                    points2Snap[j].push_back(the_edge[1]);
                                else
                                    points2Snap[j].push_back(the_edge[0]);

                                cptSnap++;
                                if (cptSnap == 3)
                                    break;
                            }

                            if (cptSnap != 3)
                                msg_error() << "Error: In snapping border, missing elements to compute barycoefs!";

                            break;
                        }

                        points2Snap[j].push_back(next_point - 1);
                        points2Snap[j].push_back(next_point);

                        if (intersected_topoElements[i - 1] == sofa::geometry::ElementType::POINT) //second dof has to be moved, first acestor must be pa
                            points2Snap[j][4] = intersected_indices[i - 1];

                        if (intersected_topoElements[i + 1] == sofa::geometry::ElementType::POINT) //second dof has to be moved, first acestor must be pa
                            points2Snap[j][5] = intersected_indices[i + 1];

                        break;
                    }

            break;
        }

        case geometry::ElementType::EDGE:
        {
            Edge theEdge = m_container->getEdge(intersected_indices[i]);
            ancestors.push_back(theEdge[0]);
            ancestors.push_back(theEdge[1]);

            baryCoefs.push_back(1.0 - intersected_barycoefs[i][0]);
            baryCoefs.push_back(intersected_barycoefs[i][0]);

            srcElems.push_back(core::topology::PointAncestorElem(sofa::geometry::ElementType::EDGE, intersected_indices[i],
                core::topology::PointAncestorElem::LocalCoords(intersected_barycoefs[i][0], 0, 0)));

            new_edge_points.push_back(next_point);
            ++next_point;
            break;
        }
        case geometry::ElementType::TRIANGLE:
        {

            Triangle theTriangle = m_container->getTriangle(intersected_indices[i]);

            ancestors.push_back(theTriangle[0]);
            ancestors.push_back(theTriangle[1]);
            ancestors.push_back(theTriangle[2]);

            baryCoefs.push_back(intersected_barycoefs[i][0]);
            baryCoefs.push_back(intersected_barycoefs[i][1]);
            baryCoefs.push_back(intersected_barycoefs[i][2]);

            srcElems.push_back(core::topology::PointAncestorElem(sofa::geometry::ElementType::TRIANGLE, intersected_indices[i],
                core::topology::PointAncestorElem::LocalCoords(intersected_barycoefs[i][1], intersected_barycoefs[i][2], 0)));

            new_edge_points.push_back(next_point);// hum...? pour les edges to split
            ++next_point;
            break;
        }
        default:
            break;

        }
    }

    bool error = false;

    // STEP 2: Computing triangles along path

    for (size_t i = 0; i < intersected_indices.size() - 1; ++i)
    {
        ElemID firstObject = intersected_indices[i];

        switch (intersected_topoElements[i])
        {
        case geometry::ElementType::POINT:
        {
            PointID thePointFirst = firstObject;

            switch (intersected_topoElements[i + 1])
            {
            case geometry::ElementType::POINT: // Triangle to create: 0 / Triangle to remove: 0
            {
                PointID thePointSecond = intersected_indices[i + 1];
                sofa::type::vector<EdgeID> edgevertexshell = m_container->getEdgesAroundVertex(thePointSecond);
                bool test = false;

                for (size_t j = 0; j < edgevertexshell.size(); j++)
                {
                    Edge e = m_container->getEdge(edgevertexshell[j]);

                    if (((e[0] == thePointSecond) && (e[1] == thePointFirst)) || ((e[1] == thePointSecond) && (e[0] == thePointFirst)))
                    {
                        test = true;
                        break;
                    }
                }
                if (!test)
                {
                    msg_error() << "SplitAlongPath: POINT::EDGE case, the edge between these points has not been found.";
                    error = true;
                }

                break;
            }
            case geometry::ElementType::EDGE: // Triangle to create: 2 / Triangle to remove: 1
            {
                EdgeID edgeIDSecond = intersected_indices[i + 1];
                TriangleID triId;
                Triangle tri;

                sofa::type::vector<TriangleID> triangleedgeshell = m_container->getTrianglesAroundEdge(edgeIDSecond);

                for (size_t j = 0; j < triangleedgeshell.size(); j++)
                {
                    triId = triangleedgeshell[j];
                    tri = m_container->getTriangle(triangleedgeshell[j]);

                    if ((tri[0] == thePointFirst) || (tri[1] == thePointFirst) || (tri[2] == thePointFirst))
                    {
                        triangles_ancestors.resize(triangles_ancestors.size() + 2);
                        triangles_barycoefs.resize(triangles_barycoefs.size() + 2);

                        triangles_ancestors[triangles_ancestors.size() - 2].push_back(triId);
                        triangles_barycoefs[triangles_barycoefs.size() - 2].push_back(1.0);
                        triangles_ancestors[triangles_ancestors.size() - 1].push_back(triId);
                        triangles_barycoefs[triangles_barycoefs.size() - 1].push_back(1.0);
                        //found = true;

                        break;
                    }
                }

                int vertxInTriangle = m_container->getVertexIndexInTriangle(tri, thePointFirst);

                if (vertxInTriangle == -1)
                {
                    msg_error() << "SplitAlongPath: vertxInTriangle not found in POINT::EDGE case";
                    error = true;

                    break;
                }


                new_triangles.push_back(Triangle(tri[vertxInTriangle], new_edge_points[i + 1], tri[(vertxInTriangle + 2) % 3]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back(Triangle(tri[vertxInTriangle], tri[(vertxInTriangle + 1) % 3], new_edge_points[i + 1]));
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triId);

                break;
            }
            case geometry::ElementType::TRIANGLE: // Triangle to create: 3 / Triangle to remove: 1
            {
                TriangleID triangleIDSecond = intersected_indices[i + 1];
                Triangle theTriangleSecond = m_container->getTriangle(triangleIDSecond);

                triangles_ancestors.resize(triangles_ancestors.size() + 3);
                triangles_barycoefs.resize(triangles_barycoefs.size() + 3);

                for (unsigned int j = 0; j < 3; j++)
                {
                    triangles_ancestors[triangles_ancestors.size() - j - 1].push_back(triangleIDSecond);
                    triangles_barycoefs[triangles_barycoefs.size() - j - 1].push_back(1.0);
                }

                new_triangles.push_back(Triangle(theTriangleSecond[0], theTriangleSecond[1], new_edge_points[i + 1]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back(Triangle(theTriangleSecond[1], theTriangleSecond[2], new_edge_points[i + 1]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back(Triangle(theTriangleSecond[2], theTriangleSecond[0], new_edge_points[i + 1]));
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triangleIDSecond);

                break;
            }
            default:
                break;

            }
            break;
        }

        case geometry::ElementType::EDGE:
        {
            PointID p1 = new_edge_points[i];
            EdgeID edgeIDFirst = firstObject;
            Edge theEdgeFirst = m_container->getEdge(firstObject);
            sofa::type::Vec<3, Real> pos1 = computeBaryEdgePoint(theEdgeFirst, intersected_barycoefs[i][0]);

            switch (intersected_topoElements[i + 1])
            {

            case geometry::ElementType::POINT: // Triangle to create: 2 / Triangle to remove: 1
            {
                PointID thePointSecond = intersected_indices[i + 1];

                TriangleID triId;
                Triangle tri;

                sofa::type::vector<TriangleID> triangleedgeshell = m_container->getTrianglesAroundEdge(edgeIDFirst);

                for (size_t j = 0; j < triangleedgeshell.size(); j++)
                {
                    triId = triangleedgeshell[j];
                    tri = m_container->getTriangle(triangleedgeshell[j]);

                    if ((tri[0] == thePointSecond) || (tri[1] == thePointSecond) || (tri[2] == thePointSecond))
                    {
                        triangles_ancestors.resize(triangles_ancestors.size() + 2);
                        triangles_barycoefs.resize(triangles_barycoefs.size() + 2);

                        triangles_ancestors[triangles_ancestors.size() - 2].push_back(triId);
                        triangles_barycoefs[triangles_barycoefs.size() - 2].push_back(1.0);
                        triangles_ancestors[triangles_ancestors.size() - 1].push_back(triId);
                        triangles_barycoefs[triangles_barycoefs.size() - 1].push_back(1.0);

                        break;
                    }
                }

                int vertxInTriangle = m_container->getVertexIndexInTriangle(tri, thePointSecond);

                if (vertxInTriangle == -1)
                {
                    msg_error() << " Error: SplitAlongPath: vertxInTriangle not found in EDGE::POINT case";
                    error = true;
                    break;
                }

                new_triangles.push_back(Triangle(thePointSecond, p1, tri[(vertxInTriangle + 2) % 3]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back(Triangle(thePointSecond, tri[(vertxInTriangle + 1) % 3], p1));
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triId);

                break;
            }
            case geometry::ElementType::EDGE: // Triangle to create: 3 / Triangle to remove: 1
            {
                PointID p2 = new_edge_points[i + 1];
                EdgeID edgeIDSecond = intersected_indices[i + 1];
                Edge theEdgeSecond = m_container->getEdge(edgeIDSecond);
                sofa::type::Vec<3, Real> pos2 = computeBaryEdgePoint(theEdgeSecond, intersected_barycoefs[i + 1][0]);

                TriangleID triId;
                Triangle tri;

                sofa::type::vector<TriangleID> triangleedgeshell = m_container->getTrianglesAroundEdge(edgeIDFirst);

                for (size_t j = 0; j < triangleedgeshell.size(); j++)
                {
                    triId = triangleedgeshell[j];
                    tri = m_container->getTriangle(triangleedgeshell[j]);
                    const EdgesInTriangle triedge = m_container->getEdgesInTriangle(triangleedgeshell[j]);

                    if ((triedge[0] == edgeIDSecond) || (triedge[1] == edgeIDSecond) || (triedge[2] == edgeIDSecond))
                    {
                        triangles_ancestors.resize(triangles_ancestors.size() + 3);
                        triangles_barycoefs.resize(triangles_barycoefs.size() + 3);

                        for (unsigned int k = 0; k < 3; k++)
                        {
                            triangles_ancestors[triangles_ancestors.size() - k - 1].push_back(triId);
                            triangles_barycoefs[triangles_barycoefs.size() - k - 1].push_back(1.0);
                        }
                        break;
                    }
                }


                // Find common corner and find incision direction in triangle
                PointID cornerInEdge1 = ((theEdgeFirst[0] == theEdgeSecond[0]) || (theEdgeFirst[0] == theEdgeSecond[1])) ? 0 : 1;
                int vertxInTriangle = m_container->getVertexIndexInTriangle(tri, theEdgeFirst[cornerInEdge1]);

                PointID vertexOrder[5]; //corner, p1, tri+1, tri+2, p2
                Coord posOrder[4];

                vertexOrder[0] = theEdgeFirst[cornerInEdge1]; 
                if (tri[(vertxInTriangle + 1) % 3] == theEdgeFirst[(cornerInEdge1 + 1) % 2])
                {
                    posOrder[0] = pos1; vertexOrder[1] = p1;
                    posOrder[3] = pos2; vertexOrder[4] = p2;
                }
                else
                {
                    posOrder[0] = pos2; vertexOrder[1] = p2;
                    posOrder[3] = pos1; vertexOrder[4] = p1;
                }

                posOrder[1] = this->getPointPosition(tri[(vertxInTriangle + 1) % 3]);
                posOrder[2] = this->getPointPosition(tri[(vertxInTriangle + 2) % 3]);
                vertexOrder[2] = tri[(vertxInTriangle + 1) % 3];
                vertexOrder[3] = tri[(vertxInTriangle + 2) % 3];


                // Create the triangle around corner
                new_triangles.emplace_back(vertexOrder[0], vertexOrder[1], vertexOrder[4]);
                new_triangles_id.push_back(next_triangle++);


                // Triangularize the remaining quad according to the delaunay criteria
                if (isQuadDeulaunayOriented(posOrder[0], posOrder[1], posOrder[2], posOrder[3]))
                {
                    new_triangles.emplace_back(vertexOrder[1], vertexOrder[2], vertexOrder[4]);
                    new_triangles_id.push_back(next_triangle++);
                    new_triangles.emplace_back(vertexOrder[2], vertexOrder[3], vertexOrder[4]);
                    new_triangles_id.push_back(next_triangle++);
                }
                else
                {
                    new_triangles.emplace_back(vertexOrder[1], vertexOrder[2], vertexOrder[3]);
                    new_triangles_id.push_back(next_triangle++);
                    new_triangles.emplace_back(vertexOrder[4], vertexOrder[1], vertexOrder[3]);
                    new_triangles_id.push_back(next_triangle++);
                }

                removed_triangles.push_back(triId);
                break;
            }
            case geometry::ElementType::TRIANGLE: // Triangle to create: 4 / Triangle to remove: 1
            {
                PointID p2 = new_edge_points[i + 1];
                TriangleID triangleIDSecond = intersected_indices[i + 1];
                Triangle theTriangleSecond = m_container->getTriangle(triangleIDSecond);

                const EdgesInTriangle triedge = m_container->getEdgesInTriangle(triangleIDSecond);
                int edgeInTriangle = m_container->getEdgeIndexInTriangle(triedge, edgeIDFirst);

                if (edgeInTriangle == -1)
                {
                    msg_error() << " Error: SplitAlongPath: edgeInTriangle not found in EDGE::TRIANGLE case";
                    error = true;
                    break;
                }

                triangles_ancestors.resize(triangles_ancestors.size() + 4);
                triangles_barycoefs.resize(triangles_barycoefs.size() + 4);

                for (unsigned int j = 0; j < 4; j++)
                {
                    triangles_ancestors[triangles_ancestors.size() - j - 1].push_back(triangleIDSecond);
                    triangles_barycoefs[triangles_barycoefs.size() - j - 1].push_back(1.0);
                }


                // create two triangles linking p with the corner
                new_triangles.emplace_back(p2, theTriangleSecond[edgeInTriangle], theTriangleSecond[(edgeInTriangle + 1) % 3]);
                new_triangles_id.push_back(next_triangle++);
                new_triangles.emplace_back(p2, theTriangleSecond[(edgeInTriangle + 2) % 3], theTriangleSecond[edgeInTriangle]);
                new_triangles_id.push_back(next_triangle++);


                // create two triangles linking p with the split edge
                new_triangles.emplace_back(p2, theTriangleSecond[(edgeInTriangle + 1) % 3], p1);
                new_triangles_id.push_back(next_triangle++);
                new_triangles.emplace_back(p2, p1, theTriangleSecond[(edgeInTriangle + 2) % 3]);
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triangleIDSecond);
                break;

            }
            default:
                break;

            }
            break;
        }
        case geometry::ElementType::TRIANGLE:
        {
            Triangle theTriangleFirst = m_container->getTriangle(firstObject);
            TriangleID triangleIDFirst = intersected_indices[i];
            PointID p1 = new_edge_points[i];
            PointID p2 = new_edge_points[i + 1];

            switch (intersected_topoElements[i + 1])
            {
            case geometry::ElementType::POINT: // Triangle to create: 3 / Triangle to remove: 1
            {
                triangles_ancestors.resize(triangles_ancestors.size() + 3);
                triangles_barycoefs.resize(triangles_barycoefs.size() + 3);

                for (unsigned int j = 0; j < 3; j++)
                {
                    triangles_ancestors[triangles_ancestors.size() - j - 1].push_back(triangleIDFirst);
                    triangles_barycoefs[triangles_barycoefs.size() - j - 1].push_back(1.0);
                }

                new_triangles.push_back(Triangle(p1, theTriangleFirst[0], theTriangleFirst[1]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back(Triangle(p1, theTriangleFirst[1], theTriangleFirst[2]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back(Triangle(p1, theTriangleFirst[2], theTriangleFirst[0]));
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triangleIDFirst);

                break;
            }
            case geometry::ElementType::EDGE: // Triangle to create: 4 / Triangle to remove: 1
            {
                EdgeID edgeIDSecond = intersected_indices[i + 1];

                const EdgesInTriangle triedge = m_container->getEdgesInTriangle(triangleIDFirst);
                int edgeInTriangle = m_container->getEdgeIndexInTriangle(triedge, edgeIDSecond);

                if (edgeInTriangle == -1)
                {
                    msg_error() << " Error: SplitAlongPath: edgeInTriangle not found in TRIANGLE::EDGE case";
                    error = true;
                    break;
                }

                triangles_ancestors.resize(triangles_ancestors.size() + 4);
                triangles_barycoefs.resize(triangles_barycoefs.size() + 4);

                for (unsigned int j = 0; j < 4; j++)
                {
                    triangles_ancestors[triangles_ancestors.size() - j - 1].push_back(triangleIDFirst);
                    triangles_barycoefs[triangles_barycoefs.size() - j - 1].push_back(1.0);
                }

                // create two triangles linking p with the corner
                new_triangles.push_back(Triangle(p1, theTriangleFirst[edgeInTriangle], theTriangleFirst[(edgeInTriangle + 1) % 3]));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back(Triangle(p1, theTriangleFirst[(edgeInTriangle + 2) % 3], theTriangleFirst[edgeInTriangle]));
                new_triangles_id.push_back(next_triangle++);


                // create two triangles linking p with the split edge
                new_triangles.push_back(Triangle(p1, theTriangleFirst[(edgeInTriangle + 1) % 3], p2));
                new_triangles_id.push_back(next_triangle++);
                new_triangles.push_back(Triangle(p1, p2, theTriangleFirst[(edgeInTriangle + 2) % 3]));
                new_triangles_id.push_back(next_triangle++);

                removed_triangles.push_back(triangleIDFirst);
                break;
            }
            case geometry::ElementType::TRIANGLE: // Triangle to create: 5 / Triangle to remove: 1
            {
                TriangleID triangleIDSecond = intersected_indices[i + 1];

                if (triangleIDSecond != triangleIDFirst)
                {
                    msg_error() << "SplitAlongPath: incision not in the mesh plan not supported yet, in TRIANGLE::TRIANGLE case";
                    error = true;
                    break;
                }

                PointID quad[2][4];
                PointID Tri1[3];
                Real tmp1 = 0.0; unsigned int cornerP1[3] = { 0,0,0 };
                Real tmp2 = 0.0; unsigned int cornerP2[3] = { 0,0,0 };

                for (unsigned int j = 0; j < 3; j++) // find first corners
                {
                    if (intersected_barycoefs[i][j] > tmp1)
                    {
                        tmp1 = intersected_barycoefs[i][j];
                        cornerP1[0] = j;
                    }

                    if (intersected_barycoefs[i + 1][j] > tmp2)
                    {
                        tmp2 = intersected_barycoefs[i + 1][j];
                        cornerP2[0] = j;
                    }
                }

                // sort other corners by decreasing barycoef
                if (intersected_barycoefs[i][(cornerP1[0] + 1) % 3] > intersected_barycoefs[i][(cornerP1[0] + 2) % 3])
                {
                    cornerP1[1] = (cornerP1[0] + 1) % 3;
                    cornerP1[2] = (cornerP1[0] + 2) % 3;
                }
                else
                {
                    cornerP1[1] = (cornerP1[0] + 2) % 3;
                    cornerP1[2] = (cornerP1[0] + 1) % 3;
                }

                if (intersected_barycoefs[i + 1][(cornerP2[0] + 1) % 3] > intersected_barycoefs[i + 1][(cornerP2[0] + 2) % 3])
                {
                    cornerP2[1] = (cornerP2[0] + 1) % 3;
                    cornerP2[2] = (cornerP2[0] + 2) % 3;
                }
                else
                {
                    cornerP2[1] = (cornerP2[0] + 2) % 3;
                    cornerP2[2] = (cornerP2[0] + 1) % 3;
                }


                if (cornerP1[0] != cornerP2[0])
                {
                    unsigned int cornerP1InTriangle = cornerP1[0];
                    unsigned int cornerP2InTriangle = cornerP2[0];

                    if ((cornerP1InTriangle + 1) % 3 == cornerP2InTriangle) // in the right direction
                    {
                        quad[0][0] = p1; quad[0][1] = theTriangleFirst[cornerP1InTriangle];
                        quad[0][3] = p2; quad[0][2] = theTriangleFirst[cornerP2InTriangle];

                        if (intersected_barycoefs[i][(cornerP1InTriangle + 2) % 3] > intersected_barycoefs[i + 1][(cornerP1InTriangle + 2) % 3]) // second quad in other direction
                        {
                            quad[1][0] = p2; quad[1][1] = theTriangleFirst[(cornerP1InTriangle + 1) % 3];
                            quad[1][3] = p1; quad[1][2] = theTriangleFirst[(cornerP1InTriangle + 2) % 3];
                            Tri1[0] = p1; Tri1[1] = theTriangleFirst[(cornerP1InTriangle + 2) % 3]; Tri1[2] = theTriangleFirst[cornerP1InTriangle];
                        }
                        else
                        {
                            quad[1][0] = p2; quad[1][1] = theTriangleFirst[(cornerP1InTriangle + 2) % 3];
                            quad[1][3] = p1; quad[1][2] = theTriangleFirst[cornerP1InTriangle];
                            Tri1[0] = p2; Tri1[1] = theTriangleFirst[(cornerP1InTriangle + 1) % 3]; Tri1[2] = theTriangleFirst[(cornerP1InTriangle + 2) % 3];
                        }
                    }
                    else     // switch order due to incision direction
                    {
                        quad[0][0] = p2; quad[0][1] = theTriangleFirst[cornerP2InTriangle];
                        quad[0][3] = p1; quad[0][2] = theTriangleFirst[cornerP1InTriangle];

                        if (intersected_barycoefs[i][(cornerP2InTriangle + 2) % 3] > intersected_barycoefs[i + 1][(cornerP2InTriangle + 2) % 3]) // second quad in other direction
                        {
                            quad[1][0] = p1; quad[1][1] = theTriangleFirst[(cornerP1InTriangle + 1) % 3];
                            quad[1][3] = p2; quad[1][2] = theTriangleFirst[(cornerP1InTriangle + 2) % 3];
                            Tri1[0] = p1; Tri1[1] = theTriangleFirst[cornerP1InTriangle]; Tri1[2] = theTriangleFirst[(cornerP1InTriangle + 1) % 3];
                        }
                        else
                        {
                            quad[1][0] = p1; quad[1][1] = theTriangleFirst[cornerP1InTriangle];
                            quad[1][3] = p2; quad[1][2] = theTriangleFirst[(cornerP1InTriangle + 1) % 3];
                            Tri1[0] = p2; Tri1[1] = theTriangleFirst[(cornerP1InTriangle + 1) % 3]; Tri1[2] = theTriangleFirst[(cornerP1InTriangle + 2) % 3];
                        }
                    }
                }
                else
                {
                    unsigned int closest, second;
                    int cornerInTriangle;

                    if (tmp1 > tmp2)
                    {
                        closest = p1; second = p2;
                        cornerInTriangle = cornerP1[0];
                    }
                    else
                    {
                        closest = p2; second = p1;
                        cornerInTriangle = cornerP2[0];
                    }

                    quad[0][0] = closest; quad[0][1] = theTriangleFirst[cornerInTriangle];
                    quad[0][3] = second; quad[0][2] = theTriangleFirst[(cornerInTriangle + 1) % 3];

                    quad[1][0] = second; quad[1][1] = theTriangleFirst[(cornerInTriangle + 2) % 3];
                    quad[1][3] = closest; quad[1][2] = theTriangleFirst[cornerInTriangle];

                    Tri1[0] = second; Tri1[1] = theTriangleFirst[(cornerInTriangle + 1) % 3]; Tri1[2] = theTriangleFirst[(cornerInTriangle + 2) % 3];
                }

                new_triangles.push_back(Triangle(Tri1[0], Tri1[1], Tri1[2]));
                new_triangles_id.push_back(next_triangle++);

                // Triangularize the remaining quad according to the delaunay criteria
                const typename DataTypes::VecCoord& coords = (this->getDOF()->read(core::vec_id::read_access::position)->getValue());
                for (unsigned int j = 0; j < 2; j++)
                {
                    //Vec<3,Real> pos[4];
                    Coord pos[4];
                    for (unsigned int k = 0; k < 4; k++)
                    {
                        if (quad[j][k] == p1)
                            for (unsigned int u = 0; u < 3; u++)
                                for (unsigned int v = 0; v < 3; v++)
                                    pos[k][v] = pos[k][v] + coords[theTriangleFirst[u]][v] * (Real)intersected_barycoefs[i][u];
                        else if (quad[j][k] == p2)
                            for (unsigned int u = 0; u < 3; u++)
                                for (unsigned int v = 0; v < 3; v++)
                                    pos[k][v] = pos[k][v] + coords[theTriangleFirst[u]][v] * (Real)intersected_barycoefs[i + 1][u];
                        else
                            pos[k] = coords[quad[j][k]];

                    }

                    if (isQuadDeulaunayOriented(pos[0], pos[1], pos[2], pos[3]))
                    {
                        new_triangles.push_back(Triangle(quad[j][1], quad[j][2], quad[j][0]));
                        new_triangles_id.push_back(next_triangle++);
                        new_triangles.push_back(Triangle(quad[j][3], quad[j][0], quad[j][2]));
                        new_triangles_id.push_back(next_triangle++);
                    }
                    else
                    {
                        new_triangles.push_back(Triangle(quad[j][2], quad[j][3], quad[j][1]));
                        new_triangles_id.push_back(next_triangle++);
                        new_triangles.push_back(Triangle(quad[j][0], quad[j][1], quad[j][3]));
                        new_triangles_id.push_back(next_triangle++);
                    }

                }

                triangles_ancestors.resize(triangles_ancestors.size() + 5);
                triangles_barycoefs.resize(triangles_barycoefs.size() + 5);

                for (unsigned int j = 0; j < 5; j++)
                {
                    triangles_ancestors[triangles_ancestors.size() - j - 1].push_back(triangleIDFirst);
                    triangles_barycoefs[triangles_barycoefs.size() - j - 1].push_back(1.0);
                }

                removed_triangles.push_back(triangleIDFirst);

                break;
            }
            default:
                break;
            }
            break;
        }

        default:
            break;
        }

        if (error)
        {
            msg_error() << "ERROR: in the incision path. ";
            return -1;
        }
    }

    // FINAL STEP : Apply changes
    PointID newP0 = next_point - (PointID)srcElems.size();
    m_modifier->addPoints(sofa::Size(srcElems.size()), srcElems);

    // m_modifier->propagateTopologicalChanges();

    // Create new edges with full ancestry information
    std::set<Edge> edges_processed;
    sofa::type::vector<Edge> edges_added;
    sofa::type::vector<core::topology::EdgeAncestorElem> edges_src;
    for (size_t ti = 0; ti < new_triangles.size(); ++ti)
    {
        Triangle t = new_triangles[ti];
        for (int tpi = 0; tpi < 3; ++tpi)
        {
            Edge e(t[tpi], t[(tpi + 1) % 3]);
            if (e[0] > e[1]) { PointID tmp = e[0]; e[0] = e[1]; e[1] = tmp; }
            if (e[0] < newP0 && e[1] < newP0 && m_container->getEdgeIndex(e[0], e[1]) != sofa::InvalidID)
                continue; // existing edge
            if (!edges_processed.insert(e).second)
                continue; // this edge was already processed
            core::topology::EdgeAncestorElem src;
            for (unsigned int k = 0; k < 2; ++k)
            {
                if (e[k] < newP0)
                { // previous point
                    src.pointSrcElems[k].type = geometry::ElementType::POINT;
                    src.pointSrcElems[k].index = e[k];
                }
                else
                {
                    src.pointSrcElems[k] = srcElems[e[k] - newP0];
                }
            }
            // Source element could be an edge if both points are from it or from its endpoints
            if (src.pointSrcElems[0].type != geometry::ElementType::TRIANGLE
                && src.pointSrcElems[1].type != geometry::ElementType::TRIANGLE
                && (src.pointSrcElems[0].type == geometry::ElementType::EDGE
                    || src.pointSrcElems[1].type == geometry::ElementType::EDGE)
                && (src.pointSrcElems[0].type == geometry::ElementType::POINT
                    || src.pointSrcElems[1].type == geometry::ElementType::POINT
                    || src.pointSrcElems[0].index == src.pointSrcElems[1].index))
            {
                unsigned int src_eid = (src.pointSrcElems[0].type == geometry::ElementType::EDGE)
                    ? src.pointSrcElems[0].index : src.pointSrcElems[1].index;
                Edge src_e = m_container->getEdge(src_eid);
                if ((src.pointSrcElems[0].type != geometry::ElementType::POINT
                    || src.pointSrcElems[0].index == src_e[0]
                    || src.pointSrcElems[0].index == src_e[1])
                    && (src.pointSrcElems[1].type != geometry::ElementType::POINT
                        || src.pointSrcElems[1].index == src_e[0]
                        || src.pointSrcElems[1].index == src_e[1]))
                {
                    src.srcElems.push_back(core::topology::TopologyElemID(geometry::ElementType::EDGE,
                        src_eid));
                }
            }
            if (src.srcElems.empty()) // within the initial triangle by default
                src.srcElems.push_back(core::topology::TopologyElemID(geometry::ElementType::TRIANGLE,
                    triangles_ancestors[ti][0]));
            edges_added.push_back(e);
            edges_src.push_back(src);
        }
    }
    m_modifier->addEdges(edges_added, edges_src);

    sofa::Size nbEdges = m_container->getNbEdges();

    //Add and remove triangles lists
    m_modifier->addRemoveTriangles(sofa::Size(new_triangles.size()), new_triangles, new_triangles_id, triangles_ancestors, triangles_barycoefs, removed_triangles);

    sofa::Size nbEdges2 = m_container->getNbEdges();

    if (nbEdges2 > nbEdges)
    {
        msg_error() << "SplitAlongPath: auto created edges up to " << nbEdges << ", while ended up with " << nbEdges2;
    }

    //WARNING can produce error TODO: check it
    if (!points2Snap.empty())
    {
        sofa::type::vector<PointID> id2Snap;
        sofa::type::vector< sofa::type::vector< PointID > > ancestors2Snap; 
        ancestors2Snap.resize(points2Snap.size());
        sofa::type::vector< sofa::type::vector< SReal > > coefs2Snap; 
        coefs2Snap.resize(points2Snap.size());

        for (size_t i = 0; i < points2Snap.size(); i++)
        {

            sofa::type::Vec<3, Real> SnapedCoord;
            PointID firstAncestor = (PointID)points2Snap[i][4];
            PointID secondAncestor = (PointID)points2Snap[i][5];

            for (unsigned int j = 0; j < 3; j++)
                SnapedCoord[j] = points2Snap[i][j + 1];

            id2Snap.push_back((PointID)points2Snap[i][0]);

            ancestors2Snap[i].push_back(firstAncestor); //coefs2Snap[i].push_back (bary_coefs[0]);
            ancestors2Snap[i].push_back(secondAncestor); //coefs2Snap[i].push_back (bary_coefs[1]);


            if (points2Snap[i].size() == 7)
            {
                coefs2Snap[i] = compute3PointsBarycoefs(SnapedCoord, firstAncestor, secondAncestor, (PointID)points2Snap[i][6]);
                ancestors2Snap[i].push_back((PointID)points2Snap[i][6]);
            }
            else
                coefs2Snap[i] = this->computeEdgeBarycentricCoordinates(SnapedCoord, firstAncestor, secondAncestor);
        }
        m_modifier->movePointsProcess(id2Snap, ancestors2Snap, coefs2Snap);
    }

    m_modifier->notifyEndingEvent();
    m_modifier->propagateTopologicalChanges();

    for (size_t i = 0; i < new_edge_points.size() - 1; ++i)
    {
        EdgeID e = m_container->getEdgeIndex(new_edge_points[i], new_edge_points[i + 1]);

        if (e == sofa::InvalidID)
            e = m_container->getEdgeIndex(new_edge_points[i + 1], new_edge_points[i]);

        if (e == sofa::InvalidID)
            msg_error() << "Edge " << new_edge_points[i] << " - " << new_edge_points[i + 1] << " NOT FOUND.";
        else
            new_edges.push_back(e);
    }
    return (int)p_ancestors.size();
}



template<class DataTypes>
void TriangleSetGeometryAlgorithms<DataTypes>::SnapAlongPath(sofa::type::vector< sofa::geometry::ElementType>& intersected_topoElements,
    sofa::type::vector<ElemID>& intersected_indices, sofa::type::vector< Vec3 >& intersected_barycoefs,
    sofa::type::vector< sofa::type::vector<Real> >& points2Snap,
    Real epsilonSnapPath)
{
    std::map <PointID, sofa::type::vector<PointID> > map_point2snap;
    std::map <PointID, sofa::type::vector<PointID> >::iterator it;
    std::map <PointID, Vec3 > map_point2bary;
    Real epsilon = epsilonSnapPath;

    //// STEP 1 - First loop to find concerned points
    for (size_t i = 0; i < intersected_indices.size(); i++)
    {
        switch (intersected_topoElements[i])
        {
            // New case to handle other topological object can be added.
            // Default: if object is a POINT , nothing has to be done.

        case geometry::ElementType::EDGE:
        {
            PointID Vertex2Snap;

            if (intersected_barycoefs[i][0] < epsilon)  // This point has to be snapped
            {
                Vertex2Snap = m_container->getEdge(intersected_indices[i])[0];
                it = map_point2snap.find(Vertex2Snap);
            }
            else if (intersected_barycoefs[i][0] > (1.0 - epsilon))
            {
                Vertex2Snap = m_container->getEdge(intersected_indices[i])[1];
                it = map_point2snap.find(Vertex2Snap);
            }
            else
            {
                break;
            }

            if (it == map_point2snap.end()) // First time this point is encounter
            {
                map_point2snap[Vertex2Snap] = sofa::type::vector<PointID>();
                map_point2bary[Vertex2Snap] = Vec3();
            }

            break;
        }
        case geometry::ElementType::TRIANGLE:
        {
            PointID Vertex2Snap;
            Vec3& barycoord = intersected_barycoefs[i];
            bool TriFind = false;

            for (unsigned int j = 0; j < 3; j++)
            {
                if (barycoord[j] > (1.0 - epsilon))  // This point has to be snapped
                {
                    Vertex2Snap = m_container->getTriangleArray()[intersected_indices[i]][j];
                    it = map_point2snap.find(Vertex2Snap);
                    TriFind = true;
                    break;
                }
            }

            if (TriFind && (it == map_point2snap.end())) // First time this point is encounter
            {
                map_point2snap[Vertex2Snap] = sofa::type::vector<PointID>();
                map_point2bary[Vertex2Snap] = Vec3();
            }

            break;
        }
        default:
            break;
        }
    }

    //// STEP 2 - Test if snapping is needed
    if (map_point2snap.empty())
    {
        return;
    }

    const typename DataTypes::VecCoord& coords = (this->getDOF()->read(core::vec_id::read_access::position)->getValue());


    //// STEP 3 - Second loop necessary to find object on the neighborhood of a snapped point
    for (size_t i = 0; i < intersected_indices.size(); i++)
    {
        switch (intersected_topoElements[i])
        {
        case geometry::ElementType::POINT:
        {
            if (map_point2snap.contains(intersected_indices[i]))
            {
                map_point2snap[intersected_indices[i]].push_back((PointID)i);

                for (unsigned int j = 0; j < 3; j++)
                    map_point2bary[intersected_indices[i]][j] += coords[intersected_indices[i]][j];
            }
            break;
        }
        case geometry::ElementType::EDGE:
        {
            Edge theEdge = m_container->getEdge(intersected_indices[i]);
            bool PointFind = false;

            for (EdgeID indEdge = 0; indEdge < 2; indEdge++)
            {
                PointID thePoint = theEdge[indEdge];
                if (map_point2snap.contains(thePoint))
                {
                    PointFind = true;
                    map_point2snap[thePoint].push_back((PointID)i);
                    // Compute new position.
                    // Step 1/3: Compute real coord of incision point on the edge
                    const Vec3& coord_bary = computeBaryEdgePoint(theEdge, intersected_barycoefs[i][0]);

                    // Step 2/3: Sum the different incision point position.
                    for (PointID j = 0; j < 3; j++)
                        map_point2bary[thePoint][j] += coord_bary[j];
                }

                if (PointFind)
                    break;
            }
            break;
        }
        case geometry::ElementType::TRIANGLE:
        {
            Triangle theTriangle = m_container->getTriangleArray()[intersected_indices[i]];
            bool PointFind = false;

            for (TriangleID indTri = 0; indTri < 3; indTri++)
            {
                PointID thePoint = theTriangle[indTri];

                if ((map_point2snap.contains(thePoint)) && (intersected_barycoefs[i][indTri] > (1 - epsilon)))
                {
                    PointFind = true;
                    map_point2snap[thePoint].push_back((PointID)i);

                    const Vec3& coord_bary = computeBaryTrianglePoint(theTriangle, intersected_barycoefs[i]);

                    for (TriangleID j = 0; j < 3; j++)
                        map_point2bary[thePoint][j] += coord_bary[j];
                }

                if (PointFind)
                    break;
            }
            break;
        }
        default:
            break;
        }
    }

    //Pre-treatment to avoid snapping near a border:
    sofa::type::vector<PointID> field2remove;
    for (it = map_point2snap.begin(); it != map_point2snap.end(); ++it)
    {
        const sofa::type::vector<EdgeID>& shell = m_container->getEdgesAroundVertex((*it).first);
        for (size_t i = 0; i < shell.size(); i++)
            if ((m_container->getTrianglesAroundEdge(shell[i])).size() == 1)
            {
                field2remove.push_back((*it).first);
                break;
            }
    }

    //deleting point on border:
    for (size_t i = 0; i < field2remove.size(); i++)
    {
        it = map_point2snap.find(field2remove[i]);
        map_point2snap.erase(it);
    }


    //// STEP 4 - Compute new coordinates of point to be snapped, and inform path that point has to be snapped
    field2remove.clear();
    points2Snap.resize(map_point2snap.size());
    unsigned int cpt = 0;
    for (it = map_point2snap.begin(); it != map_point2snap.end(); ++it)
    {
        const size_t size = ((*it).second).size();
        if (size == 1) // for border case or reincision
        {
            points2Snap.resize(points2Snap.size() - 1);
            continue;
        }

        points2Snap[cpt].push_back((*it).first); // points2Snap[X][0] => id point to snap

        // Step 3/3: Compute mean value of all incision point position.
        for (unsigned int j = 0; j < 3; j++)
        {
            points2Snap[cpt].push_back(map_point2bary[(*it).first][j] / size); // points2Snap[X][1 2 3] => real coord of point to snap
        }
        cpt++;

        // Change enum of the first object to snap to POINT, change id and label it as snapped
        intersected_topoElements[((*it).second)[0]] = sofa::geometry::ElementType::POINT;
        intersected_indices[((*it).second)[0]] = (*it).first;
        intersected_barycoefs[((*it).second)[0]][0] = -1.0;

        // If more objects are concerned, remove them from the path  (need to stock and get out of the loop to delete them)
        for (size_t i = 1; i < size; i++)
            field2remove.push_back((*it).second[i]);
    }

    //// STEP 5 - Modify incision path
    //TODO: verify that one object can't be snapped and considered at staying at the same time
    sort(field2remove.begin(), field2remove.end());

    for (size_t i = 1; i <= field2remove.size(); i++) //Delete in reverse order
    {
        intersected_topoElements.erase(intersected_topoElements.begin() + field2remove[field2remove.size() - i]);
        intersected_indices.erase(intersected_indices.begin() + field2remove[field2remove.size() - i]);
        intersected_barycoefs.erase(intersected_barycoefs.begin() + field2remove[field2remove.size() - i]);
    }

    return;
}


template<class DataTypes>
void TriangleSetGeometryAlgorithms<DataTypes>::SnapBorderPath(PointID pa, Coord& a, PointID pb, Coord& b,
    sofa::type::vector< sofa::geometry::ElementType>& intersected_topoElements,
    sofa::type::vector<ElemID>& intersected_indices,
    sofa::type::vector< Vec3 >& intersected_barycoefs,
    sofa::type::vector< sofa::type::vector<Real> >& points2Snap,
    Real epsilonSnapBorder)
{
    bool snap_a = false;
    bool snap_b = false;
    bool intersected = true;
    Real epsilon = epsilonSnapBorder;

    // Test if point has not already been snap on a point
    for (size_t i = 0; i < points2Snap.size(); i++)
    {
        if (points2Snap[i][0] == pa)
            snap_a = true;
        else if (points2Snap[i][0] == pb)
            snap_b = true;

        if (snap_a & snap_b)
            break;
    }

    // Test if point need to be snap on an edge
    if (!snap_a  && intersected_topoElements[0] == sofa::geometry::ElementType::TRIANGLE) // this means a is not close to a point, but could be close to an edge
    {
        for (unsigned int i = 0; i < 3; i++)
        {
            if (intersected_barycoefs[0][i] < epsilon)
            {
                const EdgeID theEdge = m_container->getEdgesInTriangle(intersected_indices[0])[i];
                bool find = false;
                bool allDone = false;
                bool pointDone = false;
                PointID thePoint = 0;
                if ((m_container->getTrianglesAroundEdge(theEdge)).size() > 1) //snap to point and not edge
                {
                    for (unsigned int j = 0; j < 3; j++)
                        if (intersected_barycoefs[0][j] > 1 - epsilon)
                        {
                            thePoint = m_container->getTriangle(intersected_indices[0])[j];
                            intersected_topoElements[0] = sofa::geometry::ElementType::POINT;
                            intersected_indices[0] = thePoint;
                            find = true;
                            break;
                        }

                    if (intersected_topoElements.size() <= 2)
                        break;

                    while (find)
                    {
                        pointDone = true;
                        allDone = true;
                        if (intersected_topoElements[1] == sofa::geometry::ElementType::EDGE) // just remove or need to projection?
                        {
                            const sofa::type::vector<EdgeID>& shell = m_container->getEdgesAroundVertex(thePoint);
                            for (size_t k = 0; k < shell.size(); k++)
                            {
                                if (shell[k] == intersected_indices[1])
                                {
                                    intersected_topoElements.erase(intersected_topoElements.begin() + 1);
                                    intersected_indices.erase(intersected_indices.begin() + 1);
                                    intersected_barycoefs.erase(intersected_barycoefs.begin() + 1);
                                    allDone = false;
                                    break;
                                }
                            }
                        }
                        else if (intersected_topoElements[1] == sofa::geometry::ElementType::POINT)
                        {
                            if (intersected_indices[1] == thePoint)
                            {
                                intersected_topoElements.erase(intersected_topoElements.begin() + 1);
                                intersected_indices.erase(intersected_indices.begin() + 1);
                                intersected_barycoefs.erase(intersected_barycoefs.begin() + 1);
                                pointDone = false;
                            }
                        }
                        else
                            find = false;

                        if (pointDone && allDone) //nor one not the other
                            find = false;
                    }
                    break;
                }


                if ((intersected_indices[1] == theEdge) && (intersected_topoElements[1] == sofa::geometry::ElementType::EDGE)) // Only keep this one? or need to project?
                {
                    msg_warning() << "Unexpected case reached: where intersected_indices[1] == theEdge and is an Edge. Report this issue.";
                    intersected_topoElements.erase(intersected_topoElements.begin());
                    intersected_indices.erase(intersected_indices.begin());
                    intersected_barycoefs.erase(intersected_barycoefs.begin());
                    break;
                }
                else // need to create this point by projection
                {
                    Vec3 thePoint; DataTypes::get(thePoint[0], thePoint[1], thePoint[2], a);

                    auto new_coord = this->computePointProjectionOnEdge(theEdge, thePoint, intersected);

                    if (!intersected){
                        // Orthogonal projection failed, not possible to snap this point on the ith edge of the triangle
                        continue;
                    }

                    intersected_topoElements[0] = sofa::geometry::ElementType::EDGE;

                    intersected_indices[0] = theEdge;
                    intersected_barycoefs[0][0] = new_coord[1];  // not the same order as barycoef in the incision path
                    intersected_barycoefs[0][1] = new_coord[0];
                    intersected_barycoefs[0][2] = 0.0;

                    const Edge theEdgeFirst = m_container->getEdge(theEdge);
                    Vec3 pos1 = computeBaryEdgePoint(theEdgeFirst, new_coord[1]);
                    for (unsigned int j = 0; j < std::min(3u, a.size()); j++)
                        a[j] = (decltype (a[j]))pos1[j];

                    break;
                }
            }
        }
    }

    // Same for last point
    if (!snap_b  && intersected_topoElements.back() == sofa::geometry::ElementType::TRIANGLE) // this means a is not close to a point, but could be close to an edge
    {
        for (unsigned int i = 0; i < 3; i++)
        {
            if (intersected_barycoefs.back()[i] < epsilon)
            {
                const EdgeID theEdge = m_container->getEdgesInTriangle(intersected_indices.back())[i];
                bool find = false;
                bool allDone = false;
                bool pointDone = false;
                PointID thePoint = 0;

                if ((m_container->getTrianglesAroundEdge(theEdge)).size() > 1) //snap to point and not edge
                {
                    for (unsigned int j = 0; j < 3; j++)
                        if (intersected_barycoefs.back()[j] > 1 - epsilon)
                        {
                            thePoint = m_container->getTriangle(intersected_indices.back())[j];
                            intersected_topoElements.back() = sofa::geometry::ElementType::POINT;
                            intersected_indices.back() = thePoint;
                            find = true;
                            break;
                        }

                    if (intersected_topoElements.size() <= 2)
                        break;

                    while (find)
                    {
                        const size_t pos = intersected_topoElements.size() - 2;
                        pointDone = true;
                        allDone = true;
                        if (intersected_topoElements[pos] == sofa::geometry::ElementType::EDGE) // just remove or need to projection?
                        {
                            const sofa::type::vector<EdgeID> &shell = m_container->getEdgesAroundVertex(thePoint);
                            for (size_t k = 0; k < shell.size(); k++)
                            {
                                if (shell[k] == intersected_indices[pos])
                                {
                                    intersected_topoElements.erase(intersected_topoElements.begin() + pos);
                                    intersected_indices.erase(intersected_indices.begin() + pos);
                                    intersected_barycoefs.erase(intersected_barycoefs.begin() + pos);
                                    allDone = false;
                                    break;
                                }
                            }
                        }
                        else if (intersected_topoElements[pos] == sofa::geometry::ElementType::POINT)
                        {
                            if (intersected_indices[pos] == thePoint)
                            {
                                intersected_topoElements.erase(intersected_topoElements.begin() + pos);
                                intersected_indices.erase(intersected_indices.begin() + pos);
                                intersected_barycoefs.erase(intersected_barycoefs.begin() + pos);
                                pointDone = false;
                            }
                        }
                        else
                            find = false;

                        if (pointDone && allDone) //nor one not the other
                            find = false;
                    }

                    break;
                }


                if ((intersected_indices[intersected_indices.size() - 2] == theEdge) && (intersected_topoElements[intersected_topoElements.size() - 2] == sofa::geometry::ElementType::EDGE)) // Only keep this one? or need to projection?
                {
                    msg_warning() << "Unexpected case reached: where intersected_indices[1] == theEdge and intersected_topoElements[1] is an Edge. Report this issue.";
                    intersected_topoElements.pop_back();
                    intersected_indices.pop_back();
                    intersected_barycoefs.pop_back();
                    break;
                }
                else
                {
                    Vec3 thePoint; 
                    DataTypes::get(thePoint[0], thePoint[1], thePoint[2], b);
                    auto new_coord = this->computePointProjectionOnEdge(theEdge, thePoint, intersected);

                    if (!intersected){
                        // Orthogonal projection failed, not possible to snap this point on the ith edge of the triangle
                        continue;
                    }

                    intersected_topoElements.back() = sofa::geometry::ElementType::EDGE;
                    intersected_indices.back() = theEdge;
                    intersected_barycoefs.back()[0] = new_coord[1];
                    intersected_barycoefs.back()[1] = new_coord[0];
                    intersected_barycoefs.back()[2] = 0.0;

                    const Edge theEdgeLast = m_container->getEdge(theEdge);
                    Vec3 pos1 = computeBaryEdgePoint(theEdgeLast, new_coord[1]);
                    for (unsigned int j = 0; j < std::min(3u, a.size()); j++)
                        a[j] = (decltype (a[j]))pos1[j];

                    break;
                }
            }
        }
    }
    return;
}


/** \brief Duplicates the given edges. Only works if at least the first or last point is adjacent to a border.
 * @returns true if the incision succeeded.
 */
template<class DataTypes>
bool TriangleSetGeometryAlgorithms<DataTypes>::InciseAlongEdgeList(const sofa::type::vector<EdgeID>& edges,
    sofa::type::vector<PointID>& new_points,
    sofa::type::vector<PointID>& end_points,
    bool& reachBorder)
{
    sofa::type::vector< sofa::type::vector< PointID > > p_ancestors;
    sofa::type::vector< sofa::type::vector< SReal > > p_baryCoefs;
    PointID next_point = m_container->getNbPoints();
    TriangleID next_triangle = (TriangleID)m_container->getNbTriangles();
    sofa::type::vector< Triangle > new_triangles;
    sofa::type::vector< TriangleID > new_triangles_id;
    sofa::type::vector< TriangleID > removed_triangles;
    sofa::type::vector< sofa::type::vector< TriangleID > >  triangles_ancestors;
    sofa::type::vector< sofa::type::vector< SReal > >  triangles_barycoefs;


    const size_t nbEdges = edges.size();
    if (nbEdges == 0) return true;
    sofa::type::vector<PointID> init_points;
    Edge edge;
    edge = m_container->getEdge(edges[0]);
    init_points.push_back(edge[0]);
    init_points.push_back(edge[1]);
    if (nbEdges > 1)
    {
        edge = m_container->getEdge(edges[1]);
        if (init_points[0] == edge[0] || init_points[0] == edge[1])
        {
            // swap the first points
            PointID t = init_points[0];
            init_points[0] = init_points[1];
            init_points[1] = t;
        }
        // add the rest of the points
        for (size_t i = 1; i < nbEdges; ++i)
        {
            edge = m_container->getEdge(edges[i]);
            if (edge[0] == init_points.back())
                init_points.push_back(edge[1]);
            else if (edge[1] == init_points.back())
                init_points.push_back(edge[0]);
            else
            {
                msg_error() << "Edges are not connected after number " << i - 1 << " : " << edges;
                return false;
            }
        }
    }

    sofa::type::vector< std::pair<TriangleID, TriangleID> > init_triangles;
    for (size_t i = 0; i < nbEdges; ++i)
    {
        const sofa::type::vector<TriangleID>& shell = m_container->getTrianglesAroundEdge(edges[i]);
        if (shell.size() != 2)
        {
            msg_error() << "Cannot split an edge with " << shell.size() << "!=2 attached triangles. Around edge: " << edges[i];
            msg_error() << "Which is composed of vertex: " << m_container->getEdge(edges[i]);
            return false;
        }
        init_triangles.push_back(std::make_pair(shell[0], shell[1]));
    }

    bool beginOnBorder = (m_container->getTrianglesAroundVertex(init_points.front()).size() < m_container->getEdgesAroundVertex(init_points.front()).size());
    bool endOnBorder = (m_container->getTrianglesAroundVertex(init_points.back()).size() < m_container->getEdgesAroundVertex(init_points.back()).size());

    if (!beginOnBorder) end_points.push_back(init_points.front());
    if (!endOnBorder) end_points.push_back(init_points.back());
    else
        reachBorder = true;

    /// STEP 1: Create the new points corresponding the one of the side of the now separated edges
    const size_t first_new_point = beginOnBorder ? 0 : 1;
    const size_t last_new_point = endOnBorder ? init_points.size() - 1 : init_points.size() - 2;
    std::map<PointID, PointID> splitMap;
    for (size_t i = first_new_point; i <= last_new_point; ++i)
    {
        PointID p = init_points[i];
        p_ancestors.resize(p_ancestors.size() + 1);
        sofa::type::vector< PointID >& ancestors = p_ancestors.back();
        p_baryCoefs.resize(p_baryCoefs.size() + 1);
        auto& baryCoefs = p_baryCoefs.back();
        ancestors.push_back(p);
        baryCoefs.push_back(1.0);
        new_points.push_back(next_point);
        splitMap[p] = next_point;
        ++next_point;
    }

    // STEP 2: Find all triangles that need to be attached to the new points
    std::set<TriangleID> updatedTriangles;

    //TODO : WARNING THERE SEEMS TO BE A SEG FAULT HERE
    TriangleID t0 = m_container->getTrianglesAroundEdge(edges[0])[0];
    if (beginOnBorder)
    {
        // STEP 2a: Find the triangles linking the first edge to the border
        TriangleID tid = t0;
        PointID p0 = init_points[0];
        PointID p1 = init_points[1];
        for (;;)
        {
            updatedTriangles.insert(tid);
            Triangle t = m_container->getTriangle(tid);
            PointID p2 = m_container->getOtherPointInTriangle(t, p0, p1);
            EdgeID e = m_container->getEdgeIndex(p0, p2);
            const sofa::core::topology::BaseMeshTopology::TrianglesAroundEdge& etri = m_container->getTrianglesAroundEdge(e);
            if (etri.size() != 2) break; // border or non-manifold edge
            if (etri[0] == tid)
                tid = etri[1];
            else
                tid = etri[0];
            p1 = p2;
        }
    }

    // STEP 2b: Find the triangles linking each edge to the next, by starting from the last triangle, rotate around each point until the next point is reached
    for (size_t i = 0; i < nbEdges - 1; ++i)
    {
        PointID p1 = init_points[i];
        PointID p0 = init_points[i + 1];
        PointID pnext = init_points[i + 2];
        TriangleID tid = t0;
        for (;;)
        {
            updatedTriangles.insert(tid);
            Triangle t = m_container->getTriangle(tid);
            PointID p2 = m_container->getOtherPointInTriangle(t, p0, p1);
            if (p2 == pnext) break;
            EdgeID e = m_container->getEdgeIndex(p0, p2);

            const sofa::core::topology::BaseMeshTopology::TrianglesAroundEdge& etri = m_container->getTrianglesAroundEdge(e);
            if (etri.size() < 2) break; // border or non-manifold edge
            if (etri[0] == tid)
                tid = etri[1];
            else
                tid = etri[0];
            p1 = p2;
        }
        t0 = tid;
    }
    if (endOnBorder)
    {
        // STEP 2c: Find the triangles linking the last edge to the border
        TriangleID tid = t0;
        PointID p0 = init_points[nbEdges];
        PointID p1 = init_points[nbEdges - 1];
        for (;;)
        {
            updatedTriangles.insert(tid);
            Triangle t = m_container->getTriangle(tid);
            PointID p2 = m_container->getOtherPointInTriangle(t, p0, p1);
            EdgeID e = m_container->getEdgeIndex(p0, p2);
            const sofa::core::topology::BaseMeshTopology::TrianglesAroundEdge& etri = m_container->getTrianglesAroundEdge(e);
            if (etri.size() != 2) break; // border or non-manifold edge
            if (etri[0] == tid)
                tid = etri[1];
            else
                tid = etri[0];
            p1 = p2;
        }
    }

    // STEP 3: Create new triangles by replacing indices of split points in the list of triangles to update
    for (std::set<TriangleID>::const_iterator it = updatedTriangles.begin(), itend = updatedTriangles.end(); it != itend; ++it)
    {
        TriangleID tid = *it;
        Triangle t = m_container->getTriangle(tid);
        bool changed = false;
        for (int c = 0; c < 3; ++c)
        {
            std::map<PointID, PointID>::iterator itsplit = splitMap.find(t[c]);
            if (itsplit != splitMap.end())
            {
                t[c] = itsplit->second;
                changed = true;
            }
        }
        if (!changed)
        {
            msg_error() << "Triangle " << tid << " ( " << t << " ) was flagged as updated but no change was found.";
        }
        else
        {
            new_triangles.push_back(t);
            new_triangles_id.push_back(next_triangle++);
            removed_triangles.push_back(tid);

            // Taking into account ancestors for adding triangles
            triangles_ancestors.resize(triangles_ancestors.size() + 1);
            triangles_barycoefs.resize(triangles_barycoefs.size() + 1);

            triangles_ancestors[triangles_ancestors.size() - 1].push_back(tid);
            triangles_barycoefs[triangles_barycoefs.size() - 1].push_back(1.0); //that is the question... ??
        }
    }

    // FINAL STEP : Apply changes
    // Create all the points registered to be created
    m_modifier->addPointsProcess(sofa::Size(p_ancestors.size()));

    // Warn for the creation of all the points registered to be created
    m_modifier->addPointsWarning(sofa::Size(p_ancestors.size()), p_ancestors, p_baryCoefs);

    //Add and remove triangles lists
    m_modifier->addRemoveTriangles(sofa::Size(new_triangles.size()), new_triangles, new_triangles_id, triangles_ancestors, triangles_barycoefs, removed_triangles);

    return true;
}





template<class DataTypes>
void TriangleSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    EdgeSetGeometryAlgorithms<DataTypes>::draw(vparams);

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    // Draw Triangles indices
    if (showTriangleIndices.getValue() && this->m_topology->getNbTriangles() != 0)
    {
        const VecCoord& coords =(this->object->read(core::vec_id::read_access::position)->getValue());
        float scale = this->getIndicesScale();

        //for triangles:
        scale = scale/2;

        const sofa::type::vector<Triangle> &triangleArray = this->m_topology->getTriangles();

        std::vector<type::Vec3> positions;
        for (size_t i =0; i<triangleArray.size(); i++)
        {

            Triangle the_tri = triangleArray[i];
            Coord vertex1 = coords[ the_tri[0] ];
            Coord vertex2 = coords[ the_tri[1] ];
            Coord vertex3 = coords[ the_tri[2] ];
            type::Vec3 center = type::Vec3((DataTypes::getCPos(vertex1)+DataTypes::getCPos(vertex2)+DataTypes::getCPos(vertex3))/3);

            positions.push_back(center);

        }
        vparams->drawTool()->draw3DText_Indices(positions, scale, _drawColor.getValue());
    }



    if (_draw.getValue() && this->m_topology->getNbTriangles() != 0)
    {
        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, true);

        const sofa::type::vector<Triangle> &triangleArray = this->m_topology->getTriangles();

        // Draw triangle surfaces
        const VecCoord& coords =(this->object->read(core::vec_id::read_access::position)->getValue());

        {//   Draw Triangles
            std::vector<type::Vec3> pos;
            pos.reserve(triangleArray.size()*3);
            for (size_t i = 0; i<triangleArray.size(); i++)
            {
                const Triangle& t = triangleArray[i];

                type::Vec3 bary = type::Vec3(0.0, 0.0, 0.0);
                std::vector<type::Vec3> tmpPos;
                tmpPos.resize(3);

                for (unsigned int j = 0; j<3; j++)
                {
                    tmpPos[j] = type::Vec3(DataTypes::getCPos(coords[t[j]]));
                    bary += tmpPos[j];
                }
                bary /= 3;

                for (unsigned int j = 0; j<3; j++)
                    pos.push_back(bary*0.1 + tmpPos[j]*0.9);
            }
            vparams->drawTool()->drawTriangles(pos,_drawColor.getValue());
        }

        if (!vparams->displayFlags().getShowWireFrame())
        {//   Draw triangle edges for better display
            const sofa::type::vector<Edge> &edgeArray = this->m_topology->getEdges();
            std::vector<type::Vec3> pos;
            if (!edgeArray.empty())
            {
                for (size_t i = 0; i<edgeArray.size(); i++)
                {
                    const Edge& e = edgeArray[i];
                    pos.push_back(type::Vec3(DataTypes::getCPos(coords[e[0]])));
                    pos.push_back(type::Vec3(DataTypes::getCPos(coords[e[1]])));
                }
            } else {
                for (size_t i = 0; i<triangleArray.size(); i++)
                {
                    const Triangle& t = triangleArray[i];

                    for (unsigned int j = 0; j<3; j++)
                    {
                        pos.push_back(type::Vec3(DataTypes::getCPos(coords[t[j]])));
                        pos.push_back(type::Vec3(DataTypes::getCPos(coords[t[(j+1u)%3u]])));
                    }
                }
            }

            sofa::type::RGBAColor colorL = _drawColor.getValue();
            for (auto& c: colorL)
                c /= 2;
            vparams->drawTool()->drawLines(pos, 1.0f, colorL);
        }

        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, false);
    }


    if (_drawNormals.getValue() && this->m_topology->getNbTriangles() != 0)
    {
        const VecCoord& coords =(this->object->read(core::vec_id::read_access::position)->getValue());
        const sofa::type::vector<Triangle> &triangleArray = this->m_topology->getTriangles();
        const size_t nbrTtri = triangleArray.size();

        sofa::type::RGBAColor color;
        Real normalLength = _drawNormalLength.getValue();

        sofa::type::vector<sofa::type::Vec3> vertices;
        std::vector<sofa::type::RGBAColor> colors;

        for (size_t i =0; i<nbrTtri; i++)
        {
            Triangle _tri = triangleArray[i];
            sofa::type::Vec<3,Real> normal = this->computeTriangleNormal((TriangleID)i);
            normal.normalize();

            // compute bary triangle
            Coord vertex1 = coords[ _tri[0] ];
            Coord vertex2 = coords[ _tri[1] ];
            Coord vertex3 = coords[ _tri[2] ];
            const sofa::type::Vec3 center = type::toVec3((DataTypes::getCPos(vertex1)+DataTypes::getCPos(vertex2)+DataTypes::getCPos(vertex3))/3);
            const sofa::type::Vec3 point2 = center + normal*normalLength;

            for(unsigned int j=0; j<3; j++)
                color[j] = (float)fabs(normal[j]);

            vertices.push_back(center);
            vertices.push_back(point2);
            colors.push_back(color);
        }
        vparams->drawTool()->drawLines(vertices,1.0f,colors);
    }


}


} //namespace sofa::component::topology::container::dynamic
