/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETGEOMETRYALGORITHMS_INL

#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/CommonAlgorithms.h>
#include <SofaBaseTopology/NumericalIntegrationDescriptor.inl>
#include <fstream>
namespace sofa
{

namespace component
{

namespace topology
{

const unsigned int edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};

template< class DataTypes>
NumericalIntegrationDescriptor<typename TetrahedronSetGeometryAlgorithms< DataTypes >::Real,4> &TetrahedronSetGeometryAlgorithms< DataTypes >::getTetrahedronNumericalIntegrationDescriptor()
{
    // initialize the cubature table only if needed.
    if (initializedCubatureTables==false) {
        initializedCubatureTables=true;
        defineTetrahedronCubaturePoints();
    }
    return tetrahedronNumericalIntegration;
}

template< class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::defineTetrahedronCubaturePoints() {
    typedef typename NumericalIntegrationDescriptor<typename TetrahedronSetGeometryAlgorithms< DataTypes >::Real,4>::QuadraturePoint QuadraturePoint;
    typedef typename NumericalIntegrationDescriptor<typename TetrahedronSetGeometryAlgorithms< DataTypes >::Real,4>::BarycentricCoordinatesType BarycentricCoordinatesType;
    // Gauss method
    typename NumericalIntegrationDescriptor<typename TetrahedronSetGeometryAlgorithms< DataTypes >::Real,4>::QuadratureMethod m=NumericalIntegrationDescriptor<typename TetrahedronSetGeometryAlgorithms< DataTypes >::Real,4>::GAUSS_SIMPLEX_METHOD;
    typename NumericalIntegrationDescriptor<typename TetrahedronSetGeometryAlgorithms< DataTypes >::Real,4>::QuadraturePointArray qpa;
    BarycentricCoordinatesType v;
    /// integration with linear accuracy.
    v=BarycentricCoordinatesType(0.25,0.25,0.25,0.25);
    qpa.push_back(QuadraturePoint(v,1/(Real)6));
    tetrahedronNumericalIntegration.addQuadratureMethod(m,1,qpa);
    /// integration with quadratic accuracy.
    qpa.clear();
    Real a=(5-sqrt((Real)5))/20;
    Real b=1-3*a;
    size_t i;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,1/(Real)24));
    }
    tetrahedronNumericalIntegration.addQuadratureMethod(m,2,qpa);
    /// integration with cubic accuracy.
    qpa.clear();
    v=BarycentricCoordinatesType(0.25,0.25,0.25,0.25);
    qpa.push_back(QuadraturePoint(v,(Real) -2/15));
    a=(Real)1/6;
    b=(Real)1/2;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,(Real)15/(Real)200));
    }
    tetrahedronNumericalIntegration.addQuadratureMethod(m,3,qpa);
    /// integration with quadric accuracy.
    qpa.clear();
    v=BarycentricCoordinatesType(0.25,0.25,0.25,0.25);
    Real c= -0.131555555555555556e-01;
    qpa.push_back(QuadraturePoint(v,(Real)c ));
    a=(Real)0.714285714285714285e-01;
    b=(Real)1-3*a;
    Real c1=0.762222222222222222e-02;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    a=(Real)0.399403576166799219;
    b=(Real)0.5-a;
    Real c2=((Real)1/6 -c-4*c1)/6;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,c2));
    }
    tetrahedronNumericalIntegration.addQuadratureMethod(m,4,qpa);
    /// integration with quintic accuracy and 14 points
    /// Walkington's fifth-order 14-point rule from
  // "Quadrature on Simplices of Arbitrary Dimension"
    qpa.clear();
    a=(Real)0.31088591926330060980;
    b=(Real)1-3*a;
    c1=(Real)0.018781320953002641800;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    a=(Real)  0.092735250310891226402;
    b=(Real)1-3*a;
    c2=(Real)0.012248840519393658257;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c2));
    }
    a=(Real)0.045503704125649649492;
    b=(Real)0.5-a;
    Real c3=((Real)1/6 -4*c1-4*c2)/6;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,c3));
    }
    tetrahedronNumericalIntegration.addQuadratureMethod(m,5,qpa);

/*
    v=BarycentricCoordinatesType(0.25,0.25,0.25,0.25);
    qpa.push_back(QuadraturePoint(v,(Real) 8/405));
    a=(7+sqrt((Real)15))/34;
    b=(13+3*sqrt((Real)15))/34;
     c=(2665-14*sqrt((Real)15))/226800;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c));
    }
    a=(7-sqrt((Real)15))/34;
    b=(13-3*sqrt((Real)15))/34;
    c=(2665+14*sqrt((Real)15))/226800;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c));
    }
    a=(5-sqrt((Real)15))/20;
    b=(5+sqrt((Real)15))/20;
    c=(Real)5/567;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,c));
    }
    tetrahedronNumericalIntegration.addQuadratureMethod(m,5,qpa); */
    /// integration with sixtic accuracy with 24 points
    // This rule is originally from Keast:
    // Patrick Keast,
    // Moderate Degree Tetrahedral Quadrature Formulas,
    // Computer Methods in Applied Mechanics and Engineering,
    // Volume 55, Number 3, May 1986, pages 339-348.
    qpa.clear();
    a=(Real) 0.214602871259151684;
    b=1-3*a;
    c1=(Real)  0.00665379170969464506;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c1));
    }
    a=(Real)0.0406739585346113397;
    b=1-3*a;
    c2=(Real)  0.00167953517588677620;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c2));
    }
    a=(Real) 0.322337890142275646;
    b=1-3*a;
    c3=(Real)  0.00922619692394239843;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c3));
    }
    a=(Real)0.0636610018750175299;
    b=(Real)0.269672331458315867;
    c=(Real) 0.603005664791649076;
    Real d=(Real) ((Real)1/6 -4*c1-4*c2-4*c3)/12;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=c;
        qpa.push_back(QuadraturePoint(v,d));
        v[edgesInTetrahedronArray[i][0]]=c;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,d));
    }
    tetrahedronNumericalIntegration.addQuadratureMethod(m,6,qpa);
    /// integration with  accuracy of order 8 with 45 points
    // This rule is originally from Keast:
    // Patrick Keast,
    // Moderate Degree Tetrahedral Quadrature Formulas,
    // Computer Methods in Applied Mechanics and Engineering,
    // Volume 55, Number 3, May 1986, pages 339-348.
    qpa.clear();
    v=BarycentricCoordinatesType(0.25,0.25,0.25,0.25);
    c1= (Real)-0.393270066412926145e-01;
    qpa.push_back(QuadraturePoint(v,(Real)c1 ));
    // 4 points
    a=(Real)0.127470936566639015;
    b=1-3*a;
    c2=(Real)   0.408131605934270525e-02;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c2));
    }
    // 4 points
    a=(Real)0.320788303926322960e-01;
    b=1-3*a;
    c3=(Real)   0.658086773304341943e-03;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c3));
    }
    // 6 points
    a=(Real)0.497770956432810185e-01;
    b=(Real)0.5-a;
    Real c4= (Real) 0.438425882512284693e-02;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,c4));
    }
    // 6 points
    a=(Real)0.183730447398549945;
    b=(Real)0.5-a;
    Real c5= (Real) 0.138300638425098166e-01;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,c5));
    }
    /// 12 points
    a=(Real)0.231901089397150906;
    b=(Real) 0.229177878448171174e-01;
    c=(Real) 1-2*a-b;
    d=(Real) 0.424043742468372453e-02;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=c;
        qpa.push_back(QuadraturePoint(v,d));
        v[edgesInTetrahedronArray[i][0]]=c;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,d));
    }
    /// 12 points
    a=(Real)0.379700484718286102e-01;
    b=(Real)  0.730313427807538396;
    c=(Real) 1-2*a-b;
    Real d1=((Real) 1/6 -12*d-6*(c5+c4)-4*(c3+c2)-c1)/12;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=c;
        qpa.push_back(QuadraturePoint(v,d1));
        v[edgesInTetrahedronArray[i][0]]=c;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,d1));
    }
    tetrahedronNumericalIntegration.addQuadratureMethod(m,8,qpa);
    /// integration with  accuracy of order 10 with 81 points
    // This rule is originally from
    // A SET OF SYMMETRIC QUADRATURE RULES
    // ON TRIANGLES AND TETRAHEDRA*
    // Linbo Zhang, Tao Cui and Hui Liu:
    // See software PHG http://lsec.cc.ac.cn/phg/
    qpa.clear();
    v=BarycentricCoordinatesType(0.25,0.25,0.25,0.25);
    c1= (Real).04574189830483037077884770618329337/6.0;
    qpa.push_back(QuadraturePoint(v,(Real)c1 ));
    // 4 points
    a=(Real).11425191803006935688146412277598412;
    b=1-3*a;
    c3=(Real)0.01092727610912416907498417206565671/6.0;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c3));
    }
    // 4 points
    a=(Real).01063790234539248531264164411274776;
    b=1-3*a;
    c3=(Real).00055352334192264689534558564012282/6.0;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c3));
    }
    // 4 points
    a=(Real).31274070833535645859816704980806110;
    b=1-3*a;
    c3=(Real).02569337913913269580782688316792080/6.0;
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c3));
    }
    // 6 points
    a=(Real).01631296303281644000000000000000000;
    b=(Real)0.5-a;
    c5= (Real) .00055387649657283109312967562590035/6.0;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,c5));
    }
    /// 12 points
    a=(Real).03430622963180452385835196582344460;
    b=(Real) .59830121060139461905983787517050400;
    c=(Real) 1-2*a-b;
    d=(Real) .01044842402938294329072628200105773/6.0;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=c;
        qpa.push_back(QuadraturePoint(v,d));
        v[edgesInTetrahedronArray[i][0]]=c;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,d));
    }
    /// 12 points
    a=(Real).12346418534551115945916818783743644;
    b=(Real).47120066204746310257913700590727081;
    c=(Real) 1-2*a-b;
    d=(Real) .02513844602651287118280517785487423/6.0;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=c;
        qpa.push_back(QuadraturePoint(v,d));
        v[edgesInTetrahedronArray[i][0]]=c;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,d));
    }
    /// 12 points
    a=(Real).40991962933181117418479812480531207;
    b=(Real).16546413290740130923509687990363569;
    c=(Real) 1-2*a-b;
    d=(Real) .01178620679249594711782155323755017/6.0;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=c;
        qpa.push_back(QuadraturePoint(v,d));
        v[edgesInTetrahedronArray[i][0]]=c;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,d));
    }
    /// 12 points
    a=(Real).17397243903011716743177479785668929;
    b=(Real).62916375300275643773181882027844514;
    c=(Real) 1-2*a-b;
    d=(Real) .01332022473886650471019828463616468/6.0;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=c;
        qpa.push_back(QuadraturePoint(v,d));
        v[edgesInTetrahedronArray[i][0]]=c;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,d));
    }
    /// 12 points
    a=(Real).03002157005631784150255786784038011;
    b=(Real).81213056814351208262160080755918730;
    c=(Real) 1-2*a-b;
    d=(Real) .00615987577565961666092767531756180/6.0;
    for (i=0;i<6;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[edgesInTetrahedronArray[i][0]]=b;
        v[edgesInTetrahedronArray[i][1]]=c;
        qpa.push_back(QuadraturePoint(v,d));
        v[edgesInTetrahedronArray[i][0]]=c;
        v[edgesInTetrahedronArray[i][1]]=b;
        qpa.push_back(QuadraturePoint(v,d));
    }
    tetrahedronNumericalIntegration.addQuadratureMethod(m,10,qpa);
}



template< class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::computeTetrahedronAABB(const TetraID i, Coord& minCoord, Coord& maxCoord) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    for(unsigned int i=0; i<3; ++i)
    {
        minCoord[i] = std::min(std::min(p[t[0]][i], p[t[3]][i]), std::min(p[t[1]][i], p[t[2]][i]));
        maxCoord[i] = std::max(std::max(p[t[0]][i], p[t[3]][i]), std::max(p[t[1]][i], p[t[2]][i]));
    }
}

template<class DataTypes>
typename DataTypes::Coord TetrahedronSetGeometryAlgorithms<DataTypes>::computeTetrahedronCenter(const TetraID i) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    return (p[t[0]] + p[t[1]] + p[t[2]] + p[t[3]]) * (Real) 0.25;
}

template<class DataTypes>
typename DataTypes::Coord TetrahedronSetGeometryAlgorithms<DataTypes>::computeTetrahedronCircumcenter(const TetraID i) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    Coord center = p[t[0]];
    Coord t1 = p[t[1]] - p[t[0]];
    Coord t2 = p[t[2]] - p[t[0]];
    Coord t3 = p[t[3]] - p[t[0]];
    sofa::defaulttype::Vec<3,Real> a(t1[0], t1[1], t1[2]);
    sofa::defaulttype::Vec<3,Real> b(t2[0], t2[1], t2[2]);
    sofa::defaulttype::Vec<3,Real> c(t3[0], t3[1], t3[2]);
    sofa::defaulttype::Vec<3,Real> d = (cross(b, c) * a.norm2() + cross(c, a) * b.norm2() + cross(a, b) * c.norm2()) / (12* computeTetrahedronVolume(i));

    center[0] += d[0];
    center[1] += d[1];
    center[2] += d[2];

    return center;
}

template< class DataTypes>
bool TetrahedronSetGeometryAlgorithms< DataTypes >::isPointInTetrahedron(const TetraID ind_t, const sofa::defaulttype::Vec<3,Real>& pTest) const
{
    const double ZERO = 1e-15;

    const Tetrahedron t = this->m_topology->getTetrahedron(ind_t);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    const sofa::defaulttype::Vec<3,Real> t0(p[t[0]][0], p[t[0]][1], p[t[0]][2]);
    const sofa::defaulttype::Vec<3,Real> t1(p[t[1]][0], p[t[1]][1], p[t[1]][2]);
    const sofa::defaulttype::Vec<3,Real> t2(p[t[2]][0], p[t[2]][1], p[t[2]][2]);
    const sofa::defaulttype::Vec<3,Real> t3(p[t[3]][0], p[t[3]][1], p[t[3]][2]);

    double v0 = tripleProduct(t1-pTest, t2-pTest, t3-pTest);
    double v1 = tripleProduct(pTest-t0, t2-t0, t3-t0);
    double v2 = tripleProduct(t1-t0, pTest-t0, t3-t0);
    double v3 = tripleProduct(t1-t0, t2-t0, pTest-t0);

    double V = tripleProduct(t1-t0, t2-t0, t3-t0);
    if(fabs(V)>ZERO)
        return (v0/V > -ZERO) && (v1/V > -ZERO) && (v2/V > -ZERO) && (v3/V > -ZERO);

    else
        return false;
}

template< class DataTypes>
bool TetrahedronSetGeometryAlgorithms< DataTypes >::isPointInTetrahedron(const TetraID ind_t, const sofa::defaulttype::Vec<3,Real>& pTest, sofa::defaulttype::Vec<4,Real>& shapeFunctions) const
{
    const double ZERO = 1e-15;

    const Tetrahedron t = this->m_topology->getTetrahedron(ind_t);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    const sofa::defaulttype::Vec<3,Real> t0(p[t[0]][0], p[t[0]][1], p[t[0]][2]);
    const sofa::defaulttype::Vec<3,Real> t1(p[t[1]][0], p[t[1]][1], p[t[1]][2]);
    const sofa::defaulttype::Vec<3,Real> t2(p[t[2]][0], p[t[2]][1], p[t[2]][2]);
    const sofa::defaulttype::Vec<3,Real> t3(p[t[3]][0], p[t[3]][1], p[t[3]][2]);

    double v0 = tripleProduct(t1-pTest, t2-pTest, t3-pTest);
    double v1 = tripleProduct(pTest-t0, t2-t0, t3-t0);
    double v2 = tripleProduct(t1-t0, pTest-t0, t3-t0);
    double v3 = tripleProduct(t1-t0, t2-t0, pTest-t0);

    double V = tripleProduct(t1-t0, t2-t0, t3-t0);
    if(fabs(V)>ZERO)
    {
        if( (v0/V > -ZERO) && (v1/V > -ZERO) && (v2/V > -ZERO) && (v3/V > -ZERO) )
        {
            shapeFunctions[0] = (Real)( v0/V );
            shapeFunctions[1] = (Real)( v1/V );
            shapeFunctions[2] = (Real)( v2/V );
            shapeFunctions[3] = (Real)( v3/V );
            return true;
        }
        else
            return false;
    }

    else
        return false;
}


template< class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getTetrahedronVertexCoordinates(const TetraID i, Coord pnt[4]) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    for(unsigned int i=0; i<4; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getRestTetrahedronVertexCoordinates(const TetraID i, Coord pnt[4]) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());

    for(unsigned int i=0; i<4; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeTetrahedronVolume( const TetraID i) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
    Real volume = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    if(volume<0)
        volume=-volume;
    return volume;
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeRestTetrahedronVolume( const TetraID i) const
{
    const Tetrahedron t=this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());
    Real volume = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    if(volume<0)
        volume=-volume;
    return volume;
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeRestTetrahedronVolume( const Tetrahedron t) const
{
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());
    Real volume = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    if(volume<0)
        volume=-volume;
    return volume;
}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::computeTetrahedronVolume( BasicArrayInterface<Real> &ai) const
{
    const sofa::helper::vector<Tetrahedron> &ta = this->m_topology->getTetrahedra();
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
    for (unsigned int i=0; i<ta.size(); ++i)
    {
        const Tetrahedron &t = ta[i];
        ai[i] = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    }
}

/// Finds the indices of all tetrahedra in the ball of center ind_ta and of radius dist(ind_ta, ind_tb)
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getTetraInBall(const TetraID ind_ta, const TetraID ind_tb,
        sofa::helper::vector<unsigned int> &indices) const
{
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::ConstVecCoordId::position())->getValue());

    const Tetrahedron ta=this->m_topology->getTetrahedron(ind_ta);
    const Tetrahedron tb=this->m_topology->getTetrahedron(ind_tb);

    const typename DataTypes::Coord& ca=(vect_c[ta[0]]+vect_c[ta[1]]+vect_c[ta[2]]+vect_c[ta[3]])*0.25;
    const typename DataTypes::Coord& cb=(vect_c[tb[0]]+vect_c[tb[1]]+vect_c[tb[2]]+vect_c[tb[3]])*0.25;
    sofa::defaulttype::Vec<3,Real> pa;
    sofa::defaulttype::Vec<3,Real> pb;
    pa[0] = (Real) (ca[0]);
    pa[1] = (Real) (ca[1]);
    pa[2] = (Real) (ca[2]);
    pb[0] = (Real) (cb[0]);
    pb[1] = (Real) (cb[1]);
    pb[2] = (Real) (cb[2]);

    Real d = (pa-pb)*(pa-pb);

    getTetraInBall(ind_ta, d, indices);
}

/// Finds the indices of all tetrahedra in the ball of center ind_ta and of radius dist(ind_ta, ind_tb)
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getTetraInBall(const TetraID ind_ta, Real r,
        sofa::helper::vector<unsigned int> &indices) const
{
    Real d = r;
    const Tetrahedron ta=this->m_topology->getTetrahedron(ind_ta);
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::ConstVecCoordId::position())->getValue());
    const typename DataTypes::Coord& ca=(vect_c[ta[0]]+vect_c[ta[1]]+vect_c[ta[2]]+vect_c[ta[3]])*0.25;

    sofa::defaulttype::Vec<3,Real> pa;
    pa[0] = (Real) (ca[0]);
    pa[1] = (Real) (ca[1]);
    pa[2] = (Real) (ca[2]);

    unsigned int t_test=ind_ta;
    indices.push_back(t_test);

    std::map<unsigned int, unsigned int> IndexMap;
    IndexMap.clear();
    IndexMap[t_test]=0;

    sofa::helper::vector<unsigned int> ind2test;
    ind2test.push_back(t_test);
    sofa::helper::vector<unsigned int> ind2ask;
    ind2ask.push_back(t_test);

    while(ind2test.size()>0)
    {
        ind2test.clear();
        for (unsigned int t=0; t<ind2ask.size(); t++)
        {
            unsigned int ind_t = ind2ask[t];
            core::topology::BaseMeshTopology::TrianglesInTetrahedron adjacent_triangles = this->m_topology->getTrianglesInTetrahedron(ind_t);

            for (unsigned int i=0; i<adjacent_triangles.size(); i++)
            {
                sofa::helper::vector< unsigned int > tetrahedra_to_remove = this->m_topology->getTetrahedraAroundTriangle(adjacent_triangles[i]);

                if(tetrahedra_to_remove.size()==2)
                {
                    if(tetrahedra_to_remove[0]==ind_t)
                    {
                        t_test=tetrahedra_to_remove[1];
                    }
                    else
                    {
                        t_test=tetrahedra_to_remove[0];
                    }

                    std::map<unsigned int, unsigned int>::iterator iter_1 = IndexMap.find(t_test);
                    if(iter_1 == IndexMap.end())
                    {
                        IndexMap[t_test]=0;

                        const Tetrahedron tc=this->m_topology->getTetrahedron(t_test);
                        const typename DataTypes::Coord& cc = (vect_c[tc[0]]
                                + vect_c[tc[1]]
                                + vect_c[tc[2]]
                                + vect_c[tc[3]]) * 0.25;
                        sofa::defaulttype::Vec<3,Real> pc;
                        pc[0] = (Real) (cc[0]);
                        pc[1] = (Real) (cc[1]);
                        pc[2] = (Real) (cc[2]);

                        Real d_test = (pa-pc)*(pa-pc);

                        if(d_test<d)
                        {
                            ind2test.push_back(t_test);
                            indices.push_back(t_test);
                        }
                    }
                }
            }
        }

        ind2ask.clear();
        for (unsigned int t=0; t<ind2test.size(); t++)
        {
            ind2ask.push_back(ind2test[t]);
        }
    }

    return;
}

/// Finds the indices of all tetrahedra in the ball of center c and of radius r
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getTetraInBall(const Coord& c, Real r,
        sofa::helper::vector<unsigned int> &indices) const
{
    TetraID ind_ta = core::topology::BaseMeshTopology::InvalidID;
    sofa::defaulttype::Vec<3,Real> pa;
    pa[0] = (Real) (c[0]);
    pa[1] = (Real) (c[1]);
    pa[2] = (Real) (c[2]);
    for(int i = 0; i < this->m_topology->getNbTetrahedra(); ++i)
    {
        if(isPointInTetrahedron(i, pa))
        {
            ind_ta = i;
            break;
        }
    }
    if(ind_ta == core::topology::BaseMeshTopology::InvalidID)
        std::cout << "ERROR: Can't find the seed" << std::endl;
    Real d = r;
//      const Tetrahedron &ta=this->m_topology->getTetrahedron(ind_ta);
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::ConstVecCoordId::position())->getValue());

    unsigned int t_test=ind_ta;
    indices.push_back(t_test);

    std::map<unsigned int, unsigned int> IndexMap;
    IndexMap.clear();
    IndexMap[t_test]=0;

    sofa::helper::vector<unsigned int> ind2test;
    ind2test.push_back(t_test);
    sofa::helper::vector<unsigned int> ind2ask;
    ind2ask.push_back(t_test);

    while(ind2test.size()>0)
    {
        ind2test.clear();
        for (unsigned int t=0; t<ind2ask.size(); t++)
        {
            unsigned int ind_t = ind2ask[t];
            core::topology::BaseMeshTopology::TrianglesInTetrahedron adjacent_triangles = this->m_topology->getTrianglesInTetrahedron(ind_t);

            for (unsigned int i=0; i<adjacent_triangles.size(); i++)
            {
                sofa::helper::vector< unsigned int > tetrahedra_to_remove = this->m_topology->getTetrahedraAroundTriangle(adjacent_triangles[i]);

                if(tetrahedra_to_remove.size()==2)
                {
                    if(tetrahedra_to_remove[0]==ind_t)
                    {
                        t_test=tetrahedra_to_remove[1];
                    }
                    else
                    {
                        t_test=tetrahedra_to_remove[0];
                    }

                    std::map<unsigned int, unsigned int>::iterator iter_1 = IndexMap.find(t_test);
                    if(iter_1 == IndexMap.end())
                    {
                        IndexMap[t_test]=0;

                        const Tetrahedron tc=this->m_topology->getTetrahedron(t_test);
                        const typename DataTypes::Coord& cc = (vect_c[tc[0]]
                                + vect_c[tc[1]]
                                + vect_c[tc[2]]
                                + vect_c[tc[3]]) * 0.25;
                        sofa::defaulttype::Vec<3,Real> pc;
                        pc[0] = (Real) (cc[0]);
                        pc[1] = (Real) (cc[1]);
                        pc[2] = (Real) (cc[2]);

                        Real d_test = (pa-pc)*(pa-pc);

                        if(d_test<d)
                        {
                            ind2test.push_back(t_test);
                            indices.push_back(t_test);
                        }
                    }
                }
            }
        }

        ind2ask.clear();
        for (unsigned int t=0; t<ind2test.size(); t++)
        {
            ind2ask.push_back(ind2test[t]);
        }
    }

    return;
}

/// Compute intersection point with plane which is defined by c and normal
template <typename DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::getIntersectionPointWithPlane(const TetraID ind_ta, sofa::defaulttype::Vec<3,Real>& c, sofa::defaulttype::Vec<3,Real>& normal, sofa::helper::vector< sofa::defaulttype::Vec<3,Real> >& intersectedPoint, SeqEdges& intersectedEdge)
{
    const typename DataTypes::VecCoord& vect_c = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());
    const Tetrahedron ta=this->m_topology->getTetrahedron(ind_ta);
    const EdgesInTetrahedron edgesInTetra=this->m_topology->getEdgesInTetrahedron(ind_ta);
    const SeqEdges edges=this->m_topology->getEdges();

    sofa::defaulttype::Vec<3,Real> p1,p2;
    sofa::defaulttype::Vec<3,Real> intersection;

    //intersection with edge
    for(unsigned int i=0; i<edgesInTetra.size(); i++)
    {
        p1=vect_c[edges[edgesInTetra[i]][0]]; p2=vect_c[edges[edgesInTetra[i]][1]];
        if(computeIntersectionEdgeWithPlane(p1,p2,c,normal,intersection))
        {
            intersectedPoint.push_back(intersection);
            intersectedEdge.push_back(edges[edgesInTetra[i]]);
        }
    }

    static FILE* f1=fopen("tetra.txt","w");
    static FILE* f2=fopen("inter.txt","w");
    if(intersectedPoint.size()==3)
    {
        for(int i=0; i<4; i++)
        {
            p1=vect_c[ta[i]];
            fprintf(f1,"%d %f %f %f\n",ta[i],p1[0],p1[1],p1[2]);
        }
        for(unsigned int i=0; i<intersectedPoint.size(); i++)
        {
            fprintf(f2,"%f %f %f\n",intersectedPoint[i][0],intersectedPoint[i][1],intersectedPoint[i][2]);
        }
    }
}

template <typename DataTypes>
bool TetrahedronSetGeometryAlgorithms<DataTypes>::computeIntersectionEdgeWithPlane(sofa::defaulttype::Vec<3,Real>& p1,
                                                                                   sofa::defaulttype::Vec<3,Real>& p2,
                                                                                   sofa::defaulttype::Vec<3,Real>& c,
                                                                                   sofa::defaulttype::Vec<3,Real>& normal,
                                                                                   sofa::defaulttype::Vec<3,Real>& intersection)
{
    //plane equation
    normal.normalize();
    double d=normal*c;

    //line equation
    double t;

    //compute intersection
    t=(d-normal*p1)/(normal*(p2-p1));

    if((t<1)&&(t>0))
    {
        intersection=p1+(p2-p1)*t;
        return true;
    }
    else
        return false;
}

template <typename DataTypes>
bool TetrahedronSetGeometryAlgorithms<DataTypes>::checkNodeSequence(Tetra& tetra)
{
    const typename DataTypes::VecCoord& vect_c = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());
    sofa::defaulttype::Vec<3,Real> vec[3];
    for(int i=1; i<4; i++)
    {
        vec[i-1]=vect_c[tetra[i]]-vect_c[tetra[0]];
        vec[i-1].normalize();
    }
    Real dotProduct=(vec[1].cross(vec[0]))*vec[2];
    if(dotProduct<0)
        return true;
    else
        return false;
}

/// Write the current mesh into a msh file
template <typename DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::writeMSHfile(const char *filename) const
{
    std::ofstream myfile;
    myfile.open (filename);

    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::ConstVecCoordId::position())->getValue());

    const size_t numVertices = vect_c.size();

    myfile << "$NOD\n";
    myfile << numVertices <<"\n";

    for (size_t i=0; i<numVertices; ++i)
    {
        double x = (double) vect_c[i][0];
        double y = (double) vect_c[i][1];
        double z = (double) vect_c[i][2];

        myfile << i+1 << " " << x << " " << y << " " << z <<"\n";
    }

    myfile << "$ENDNOD\n";
    myfile << "$ELM\n";

    const sofa::helper::vector<Tetrahedron> &tea = this->m_topology->getTetrahedra();

    myfile << tea.size() <<"\n";

    for (unsigned int i=0; i<tea.size(); ++i)
    {
        myfile << i+1 << " 4 1 1 4 " << tea[i][0]+1 << " " << tea[i][1]+1 << " " << tea[i][2]+1 << " " << tea[i][3]+1 <<"\n";
    }

    myfile << "$ENDELM\n";

    myfile.close();
}




template<class DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    TriangleSetGeometryAlgorithms<DataTypes>::draw(vparams);

    const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
    //Draw tetra indices
    if (d_showTetrahedraIndices.getValue())
    {
        const sofa::defaulttype::Vec4f& color_tmp = d_drawColorTetrahedra.getValue();
        defaulttype::Vec4f color4(color_tmp[0] - 0.2f, color_tmp[1] - 0.2f, color_tmp[2] - 0.2f, 1.0);
        float scale = this->getIndicesScale();

        //for tetra:
        scale = scale/2;

        const sofa::helper::vector<Tetrahedron> &tetraArray = this->m_topology->getTetrahedra();

        helper::vector<defaulttype::Vector3> positions;
        for (unsigned int i =0; i<tetraArray.size(); i++)
        {

            Tetrahedron the_tetra = tetraArray[i];
            Coord vertex1 = coords[ the_tetra[0] ];
            Coord vertex2 = coords[ the_tetra[1] ];
            Coord vertex3 = coords[ the_tetra[2] ];
            Coord vertex4 = coords[ the_tetra[3] ];
            defaulttype::Vector3 center; center = (DataTypes::getCPos(vertex1)+DataTypes::getCPos(vertex2)+DataTypes::getCPos(vertex3)+DataTypes::getCPos(vertex4))/4;

            positions.push_back(center);

        }
        vparams->drawTool()->draw3DText_Indices(positions, scale, color4);
    }

    // Draw Tetra
    if (d_drawTetrahedra.getValue())
    {
        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, true);
        const sofa::defaulttype::Vec4f& color_tmp = d_drawColorTetrahedra.getValue();
        defaulttype::Vec4f color4(color_tmp[0] - 0.2f, color_tmp[1] - 0.2f, color_tmp[2] - 0.2f, 1.0);

        const sofa::helper::vector<Tetrahedron> &tetraArray = this->m_topology->getTetrahedra();
        std::vector<defaulttype::Vector3>   pos;
        pos.reserve(tetraArray.size()*4u);

        for (unsigned int i = 0; i<tetraArray.size(); ++i)
        {
            const Tetrahedron& tet = tetraArray[i];
            for (unsigned int j = 0u; j<4u; ++j)
            {
                pos.push_back(defaulttype::Vector3(DataTypes::getCPos(coords[tet[j]])));
            }
        }

        const float& scale = d_drawScaleTetrahedra.getValue();

        if (scale >= 1.0 && scale < 0.001)
            vparams->drawTool()->drawTetrahedra(pos, color4);
        else
            vparams->drawTool()->drawScaledTetrahedra(pos, color4, scale);

        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, false);
    }
}



} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TETEAHEDRONSETGEOMETRYALGORITHMS_INL
