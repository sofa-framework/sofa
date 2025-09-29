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
#include <sofa/component/topology/container/dynamic/TetrahedronSetGeometryAlgorithms.h>

#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyModifier.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/topology/container/dynamic/CommonAlgorithms.h>
#include <sofa/component/topology/container/dynamic/NumericalIntegrationDescriptor.inl>
#include <fstream>

namespace sofa::component::topology::container::dynamic
{

using sofa::core::objectmodel::ComponentState;
using namespace sofa::core::topology;

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
    sofa::Size i;
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
    Real c= (Real)-0.131555555555555556e-01;
    qpa.push_back(QuadraturePoint(v,(Real)c ));
    a=(Real)0.714285714285714285e-01;
    b=(Real)1-3*a;
    Real c1=(Real)0.762222222222222222e-02;
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
    c1= (Real)(.04574189830483037077884770618329337/6.0);
    qpa.push_back(QuadraturePoint(v,(Real)c1 ));
    // 4 points
    a=(Real).11425191803006935688146412277598412;
    b=1-3*a;
    c3=(Real)(0.01092727610912416907498417206565671/6.0);
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c3));
    }
    // 4 points
    a=(Real).01063790234539248531264164411274776;
    b=1-3*a;
    c3=(Real)(.00055352334192264689534558564012282/6.0);
    for (i=0;i<4;++i) {
        v=BarycentricCoordinatesType(a,a,a,a);
        v[i]=b;
        qpa.push_back(QuadraturePoint(v,c3));
    }
    // 4 points
    a=(Real).31274070833535645859816704980806110;
    b=1-3*a;
    c3=(Real)(.02569337913913269580782688316792080/6.0);
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
    d=(Real) (.01044842402938294329072628200105773/6.0);
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
    d=(Real) (.01178620679249594711782155323755017/6.0);
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
void TetrahedronSetGeometryAlgorithms< DataTypes >::init()
{
    TriangleSetGeometryAlgorithms<DataTypes>::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);
    if (m_container)
    {
        m_intialNbPoints = m_container->getNbPoints();
    }
    else
    {
        msg_error() << "No " << TetrahedronSetTopologyContainer::GetClass()->className << " can be found in the current context.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

template< class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::computeTetrahedronAABB(const TetraID i, Coord& minCoord, Coord& maxCoord) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());

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
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());

    return (p[t[0]] + p[t[1]] + p[t[2]] + p[t[3]]) * (Real) 0.25;
}

template<class DataTypes>
typename DataTypes::Coord TetrahedronSetGeometryAlgorithms<DataTypes>::computeTetrahedronCircumcenter(const TetraID i) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());

    Coord center = p[t[0]];
    Coord t1 = p[t[1]] - p[t[0]];
    Coord t2 = p[t[2]] - p[t[0]];
    Coord t3 = p[t[3]] - p[t[0]];
    sofa::type::Vec<3, Real> a, b, c;

    constexpr auto minDimensions = std::min<sofa::Size>(DataTypes::spatial_dimensions, 3u);
    std::copy_n(t1.begin(), minDimensions, a.begin());
    std::copy_n(t2.begin(), minDimensions, b.begin());
    std::copy_n(t3.begin(), minDimensions, c.begin());

    sofa::type::Vec<3,Real> d = (cross(b, c) * a.norm2() + cross(c, a) * b.norm2() + cross(a, b) * c.norm2()) / (12* computeTetrahedronVolume(i));

    center[0] += d[0];
    center[1] += d[1];
    center[2] += d[2];

    return center;
}

template< class DataTypes>
bool TetrahedronSetGeometryAlgorithms< DataTypes >::isPointInTetrahedron(const TetraID ind_t, const sofa::type::Vec<3,Real>& pTest) const
{
    const Real ZERO = 1e-15;

    const Tetrahedron t = this->m_topology->getTetrahedron(ind_t);
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());

    const sofa::type::Vec<3,Real> t0(p[t[0]][0], p[t[0]][1], p[t[0]][2]);
    const sofa::type::Vec<3,Real> t1(p[t[1]][0], p[t[1]][1], p[t[1]][2]);
    const sofa::type::Vec<3,Real> t2(p[t[2]][0], p[t[2]][1], p[t[2]][2]);
    const sofa::type::Vec<3,Real> t3(p[t[3]][0], p[t[3]][1], p[t[3]][2]);

    Real v0 = tripleProduct(t1-pTest, t2-pTest, t3-pTest);
    Real v1 = tripleProduct(pTest-t0, t2-t0, t3-t0);
    Real v2 = tripleProduct(t1-t0, pTest-t0, t3-t0);
    Real v3 = tripleProduct(t1-t0, t2-t0, pTest-t0);

    Real V = tripleProduct(t1-t0, t2-t0, t3-t0);
    if(fabs(V)>ZERO)
        return (v0/V > -ZERO) && (v1/V > -ZERO) && (v2/V > -ZERO) && (v3/V > -ZERO);

    else
        return false;
}

template< class DataTypes>
bool TetrahedronSetGeometryAlgorithms< DataTypes >::isPointInTetrahedron(const TetraID ind_t, const sofa::type::Vec<3,Real>& pTest, sofa::type::Vec<4,Real>& shapeFunctions) const
{
    const Real ZERO = 1e-15;

    const Tetrahedron t = this->m_topology->getTetrahedron(ind_t);
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());

    const sofa::type::Vec<3,Real> t0(p[t[0]][0], p[t[0]][1], p[t[0]][2]);
    const sofa::type::Vec<3,Real> t1(p[t[1]][0], p[t[1]][1], p[t[1]][2]);
    const sofa::type::Vec<3,Real> t2(p[t[2]][0], p[t[2]][1], p[t[2]][2]);
    const sofa::type::Vec<3,Real> t3(p[t[3]][0], p[t[3]][1], p[t[3]][2]);

    Real v0 = tripleProduct(t1-pTest, t2-pTest, t3-pTest);
    Real v1 = tripleProduct(pTest-t0, t2-t0, t3-t0);
    Real v2 = tripleProduct(t1-t0, pTest-t0, t3-t0);
    Real v3 = tripleProduct(t1-t0, t2-t0, pTest-t0);

    Real V = tripleProduct(t1-t0, t2-t0, t3-t0);
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
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());

    for(unsigned int i=0; i<4; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getRestTetrahedronVertexCoordinates(const TetraID i, Coord pnt[4]) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p = (this->object->read(core::vec_id::read_access::restPosition)->getValue());

    for(unsigned int i=0; i<4; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeTetrahedronVolume( const TetraID i) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());
    Real volume = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    if(volume<0)
        volume=-volume;
    return volume;
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeRestTetrahedronVolume( const TetraID i) const
{
    return computeRestTetrahedronVolume(this->m_topology->getTetrahedron(i));
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeRestTetrahedronVolume( const Tetrahedron& t) const
{
    const typename DataTypes::VecCoord& p = (this->object->read(core::vec_id::read_access::restPosition)->getValue());
    Real volume = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    if(volume<0)
        volume=-volume;
    return volume;
}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::computeTetrahedronVolume( BasicArrayInterface<Real> &ai) const
{
    const sofa::type::vector<Tetrahedron> &ta = this->m_topology->getTetrahedra();
    const typename DataTypes::VecCoord& p =(this->object->read(core::vec_id::read_access::position)->getValue());
    for (unsigned int i=0; i<ta.size(); ++i)
    {
        const Tetrahedron &t = ta[i];
        ai[i] = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    }
}

template<class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms<DataTypes>::computeDihedralAngle(const TetraID tetraId, const EdgeID edgeId) const
{
    Real angle = 0.0;
    const typename DataTypes::VecCoord& positions = (this->object->read(core::vec_id::read_access::position)->getValue());
    const Tetrahedron& tetra = this->m_topology->getTetrahedron(tetraId);
    const EdgesInTetrahedron& edgeIds = this->m_topology->getEdgesInTetrahedron(tetraId);
    const Edge& edge = this->m_topology->getEdge(edgeIds[edgeId]);

    unsigned int idA = edge[0];
    unsigned int idB = edge[1];

    unsigned int idC = sofa::InvalidID, idD = sofa::InvalidID;
    for (unsigned int i = 0; i < 4; i++)
    {
        if (tetra[i] != idA && tetra[i] != idB)
        {
            if (idC == sofa::InvalidID)
                idC = tetra[i];
            else
                idD = tetra[i];
        }
    }

    typename DataTypes::Coord pAB = positions[idB] - positions[idA];
    typename DataTypes::Coord pAC = positions[idC] - positions[idA];
    typename DataTypes::Coord pAD = positions[idD] - positions[idA];

    sofa::type::Vec<3, Real> AB = sofa::type::Vec<3, Real>(pAB);
    sofa::type::Vec<3, Real> AC = sofa::type::Vec<3, Real>(pAC);
    sofa::type::Vec<3, Real> AD = sofa::type::Vec<3, Real>(pAD);
    
    sofa::type::Vec<3, Real> nF1 = AB.cross(AC);
    sofa::type::Vec<3, Real> nF2 = AB.cross(AD);

    nF1.normalize();
    nF2.normalize();
    Real cosTheta = nF1 * nF2;
    angle = std::acos(cosTheta) * (180 / M_PI);    

    return angle;
}


/// Finds the indices of all tetrahedra in the ball of center ind_ta and of radius dist(ind_ta, ind_tb)
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getTetraInBall(const TetraID ind_ta, const TetraID ind_tb,
        sofa::type::vector<TetrahedronID> &indices) const
{
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

    const Tetrahedron ta=this->m_topology->getTetrahedron(ind_ta);
    const Tetrahedron tb=this->m_topology->getTetrahedron(ind_tb);

    const typename DataTypes::Coord& ca=(vect_c[ta[0]]+vect_c[ta[1]]+vect_c[ta[2]]+vect_c[ta[3]])*0.25;
    const typename DataTypes::Coord& cb=(vect_c[tb[0]]+vect_c[tb[1]]+vect_c[tb[2]]+vect_c[tb[3]])*0.25;

    sofa::type::Vec<3,Real> pa, pb;

    constexpr auto minDimensions = std::min<sofa::Size>(DataTypes::spatial_dimensions, 3u);
    std::copy_n(ca.begin(), minDimensions, pa.begin());
    std::copy_n(cb.begin(), minDimensions, pb.begin());

    Real d = (pa-pb)*(pa-pb);

    getTetraInBall(ind_ta, d, indices);
}

/// Finds the indices of all tetrahedra in the ball of center ind_ta and of radius dist(ind_ta, ind_tb)
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getTetraInBall(const TetraID ind_ta, Real r,
        sofa::type::vector<TetrahedronID> &indices) const
{
    Real d = r;
    const Tetrahedron ta=this->m_topology->getTetrahedron(ind_ta);
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());
    const typename DataTypes::Coord& ca=(vect_c[ta[0]]+vect_c[ta[1]]+vect_c[ta[2]]+vect_c[ta[3]])*0.25;

    sofa::type::Vec<3,Real> pa;
    std::copy_n(ca.begin(), std::min<sofa::Size>(DataTypes::spatial_dimensions, 3u), pa.begin());

    TetrahedronID t_test=ind_ta;
    indices.push_back(t_test);

    std::map<TetrahedronID, TetrahedronID> IndexMap;
    IndexMap.clear();
    IndexMap[t_test]=0;

    sofa::type::vector<TetrahedronID> ind2test;
    ind2test.push_back(t_test);
    sofa::type::vector<TetrahedronID> ind2ask;
    ind2ask.push_back(t_test);

    while(ind2test.size()>0)
    {
        ind2test.clear();
        for (size_t t=0; t<ind2ask.size(); t++)
        {
            TetrahedronID ind_t = ind2ask[t];
            core::topology::BaseMeshTopology::TrianglesInTetrahedron adjacent_triangles = this->m_topology->getTrianglesInTetrahedron(ind_t);

            for (sofa::Index i=0; i<adjacent_triangles.size(); i++)
            {
                const auto& tetrahedra_to_remove = this->m_topology->getTetrahedraAroundTriangle(adjacent_triangles[i]);

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

                    std::map<TetrahedronID, TetrahedronID>::iterator iter_1 = IndexMap.find(t_test);
                    if(iter_1 == IndexMap.end())
                    {
                        IndexMap[t_test]=0;

                        const Tetrahedron tc=this->m_topology->getTetrahedron(t_test);
                        const typename DataTypes::Coord& cc = (vect_c[tc[0]]
                                + vect_c[tc[1]]
                                + vect_c[tc[2]]
                                + vect_c[tc[3]]) * 0.25;
                        sofa::type::Vec<3,Real> pc;

                        std::copy_n(cc.begin(), std::min<sofa::Size>(DataTypes::spatial_dimensions, 3u), pc.begin());

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
        for (size_t t=0; t<ind2test.size(); t++)
        {
            ind2ask.push_back(ind2test[t]);
        }
    }

    return;
}

/// Finds the indices of all tetrahedra in the ball of center c and of radius r
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getTetraInBall(const Coord& c, Real r,
        sofa::type::vector<TetrahedronID> &indices) const
{
    TetraID ind_ta = sofa::InvalidID;
    sofa::type::Vec<3,Real> pa;
    pa[0] = (Real) (c[0]);
    pa[1] = (Real) (c[1]);
    pa[2] = (Real) (c[2]);
    for(sofa::Size i = 0; i < this->m_topology->getNbTetrahedra(); ++i)
    {
        if(isPointInTetrahedron(i, pa))
        {
            ind_ta = i;
            break;
        }
    }
    if(ind_ta == sofa::InvalidID)
        msg_error() << "getTetraInBall, Can't find the seed.";
    Real d = r;
//      const Tetrahedron &ta=this->m_topology->getTetrahedron(ind_ta);
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::vec_id::read_access::position)->getValue());

    TetrahedronID t_test=ind_ta;
    indices.push_back(t_test);

    std::map<TetrahedronID, TetrahedronID> IndexMap;
    IndexMap.clear();
    IndexMap[t_test]=0;

    sofa::type::vector<TetrahedronID> ind2test;
    ind2test.push_back(t_test);
    sofa::type::vector<TetrahedronID> ind2ask;
    ind2ask.push_back(t_test);

    while(ind2test.size()>0)
    {
        ind2test.clear();
        for (sofa::Index t=0; t<ind2ask.size(); t++)
        {
            TetrahedronID ind_t = ind2ask[t];
            core::topology::BaseMeshTopology::TrianglesInTetrahedron adjacent_triangles = this->m_topology->getTrianglesInTetrahedron(ind_t);

            for (sofa::Index i=0; i<adjacent_triangles.size(); i++)
            {
                sofa::type::vector< TetrahedronID > tetrahedra_to_remove = this->m_topology->getTetrahedraAroundTriangle(adjacent_triangles[i]);

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

                    std::map<TetrahedronID, TetrahedronID>::iterator iter_1 = IndexMap.find(t_test);
                    if(iter_1 == IndexMap.end())
                    {
                        IndexMap[t_test]=0;

                        const Tetrahedron tc=this->m_topology->getTetrahedron(t_test);
                        const typename DataTypes::Coord& cc = (vect_c[tc[0]]
                                + vect_c[tc[1]]
                                + vect_c[tc[2]]
                                + vect_c[tc[3]]) * 0.25;
                        sofa::type::Vec<3,Real> pc;
                        std::copy_n(cc.begin(), std::min<sofa::Size>(DataTypes::spatial_dimensions, 3u), pc.begin());

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
        for (size_t t=0; t<ind2test.size(); t++)
        {
            ind2ask.push_back(ind2test[t]);
        }
    }

    return;
}

/// Compute intersection point with plane which is defined by c and normal
template <typename DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::getIntersectionPointWithPlane(const TetraID ind_ta, const sofa::type::Vec<3,Real>& planP0, const sofa::type::Vec<3,Real>& normal, 
    sofa::type::vector< sofa::type::Vec<3,Real> >& intersectedPoint, SeqEdges& intersectedEdge)
{
    const typename DataTypes::VecCoord& vect_c = (this->object->read(core::vec_id::read_access::position)->getValue());
    const EdgesInTetrahedron& edgesInTetra = this->m_topology->getEdgesInTetrahedron(ind_ta);
    const SeqEdges& edges = this->m_topology->getEdges();

    sofa::type::Vec<3,Real> p1,p2;
    sofa::type::Vec<3,Real> intersection;

    //intersection with edge
    for(const auto edgeId : edgesInTetra)
    {
        const Edge& edge = edges[edgeId];
        p1 = toVecN<3,Real>(vect_c[edge[0]]);
        p2 = toVecN<3,Real>(vect_c[edge[1]]);

        if(computeIntersectionEdgeWithPlane(p1, p2, planP0, normal, intersection))
        {
            intersectedPoint.push_back(intersection);
            intersectedEdge.push_back(edge);
        }
    }
}

template <typename DataTypes>
bool TetrahedronSetGeometryAlgorithms<DataTypes>::computeIntersectionEdgeWithPlane(const sofa::type::Vec<3, Real>& edgeP1,
    const sofa::type::Vec<3, Real>& edgeP2,
    const sofa::type::Vec<3, Real>& planP0,
    const sofa::type::Vec<3, Real>& normal,
    sofa::type::Vec<3,Real>& intersection)
{
    //plane equation
    sofa::type::Vec<3, Real> planNorm = normal.normalized();
    Real d = planNorm * planP0;

    //compute intersection between line and plane equation
    Real t = (d - planNorm * edgeP1) / (planNorm*(edgeP2 - edgeP1));

    if((t<=1) && (t>=0))
    {
        intersection = edgeP1 + (edgeP2 - edgeP1)*t;
        return true;
    }
    else
        return false;
}

template< class DataTypes>
bool TetrahedronSetGeometryAlgorithms< DataTypes >::checkNodeSequence(const TetraID tetraId) const
{
    return checkNodeSequence(this->m_topology->getTetrahedron(tetraId));
}

template <typename DataTypes>
bool TetrahedronSetGeometryAlgorithms<DataTypes>::checkNodeSequence(const Tetrahedron& tetra) const
{
    const typename DataTypes::VecCoord& vect_c = (this->object->read(core::vec_id::read_access::position)->getValue());
    sofa::type::Vec<3,Real> vec[3];
    for(int i=1; i<4; i++)
    {
        vec[i-1]=type::toVec3(vect_c[tetra[i]]-vect_c[tetra[0]]);
        vec[i-1].normalize();
    }
    Real dotProduct=(vec[1].cross(vec[0]))*vec[2];
    if(dotProduct<0)
        return true;
    else
        return false;
}


template< class DataTypes>
bool TetrahedronSetGeometryAlgorithms< DataTypes >::isTetrahedronElongated(const TetraID tetraId, SReal factorLength) const
{
    const typename DataTypes::VecCoord& coords = (this->object->read(core::vec_id::read_access::position)->getValue());
    const Tetrahedron& tetra = this->m_topology->getTetrahedron(tetraId);    

    typename DataTypes::VecCoord points;
    points.resize(4);
    for (unsigned int i = 0; i < 4; i++) {
        points[i] = coords[ tetra[i] ];
    }

    Real minLength = std::numeric_limits<Real>::max();
    Real maxLength = std::numeric_limits<Real>::min();
    
    for (unsigned int i = 0; i < 4; i++)
    {
        for (unsigned int j = i + 1; j < 4; j++)
        {
            Real length = (points[j] - points[i]).norm2();
            if (length < minLength) {
                minLength = length;
            }
                
            if (length > maxLength) {
                maxLength = length;
            }            
        }
    }

    if (minLength*factorLength < maxLength) {
        return true;
    }
    else
        return false;
}


template< class DataTypes>
bool TetrahedronSetGeometryAlgorithms< DataTypes >::checkTetrahedronDihedralAngles(const TetraID tetraId, SReal minAngle, SReal maxAngle) const
{
    bool badAngle = false;
    for (unsigned int eId = 0; eId < 6; eId++)
    {
        Real angle = computeDihedralAngle(tetraId, eId);
        if (angle < minAngle) {
            badAngle = true;
            break;
        }
        else if (angle > maxAngle) {
            badAngle = true;
            break;
        }
    }

    return !badAngle;
}


template< class DataTypes>
bool TetrahedronSetGeometryAlgorithms< DataTypes >::checkTetrahedronValidity(const TetraID tetraId, SReal minAngle, SReal maxAngle, SReal factorLength) const
{
    // test orientation first
    if (checkNodeSequence(tetraId) == false) {
        return false;
    }

    // test elongated shape
    if (isTetrahedronElongated(tetraId, factorLength) == true) {
        return false;
    }

    // test dihedral angles
    if (checkTetrahedronDihedralAngles(tetraId, minAngle, maxAngle) == false)
    {
        return false;
    }

    return true;
}


template <typename DataTypes>
const sofa::type::vector<BaseMeshTopology::TetraID>& TetrahedronSetGeometryAlgorithms<DataTypes>::computeBadTetrahedron(SReal minAngle, SReal maxAngle, SReal factorLength)
{
    m_badTetraIds.clear();
    for (sofa::Index i = 0; i < this->m_topology->getNbTetrahedra(); ++i)
    {
        // test orientation first
        if (checkNodeSequence(i) == false)
        {
            m_badTetraIds.push_back(i);
            continue;
        }

        // test elongated shape
        if (isTetrahedronElongated(i, factorLength) == true)
        {
            m_badTetraIds.push_back(i);
            continue;
        }

        // test dihedral angles
        if (checkTetrahedronDihedralAngles(i, minAngle, maxAngle) == false)
        {
            m_badTetraIds.push_back(i);
            continue;
        }        
    }

    return m_badTetraIds;
}

template <typename DataTypes>
const sofa::type::vector<BaseMeshTopology::TetraID>& TetrahedronSetGeometryAlgorithms<DataTypes>::getBadTetrahedronIds()
{
    return m_badTetraIds;
}


template<class DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::subDivideTetrahedronsWithPlane(sofa::type::vector< sofa::type::vector<SReal> >& coefs, sofa::type::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal)
{
    //Current topological state
    const sofa::Size nbPoint=this->m_container->getNbPoints();
    const TetrahedronID nbTetra = (TetrahedronID)this->m_container->getNbTetrahedra();

    //Number of to be added points
    const sofa::Size nbTobeAddedPoints = sofa::Size(intersectedEdgeID.size()*2);

    //barycentric coordinates of to be added points
    sofa::type::vector< sofa::type::vector<PointID> > ancestors;
    for(sofa::Index i=0; i<intersectedEdgeID.size(); i++)
    {
        Edge theEdge=m_container->getEdge(intersectedEdgeID[i]);
        sofa::type::vector< EdgeID > ancestor;
        ancestor.push_back(theEdge[0]); ancestor.push_back(theEdge[1]);
        ancestors.push_back(ancestor); ancestors.push_back(ancestor);
    }

    //Number of to be added tetras
    int nbTobeAddedTetras=0;

    //To be added components
    sofa::type::vector<Tetra>			toBeAddedTetra;
    sofa::type::vector<TetraID>		toBeAddedTetraIndex;

    //To be removed components
    sofa::type::vector<TetraID>		toBeRemovedTetraIndex;

    sofa::type::vector<TetraID> intersectedTetras;
    sofa::type::vector<sofa::type::vector<EdgeID> > intersectedEdgesInTetra;
    int nbIntersectedTetras=0;

    //Getting intersected tetrahedron
    for( size_t i=0; i<intersectedEdgeID.size(); i++)
    {
        //Getting the tetrahedron around each intersected edge
        TetrahedraAroundEdge tetrasIdx=m_container->getTetrahedraAroundEdge(intersectedEdgeID[i]);
        for( size_t j=0; j<tetrasIdx.size(); j++)
        {
            bool flag=true;
            for( size_t k=0; k<intersectedTetras.size(); k++)
            {
                if(intersectedTetras[k]==tetrasIdx[j])
                {
                    flag=false;
                    intersectedEdgesInTetra[k].push_back(intersectedEdgeID[i]);
                    break;
                }
            }
            if(flag)
            {
                intersectedTetras.push_back(tetrasIdx[j]);
                nbIntersectedTetras++;
                intersectedEdgesInTetra.resize(nbIntersectedTetras);
                intersectedEdgesInTetra[nbIntersectedTetras-1].push_back(intersectedEdgeID[i]);
            }
        }
    }

    m_modifier->addPointsProcess(nbTobeAddedPoints);
    m_modifier->addPointsWarning(nbTobeAddedPoints, ancestors, coefs, true);

    //sub divide the each intersected tetrahedron
    for( size_t i=0; i<intersectedTetras.size(); i++)
    {
        //determine the index of intersected point
        sofa::type::vector<PointID> intersectedPointID;
        intersectedPointID.resize(intersectedEdgesInTetra[i].size());
        for( size_t j=0; j<intersectedEdgesInTetra[i].size(); j++)
        {
            for( size_t k=0; k<intersectedEdgeID.size(); k++)
            {
                if(intersectedEdgesInTetra[i][j]==intersectedEdgeID[k])
                {
                    intersectedPointID[j] = (PointID)(nbPoint+k*2);
                }
            }
        }
        nbTobeAddedTetras+=subDivideTetrahedronWithPlane(intersectedTetras[i],intersectedEdgesInTetra[i],intersectedPointID, planeNormal, toBeAddedTetra);

        //add the intersected tetrahedron to the to be removed tetrahedron list
        toBeRemovedTetraIndex.push_back(intersectedTetras[i]);
    }

    for(int i=0; i<nbTobeAddedTetras; i++)
        toBeAddedTetraIndex.push_back(nbTetra+i);

    //tetrahedron addition
    m_modifier->addTetrahedraProcess(toBeAddedTetra);
    m_modifier->addTetrahedraWarning(toBeAddedTetra.size(), (const sofa::type::vector< Tetra >&) toBeAddedTetra, toBeAddedTetraIndex);

    m_modifier->propagateTopologicalChanges();

    //tetrahedron removal
    m_modifier->removeTetrahedra(toBeRemovedTetraIndex);
    m_modifier->notifyEndingEvent();
    m_modifier->propagateTopologicalChanges();

    dmsg_info() << "NbCutElement=" << toBeRemovedTetraIndex.size() << " NbAddedElement=" << toBeAddedTetraIndex.size();
}

template<class DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::subDivideTetrahedronsWithPlane(sofa::type::vector<Coord>& intersectedPoints, sofa::type::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal)
{
    //Current topological state
    const sofa::Size nbPoint=this->m_container->getNbPoints();
    const TetrahedronID nbTetra = (TetrahedronID)this->m_container->getNbTetrahedra();

    //Number of to be added points
    const sofa::Size nbTobeAddedPoints = sofa::Size(intersectedEdgeID.size()*2);

    //barycentric coordinates of to be added points
    sofa::type::vector< sofa::type::vector<PointID> > ancestors;
    sofa::type::vector< sofa::type::vector<SReal> > coefs;
    for( size_t i=0; i<intersectedPoints.size(); i++)
    {
        Edge theEdge=m_container->getEdge(intersectedEdgeID[i]);
        sofa::type::Vec<3,Real> p;
        p[0]=intersectedPoints[i][0]; p[1]=intersectedPoints[i][1]; p[2]=intersectedPoints[i][2];
        sofa::type::vector< SReal > coef = this->computeEdgeBarycentricCoordinates(p, theEdge[0], theEdge[1]);

        sofa::type::vector< EdgeID > ancestor;
        ancestor.push_back(theEdge[0]); ancestor.push_back(theEdge[1]);

        ancestors.push_back(ancestor); ancestors.push_back(ancestor);
        coefs.push_back(coef); coefs.push_back(coef);
    }

    //Number of to be added tetras
    int nbTobeAddedTetras=0;

    //To be added components
    sofa::type::vector<Tetra>			toBeAddedTetra;
    sofa::type::vector<TetraID>		toBeAddedTetraIndex;

    //To be removed components
    sofa::type::vector<TetraID>		toBeRemovedTetraIndex;

    sofa::type::vector<TetraID> intersectedTetras;
    sofa::type::vector<sofa::type::vector<EdgeID> > intersectedEdgesInTetra;
    int nbIntersectedTetras=0;

    //Getting intersected tetrahedron
    for( size_t i=0; i<intersectedEdgeID.size(); i++)
    {
        //Getting the tetrahedron around each intersected edge
        TetrahedraAroundEdge tetrasIdx=m_container->getTetrahedraAroundEdge(intersectedEdgeID[i]);
        for( size_t j=0; j<tetrasIdx.size(); j++)
        {
            bool flag=true;
            for( size_t k=0; k<intersectedTetras.size(); k++)
            {
                if(intersectedTetras[k]==tetrasIdx[j])
                {
                    flag=false;
                    intersectedEdgesInTetra[k].push_back(intersectedEdgeID[i]);
                    break;
                }
            }
            if(flag)
            {
                intersectedTetras.push_back(tetrasIdx[j]);
                nbIntersectedTetras++;
                intersectedEdgesInTetra.resize(nbIntersectedTetras);
                intersectedEdgesInTetra[nbIntersectedTetras-1].push_back(intersectedEdgeID[i]);
            }
        }
    }

    m_modifier->addPointsProcess(nbTobeAddedPoints);
    m_modifier->addPointsWarning(nbTobeAddedPoints, ancestors, coefs, true);

    //sub divide the each intersected tetrahedron
    for( size_t i=0; i<intersectedTetras.size(); i++)
    {
        //determine the index of intersected point
        sofa::type::vector<PointID> intersectedPointID;
        intersectedPointID.resize(intersectedEdgesInTetra[i].size());
        for( size_t j=0; j<intersectedEdgesInTetra[i].size(); j++)
        {
            for( size_t k=0; k<intersectedEdgeID.size(); k++)
            {
                if(intersectedEdgesInTetra[i][j]==intersectedEdgeID[k])
                {
                    intersectedPointID[j]=(PointID)(nbPoint+k*2);
                }
            }
        }
        nbTobeAddedTetras+=subDivideTetrahedronWithPlane(intersectedTetras[i],intersectedEdgesInTetra[i],intersectedPointID, planeNormal, toBeAddedTetra);

        //add the intersected tetrahedron to the to be removed tetrahedron list
        toBeRemovedTetraIndex.push_back(intersectedTetras[i]);
    }

    for(int i=0; i<nbTobeAddedTetras; i++)
        toBeAddedTetraIndex.push_back(nbTetra+i);

    //tetrahedron addition
    m_modifier->addTetrahedraProcess(toBeAddedTetra);
    m_modifier->addTetrahedraWarning(toBeAddedTetra.size(), (const sofa::type::vector< Tetra >&) toBeAddedTetra, toBeAddedTetraIndex);

    m_modifier->propagateTopologicalChanges();

    //tetrahedron removal
    m_modifier->removeTetrahedra(toBeRemovedTetraIndex);
    m_modifier->notifyEndingEvent();
    m_modifier->propagateTopologicalChanges();

    dmsg_info() << "NbCutElement=" << toBeRemovedTetraIndex.size() << " NbAddedElement=" << toBeAddedTetraIndex.size();
}

template<class DataTypes>
int TetrahedronSetGeometryAlgorithms<DataTypes>::subDivideTetrahedronWithPlane(TetraID tetraIdx, sofa::type::vector<EdgeID>& intersectedEdgeID, sofa::type::vector<PointID>& intersectedPointID, Coord planeNormal, sofa::type::vector<Tetra>& toBeAddedTetra)
{
    Tetra intersectedTetra=this->m_container->getTetra(tetraIdx);
    int nbAddedTetra;
    Coord edgeDirec;

    //1. Number of intersected edge = 1
    if(intersectedEdgeID.size()==1)
    {
        Edge intersectedEdge=this->m_container->getEdge(intersectedEdgeID[0]);
        sofa::type::vector<PointID> pointsID;

        //find the point index of tetrahedron which are not included to the intersected edge
        for(int j=0; j<4; j++)
        {
            if(!(intersectedTetra[j]==intersectedEdge[0]))
            {
                if(!(intersectedTetra[j]==intersectedEdge[1]))
                    pointsID.push_back(intersectedTetra[j]);
            }
        }

        //construct subdivided tetrahedrons
        Tetra subTetra[2];
        edgeDirec=this->computeEdgeDirection(intersectedEdgeID[0])*-1;
        Real dot=edgeDirec*planeNormal;

        //inspect the tetrahedron is already subdivided
        if((pointsID[0]>=m_intialNbPoints) && (pointsID[1]>=m_intialNbPoints))
        {
            if((pointsID[0]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]+1;		subTetra[0][2]=pointsID[1]+1;		subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]+1;		subTetra[1][2]=pointsID[1]+1;		subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]-1;		subTetra[1][2]=pointsID[1]-1;			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]-1;		subTetra[0][2]=pointsID[1]-1;			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else if((pointsID[0]>=m_intialNbPoints))
        {
            if((pointsID[0]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]+1;		subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]+1;		subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]-1;		subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]-1;		subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else if((pointsID[1]>=m_intialNbPoints))
        {
            if((pointsID[1]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1]+1;		subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1]+1;		subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1]-1;			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1]-1;			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else
        {
            if(dot>0)
            {
                subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]; subTetra[0][2]=pointsID[1]; subTetra[0][3]=intersectedPointID[0]+1;
                subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]; subTetra[1][2]=pointsID[1]; subTetra[1][3]=intersectedPointID[0];
            }
            else
            {
                subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]; subTetra[0][2]=pointsID[1]; subTetra[0][3]=intersectedPointID[0];
                subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]; subTetra[1][2]=pointsID[1]; subTetra[1][3]=intersectedPointID[0]+1;
            }
        }

        //add the sub divided tetrahedra to the to be added tetrahedra list
        for(int i=0; i<2; i++)
        {
            if(!(checkNodeSequence(subTetra[i])))
            {
                TetrahedronID temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=2;
        return nbAddedTetra;
    }

    //2. Number of intersected edge = 2
    if(intersectedEdgeID.size()==2)
    {
        Edge intersectedEdge[2];
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            EdgeID temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }
        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);

        sofa::type::vector<PointID> pointsID;
        pointsID.resize(4);

        //find the point index which included both intersected edge
        if(intersectedEdge[0][0]==intersectedEdge[1][0])
        {
            pointsID[0]=intersectedEdge[0][0];
            pointsID[1]=intersectedEdge[0][1];
            pointsID[2]=intersectedEdge[1][1];
            edgeDirec=this->computeEdgeDirection(intersectedEdgeID[0])*-1;
        }
        else if(intersectedEdge[0][0]==intersectedEdge[1][1])
        {
            pointsID[0]=intersectedEdge[0][0];
            pointsID[1]=intersectedEdge[0][1];
            pointsID[2]=intersectedEdge[1][0];
            edgeDirec=this->computeEdgeDirection(intersectedEdgeID[0])*-1;
        }
        else if(intersectedEdge[0][1]==intersectedEdge[1][0])
        {
            pointsID[0]=intersectedEdge[0][1];
            pointsID[1]=intersectedEdge[0][0];
            pointsID[2]=intersectedEdge[1][1];
            edgeDirec=this->computeEdgeDirection(intersectedEdgeID[0]);
        }
        else
        {
            pointsID[0]=intersectedEdge[0][1];
            pointsID[1]=intersectedEdge[0][0];
            pointsID[2]=intersectedEdge[1][0];
            edgeDirec=this->computeEdgeDirection(intersectedEdgeID[0]);
        }

        //find the point index of tetrahedron which are not included to the intersected edge
        for(int i=0; i<4; i++)
        {
            bool flag=true;
            for(int j=0; j<3; j++)
            {
                if(intersectedTetra[i]==pointsID[j])
                {
                    flag=false;
                    break;
                }
            }
            if(flag)
            {
                pointsID[3]=intersectedTetra[i];
                break;
            }
        }

        //construct subdivided tetrahedrons
        Real dot=edgeDirec*planeNormal;
        Tetra subTetra[3];

        if(pointsID[3]>=m_intialNbPoints)
        {
            if((pointsID[3]-m_intialNbPoints)%2==0)//normal �ݴ�
            {
                if(dot>0)
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3]+1;
                    subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                    subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3];
                    subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3]+1;
                    subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3];
                    subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3]-1;
                    subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3]-1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3]-1;
                    subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                    subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
                }
            }
        }

        else
        {
            if(dot>0)
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3];
                subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
            }
            else
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3];
                subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
            }
        }

        //add the sub divided tetrahedra to the to be added tetrahedra list
        for(int i=0; i<3; i++)
        {
            if(!(checkNodeSequence(subTetra[i])))
            {
                TetrahedronID temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=3;
        return nbAddedTetra;
    }

    //3. Number of intersected edge = 3
    if(intersectedEdgeID.size()==3)
    {
        int DIVISION_STATE=0;			//1: COMPLETE DIVISION, 2: PARTIAL DIVISION
        Edge intersectedEdge[3];

        //sorting
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            int temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }
        if(intersectedEdgeID[1]>intersectedEdgeID[2])
        {
            int temp=intersectedEdgeID[1];
            intersectedEdgeID[1]=intersectedEdgeID[2];
            intersectedEdgeID[2]=temp;

            temp=intersectedPointID[1];
            intersectedPointID[1]=intersectedPointID[2];
            intersectedPointID[2]=temp;
        }
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            int temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }

        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);
        intersectedEdge[2]=this->m_container->getEdge(intersectedEdgeID[2]);

        sofa::type::vector<PointID> pointsID;
        pointsID.resize(4);

        for(int i=1; i<3; i++)
        {
            if(intersectedEdge[0][0]==intersectedEdge[i][0])
            {
                pointsID[0]=intersectedEdge[0][0];
                break;
            }
            if(intersectedEdge[0][0]==intersectedEdge[i][1])
            {
                pointsID[0]=intersectedEdge[0][0];
                break;
            }
            if(intersectedEdge[0][1]==intersectedEdge[i][0])
            {
                pointsID[0]=intersectedEdge[0][1];
                break;
            }
            if(intersectedEdge[0][1]==intersectedEdge[i][1])
            {
                pointsID[0]=intersectedEdge[0][1];
                break;
            }
        }

        //determine devision state
        int nbEdgeSharingPoint=0;
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<2; j++)
            {
                if(pointsID[0]==intersectedEdge[i][j])
                    nbEdgeSharingPoint++;
            }
        }
        if(nbEdgeSharingPoint==3)
            DIVISION_STATE=1;
        if(nbEdgeSharingPoint==2)
            DIVISION_STATE=2;

        //DIVISION STATE 1
        if(DIVISION_STATE==1)
        {
            if(pointsID[0]==intersectedEdge[0][0])
                edgeDirec=this->computeEdgeDirection(intersectedEdgeID[0])*-1;
            else
                edgeDirec=this->computeEdgeDirection(intersectedEdgeID[0]);
            for(int i=0; i<3; i++)
            {
                for(int j=0; j<2; j++)
                {
                    if(!(pointsID[0]==intersectedEdge[i][j]))
                        pointsID[i+1]=intersectedEdge[i][j];
                }
            }

            //construct subdivided tetrahedrons
            Real dot=edgeDirec*planeNormal;
            Tetra subTetra[4];
            if(dot>0)
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=intersectedPointID[2]+1;
                subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=intersectedPointID[2];	subTetra[1][3]=pointsID[1];
                subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=intersectedPointID[2];	subTetra[2][2]=pointsID[1];				subTetra[2][3]=pointsID[2];
                subTetra[3][0]=intersectedPointID[2];	subTetra[3][1]=pointsID[1];				subTetra[3][2]=pointsID[2];				subTetra[3][3]=pointsID[3];
            }
            else
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=intersectedPointID[2];
                subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1;	subTetra[1][2]=intersectedPointID[2]+1;	subTetra[1][3]=pointsID[1];
                subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=intersectedPointID[2]+1;	subTetra[2][2]=pointsID[1];				subTetra[2][3]=pointsID[2];
                subTetra[3][0]=intersectedPointID[2]+1;	subTetra[3][1]=pointsID[1]			;	subTetra[3][2]=pointsID[2];				subTetra[3][3]=pointsID[3];
            }

            //add the sub divided tetrahedra to the to be added tetrahedra list
            for(int i=0; i<4; i++)
            {
                if(!(checkNodeSequence(subTetra[i])))
                {
                    TetrahedronID temp=subTetra[i][1];
                    subTetra[i][1]=subTetra[i][2];
                    subTetra[i][2]=temp;
                }
                toBeAddedTetra.push_back(subTetra[i]);
            }
            nbAddedTetra=4;
            return nbAddedTetra;
        }

        //DIVISION STATE 2
        if(DIVISION_STATE==2)
        {
            Coord edgeDirec;
            if(pointsID[0]==intersectedEdge[0][0])
                edgeDirec=this->computeEdgeDirection(intersectedEdgeID[0])*-1;
            else
                edgeDirec=this->computeEdgeDirection(intersectedEdgeID[0]);

            int secondIntersectedEdgeIndex = 0, thirdIntersectedEdgeIndex = 0;
            int conectedIndex/*, nonConectedIndex*/;

            if(pointsID[0]==intersectedEdge[0][0])
                pointsID[1]=intersectedEdge[0][1];
            else
                pointsID[1]=intersectedEdge[0][0];

            if(pointsID[0]==intersectedEdge[1][0])
            {
                secondIntersectedEdgeIndex=1;
                thirdIntersectedEdgeIndex=2;
                pointsID[2]=intersectedEdge[1][1];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[2][0])
                    {
                        pointsID[3]=intersectedEdge[2][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[2][1])
                    {
                        pointsID[3]=intersectedEdge[2][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[1][1])
            {
                secondIntersectedEdgeIndex=1;
                thirdIntersectedEdgeIndex=2;
                pointsID[2]=intersectedEdge[1][0];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[2][0])
                    {
                        pointsID[3]=intersectedEdge[2][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[2][1])
                    {
                        pointsID[3]=intersectedEdge[2][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[2][0])
            {
                secondIntersectedEdgeIndex=2;
                thirdIntersectedEdgeIndex=1;
                pointsID[2]=intersectedEdge[2][1];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[1][0])
                    {
                        pointsID[3]=intersectedEdge[1][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[1][1])
                    {
                        pointsID[3]=intersectedEdge[1][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[2][1])
            {
                secondIntersectedEdgeIndex=2;
                thirdIntersectedEdgeIndex=1;
                pointsID[2]=intersectedEdge[2][0];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[1][0])
                    {
                        pointsID[3]=intersectedEdge[1][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[1][1])
                    {
                        pointsID[3]=intersectedEdge[1][0];
                        break;
                    }
                }
            }

            if(pointsID[3]==intersectedEdge[thirdIntersectedEdgeIndex][0])
            {
                if(pointsID[1]==intersectedEdge[thirdIntersectedEdgeIndex][1])
                {
                    conectedIndex=1;
                    //nonConectedIndex=2;
                }
                else
                {
                    conectedIndex=2;
                    //nonConectedIndex=1;
                }
            }
            else
            {
                if(pointsID[1]==intersectedEdge[thirdIntersectedEdgeIndex][0])
                {
                    conectedIndex=1;
                    //nonConectedIndex=2;
                }
                else
                {
                    conectedIndex=2;
                    //nonConectedIndex=1;
                }
            }

            //construct subdivided tetrahedrons
            Real dot=edgeDirec*planeNormal;
            Tetra subTetra[5];

            if(dot>0)
            {
                if(secondIntersectedEdgeIndex==1)
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0]+1;								subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                }
                else
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0]+1;								subTetra[0][3]=intersectedPointID[secondIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[1];				subTetra[2][1]=pointsID[3];				subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];				subTetra[3][1]=intersectedPointID[0];	subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];				subTetra[4][1]=pointsID[2];				subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                }
            }
            else
            {
                if(secondIntersectedEdgeIndex==1)
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0];								subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                }
                else
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0];								subTetra[0][3]=intersectedPointID[secondIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                }
            }

            //add the sub divided tetrahedra to the to be added tetrahedra list
            for(int i=0; i<5; i++)
            {
                if(!(checkNodeSequence(subTetra[i])))
                {
                    TetrahedronID temp=subTetra[i][1];
                    subTetra[i][1]=subTetra[i][2];
                    subTetra[i][2]=temp;
                }
                toBeAddedTetra.push_back(subTetra[i]);
            }
            nbAddedTetra=5;
            return nbAddedTetra;
            return 0;
        }
    }

    //Sub-division STATE2 : 4 edges are intersected by the plane
    if(intersectedEdgeID.size()==4)
    {
        Edge intersectedEdge[4];
        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);
        intersectedEdge[2]=this->m_container->getEdge(intersectedEdgeID[2]);
        intersectedEdge[3]=this->m_container->getEdge(intersectedEdgeID[3]);

        sofa::type::vector<PointID> pointsID;
        pointsID.resize(4);

        sofa::type::vector<PointID> localIndex;
        localIndex.resize(4);
        localIndex[0]=0;

        pointsID[0]=intersectedEdge[0][0]; pointsID[1]=intersectedEdge[0][1];
        for(int j=1; j<4; j++)
        {
            while(1)
            {
                if(intersectedEdge[0][0]==intersectedEdge[j][0])
                    break;
                else if(intersectedEdge[0][0]==intersectedEdge[j][1])
                    break;
                else if(intersectedEdge[0][1]==intersectedEdge[j][0])
                    break;
                else if(intersectedEdge[0][1]==intersectedEdge[j][1])
                    break;
                else
                {
                    pointsID[2]=intersectedEdge[j][0]; pointsID[3]=intersectedEdge[j][1];
                    break;
                }

            }
        }
        for(int j=1; j<4; j++)
        {
            if((intersectedEdge[j][0]==pointsID[0])||(intersectedEdge[j][1]==pointsID[0]))
            {
                if((intersectedEdge[j][0]==pointsID[3])||(intersectedEdge[j][1]==pointsID[3]))
                {
                    int temp=pointsID[3];
                    pointsID[3]=pointsID[2];
                    pointsID[2]=temp;
                }
            }
        }
        for(int j=1; j<4; j++)
        {
            if((intersectedEdge[j][0]==pointsID[0]) || (intersectedEdge[j][1]==pointsID[0]))
                localIndex[1]=j;
            else if((intersectedEdge[j][0]==pointsID[1]) || (intersectedEdge[j][1]==pointsID[1]))
                localIndex[3]=j;
            else
                localIndex[2]=j;
        }

        Coord edgeDirec;
        edgeDirec=this->computeEdgeDirection(intersectedEdgeID[0])*-1;

        //construct subdivided tetrahedrons
        Real dot=edgeDirec*planeNormal;
        Tetra subTetra[6];
        if(dot>0)
        {
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[3]])
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]]+1;	subTetra[0][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]]+1;	subTetra[0][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[0]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[3]]+1;	subTetra[0][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[2]]+1;	subTetra[0][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[2][0]=pointsID[0];			subTetra[2][1]=intersectedPointID[localIndex[0]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
            }
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[1]])
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[2]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]];	subTetra[4][2]=intersectedPointID[localIndex[1]];	subTetra[4][3]=intersectedPointID[localIndex[2]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]];	subTetra[5][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[3][0]=pointsID[2];		subTetra[3][1]=intersectedPointID[localIndex[1]];	subTetra[3][2]=intersectedPointID[localIndex[2]];	subTetra[3][3]=intersectedPointID[localIndex[3]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]];	subTetra[4][2]=intersectedPointID[localIndex[1]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]];	subTetra[5][3]=intersectedPointID[localIndex[3]];
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[1]];
                    subTetra[4][0]=pointsID[1];		subTetra[4][1]=intersectedPointID[localIndex[1]];	subTetra[4][2]=intersectedPointID[localIndex[2]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]];	subTetra[5][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[1]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[1]];	subTetra[4][2]=intersectedPointID[localIndex[2]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]];	subTetra[5][3]=intersectedPointID[localIndex[3]];
                }
            }
        }
        else
        {
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[3]])
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]];	subTetra[0][3]=intersectedPointID[localIndex[1]];
                    subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[3]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]];	subTetra[0][3]=intersectedPointID[localIndex[2]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[2]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[0]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[3]];	subTetra[0][3]=intersectedPointID[localIndex[1]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[3]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[2]];	subTetra[0][3]=intersectedPointID[localIndex[3]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[2]];
                    subTetra[2][0]=pointsID[0];			subTetra[2][1]=intersectedPointID[localIndex[0]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
            }
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[1]])
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]]+1;	subTetra[4][2]=intersectedPointID[localIndex[1]]+1;	subTetra[4][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]]+1;	subTetra[5][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[3][0]=pointsID[2];		subTetra[3][1]=intersectedPointID[localIndex[1]]+1;	subTetra[3][2]=intersectedPointID[localIndex[2]]+1;	subTetra[3][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]]+1;	subTetra[4][2]=intersectedPointID[localIndex[1]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]]+1;	subTetra[5][3]=intersectedPointID[localIndex[3]]+1;
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[4][0]=pointsID[1];		subTetra[4][1]=intersectedPointID[localIndex[1]]+1;	subTetra[4][2]=intersectedPointID[localIndex[2]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]]+1;	subTetra[5][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[1]]+1;	subTetra[4][2]=intersectedPointID[localIndex[2]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]]+1;	subTetra[5][3]=intersectedPointID[localIndex[3]]+1;
                }
            }
        }

        for(int i=0; i<6; i++)
        {
            if(!(checkNodeSequence(subTetra[i])))
            {
                TetrahedronID temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=6;
        return nbAddedTetra;
    }
    return 0;
}








































template<class DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::subDivideRestTetrahedronsWithPlane(sofa::type::vector< sofa::type::vector<SReal> >& coefs, sofa::type::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal)
{
    //Current topological state
    const sofa::Size nbPoint=this->m_container->getNbPoints();
    const TetrahedronID nbTetra = (TetrahedronID)this->m_container->getNbTetrahedra();

    //Number of to be added points
    const sofa::Size nbTobeAddedPoints = sofa::Size(intersectedEdgeID.size()*2);

    //barycentric coordinates of to be added points
    sofa::type::vector< sofa::type::vector<PointID> > ancestors;
    for( size_t i=0; i<intersectedEdgeID.size(); i++)
    {
        Edge theEdge=m_container->getEdge(intersectedEdgeID[i]);
        sofa::type::vector< EdgeID > ancestor;
        ancestor.push_back(theEdge[0]); ancestor.push_back(theEdge[1]);
        ancestors.push_back(ancestor); ancestors.push_back(ancestor);
    }

    //Number of to be added tetras
    sofa::Size nbTobeAddedTetras=0;

    //To be added components
    sofa::type::vector<Tetra>			toBeAddedTetra;
    sofa::type::vector<TetraID>		toBeAddedTetraIndex;

    //To be removed components
    sofa::type::vector<TetraID>		toBeRemovedTetraIndex;

    sofa::type::vector<TetraID> intersectedTetras;
    sofa::type::vector<sofa::type::vector<EdgeID> > intersectedEdgesInTetra;
    int nbIntersectedTetras=0;

    //Getting intersected tetrahedron
    for( size_t i=0; i<intersectedEdgeID.size(); i++)
    {
        //Getting the tetrahedron around each intersected edge
        TetrahedraAroundEdge tetrasIdx=m_container->getTetrahedraAroundEdge(intersectedEdgeID[i]);
        for( size_t j=0; j<tetrasIdx.size(); j++)
        {
            bool flag=true;
            for( size_t k=0; k<intersectedTetras.size(); k++)
            {
                if(intersectedTetras[k]==tetrasIdx[j])
                {
                    flag=false;
                    intersectedEdgesInTetra[k].push_back(intersectedEdgeID[i]);
                    break;
                }
            }
            if(flag)
            {
                intersectedTetras.push_back(tetrasIdx[j]);
                nbIntersectedTetras++;
                intersectedEdgesInTetra.resize(nbIntersectedTetras);
                intersectedEdgesInTetra[nbIntersectedTetras-1].push_back(intersectedEdgeID[i]);
            }
        }
    }

    m_modifier->addPointsProcess(nbTobeAddedPoints);
    m_modifier->addPointsWarning(nbTobeAddedPoints, ancestors, coefs, true);

    //sub divide the each intersected tetrahedron
    for( size_t i=0; i<intersectedTetras.size(); i++)
    {
        //determine the index of intersected point
        sofa::type::vector<PointID> intersectedPointID;
        intersectedPointID.resize(intersectedEdgesInTetra[i].size());
        for( size_t j=0; j<intersectedEdgesInTetra[i].size(); j++)
        {
            for( size_t k=0; k<intersectedEdgeID.size(); k++)
            {
                if(intersectedEdgesInTetra[i][j]==intersectedEdgeID[k])
                {
                    intersectedPointID[j]= (PointID)(nbPoint+k*2);
                }
            }
        }
        nbTobeAddedTetras+=subDivideRestTetrahedronWithPlane(intersectedTetras[i],intersectedEdgesInTetra[i],intersectedPointID, planeNormal, toBeAddedTetra);

        //add the intersected tetrahedron to the to be removed tetrahedron list
        toBeRemovedTetraIndex.push_back(intersectedTetras[i]);
    }

    for(sofa::Index i=0; i<nbTobeAddedTetras; i++)
        toBeAddedTetraIndex.push_back(nbTetra+i);

    //tetrahedron addition
    m_modifier->addTetrahedraProcess(toBeAddedTetra);
    m_modifier->addTetrahedraWarning(toBeAddedTetra.size(), (const sofa::type::vector< Tetra >&) toBeAddedTetra, toBeAddedTetraIndex);

    m_modifier->propagateTopologicalChanges();

    //tetrahedron removal
    m_modifier->removeTetrahedra(toBeRemovedTetraIndex);
    m_modifier->notifyEndingEvent();
    m_modifier->propagateTopologicalChanges();

    dmsg_info() << "NbCutElement=" << toBeRemovedTetraIndex.size() << " NbAddedElement=" << toBeAddedTetraIndex.size();
}

template<class DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::subDivideRestTetrahedronsWithPlane(sofa::type::vector<Coord>& intersectedPoints, sofa::type::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal)
{
    //Current topological state
    const sofa::Size nbPoint=this->m_container->getNbPoints();
    const TetrahedronID nbTetra = (TetrahedronID)this->m_container->getNbTetrahedra();

    //Number of to be added points
    const sofa::Size nbTobeAddedPoints = sofa::Size(intersectedEdgeID.size()*2);

    //barycentric coordinates of to be added points
    sofa::type::vector< sofa::type::vector<PointID> > ancestors;
    sofa::type::vector< sofa::type::vector<SReal> > coefs;
    for(sofa::Index i=0; i<intersectedPoints.size(); i++)
    {
        Edge theEdge=m_container->getEdge(intersectedEdgeID[i]);
        sofa::type::Vec<3,Real> p;
        p[0]=intersectedPoints[i][0]; p[1]=intersectedPoints[i][1]; p[2]=intersectedPoints[i][2];
        sofa::type::vector< SReal > coef = this->computeEdgeBarycentricCoordinates(p, theEdge[0], theEdge[1], true);

        sofa::type::vector< EdgeID > ancestor;
        ancestor.push_back(theEdge[0]); ancestor.push_back(theEdge[1]);

        ancestors.push_back(ancestor); ancestors.push_back(ancestor);
        coefs.push_back(coef); coefs.push_back(coef);
    }

    //Number of to be added tetras
    int nbTobeAddedTetras=0;

    //To be added components
    sofa::type::vector<Tetra>			toBeAddedTetra;
    sofa::type::vector<TetraID>		toBeAddedTetraIndex;

    //To be removed components
    sofa::type::vector<TetraID>		toBeRemovedTetraIndex;

    sofa::type::vector<TetraID> intersectedTetras;
    sofa::type::vector<sofa::type::vector<EdgeID> > intersectedEdgesInTetra;
    int nbIntersectedTetras=0;

    //Getting intersected tetrahedron
    for( size_t i=0; i<intersectedEdgeID.size(); i++)
    {
        //Getting the tetrahedron around each intersected edge
        TetrahedraAroundEdge tetrasIdx=m_container->getTetrahedraAroundEdge(intersectedEdgeID[i]);
        for( size_t j=0; j<tetrasIdx.size(); j++)
        {
            bool flag=true;
            for( size_t k=0; k<intersectedTetras.size(); k++)
            {
                if(intersectedTetras[k]==tetrasIdx[j])
                {
                    flag=false;
                    intersectedEdgesInTetra[k].push_back(intersectedEdgeID[i]);
                    break;
                }
            }
            if(flag)
            {
                intersectedTetras.push_back(tetrasIdx[j]);
                nbIntersectedTetras++;
                intersectedEdgesInTetra.resize(nbIntersectedTetras);
                intersectedEdgesInTetra[nbIntersectedTetras-1].push_back(intersectedEdgeID[i]);
            }
        }
    }

    m_modifier->addPointsProcess(nbTobeAddedPoints);
    m_modifier->addPointsWarning(nbTobeAddedPoints, ancestors, coefs, true);

    //sub divide the each intersected tetrahedron
    for( size_t i=0; i<intersectedTetras.size(); i++)
    {
        //determine the index of intersected point
        sofa::type::vector<PointID> intersectedPointID;
        intersectedPointID.resize(intersectedEdgesInTetra[i].size());
        for( size_t j=0; j<intersectedEdgesInTetra[i].size(); j++)
        {
            for( size_t k=0; k<intersectedEdgeID.size(); k++)
            {
                if(intersectedEdgesInTetra[i][j]==intersectedEdgeID[k])
                {
                    intersectedPointID[j]=(PointID)(nbPoint+k*2);
                }
            }
        }
        nbTobeAddedTetras+=subDivideRestTetrahedronWithPlane(intersectedTetras[i],intersectedEdgesInTetra[i],intersectedPointID, planeNormal, toBeAddedTetra);

        //add the intersected tetrahedron to the to be removed tetrahedron list
        toBeRemovedTetraIndex.push_back(intersectedTetras[i]);
    }

    for(int i=0; i<nbTobeAddedTetras; i++)
        toBeAddedTetraIndex.push_back(nbTetra+i);

    //tetrahedron addition
    m_modifier->addTetrahedraProcess(toBeAddedTetra);
    m_modifier->addTetrahedraWarning(toBeAddedTetra.size(), (const sofa::type::vector< Tetra >&) toBeAddedTetra, toBeAddedTetraIndex);

    m_modifier->propagateTopologicalChanges();

    //tetrahedron removal
    m_modifier->removeTetrahedra(toBeRemovedTetraIndex);
    m_modifier->notifyEndingEvent();
    m_modifier->propagateTopologicalChanges();

    dmsg_info() << "NbCutElement=" << toBeRemovedTetraIndex.size() << " NbAddedElement=" << toBeAddedTetraIndex.size();
}

template<class DataTypes>
int TetrahedronSetGeometryAlgorithms<DataTypes>::subDivideRestTetrahedronWithPlane(TetraID tetraIdx, sofa::type::vector<EdgeID>& intersectedEdgeID, sofa::type::vector<PointID>& intersectedPointID, Coord planeNormal, sofa::type::vector<Tetra>& toBeAddedTetra)
{
    Tetra intersectedTetra=this->m_container->getTetra(tetraIdx);
    int nbAddedTetra;
    Coord edgeDirec;

    //1. Number of intersected edge = 1
    if(intersectedEdgeID.size()==1)
    {
        Edge intersectedEdge=this->m_container->getEdge(intersectedEdgeID[0]);
        sofa::type::vector<PointID> pointsID;

        //find the point index of tetrahedron which are not included to the intersected edge
        for(int j=0; j<4; j++)
        {
            if(!(intersectedTetra[j]==intersectedEdge[0]))
            {
                if(!(intersectedTetra[j]==intersectedEdge[1]))
                    pointsID.push_back(intersectedTetra[j]);
            }
        }

        //construct subdivided tetrahedrons
        Tetra subTetra[2];
        edgeDirec=this->computeRestEdgeDirection(intersectedEdgeID[0])*-1;
        Real dot=edgeDirec*planeNormal;

        //inspect the tetrahedron is already subdivided
        if((pointsID[0]>=m_intialNbPoints) && (pointsID[1]>=m_intialNbPoints))
        {
            if((pointsID[0]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]+1;		subTetra[0][2]=pointsID[1]+1;		subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]+1;		subTetra[1][2]=pointsID[1]+1;		subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]-1;		subTetra[1][2]=pointsID[1]-1;			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]-1;		subTetra[0][2]=pointsID[1]-1;			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else if((pointsID[0]>=m_intialNbPoints))
        {
            if((pointsID[0]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]+1;		subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]+1;		subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]-1;		subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]-1;		subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else if((pointsID[1]>=m_intialNbPoints))
        {
            if((pointsID[1]-m_intialNbPoints)%2==0)
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1]+1;		subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1]+1;		subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1];				subTetra[0][3]=intersectedPointID[0]+1;
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1]-1;			subTetra[1][3]=intersectedPointID[0];
                }
                else
                {
                    subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0];			subTetra[0][2]=pointsID[1]-1;			subTetra[0][3]=intersectedPointID[0];
                    subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0];			subTetra[1][2]=pointsID[1];				subTetra[1][3]=intersectedPointID[0]+1;
                }
            }
        }

        else
        {
            if(dot>0)
            {
                subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]; subTetra[0][2]=pointsID[1]; subTetra[0][3]=intersectedPointID[0]+1;
                subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]; subTetra[1][2]=pointsID[1]; subTetra[1][3]=intersectedPointID[0];
            }
            else
            {
                subTetra[0][0]=intersectedEdge[0]; subTetra[0][1]=pointsID[0]; subTetra[0][2]=pointsID[1]; subTetra[0][3]=intersectedPointID[0];
                subTetra[1][0]=intersectedEdge[1]; subTetra[1][1]=pointsID[0]; subTetra[1][2]=pointsID[1]; subTetra[1][3]=intersectedPointID[0]+1;
            }
        }

        //add the sub divided tetrahedra to the to be added tetrahedra list
        for(int i=0; i<2; i++)
        {
            if(!(checkNodeSequence(subTetra[i])))
            {
                TetrahedronID temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=2;
        return nbAddedTetra;
    }

    //2. Number of intersected edge = 2
    if(intersectedEdgeID.size()==2)
    {
        Edge intersectedEdge[2];
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            int temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }
        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);

        sofa::type::vector<PointID> pointsID;
        pointsID.resize(4);

        //find the point index which included both intersected edge
        if(intersectedEdge[0][0]==intersectedEdge[1][0])
        {
            pointsID[0]=intersectedEdge[0][0];
            pointsID[1]=intersectedEdge[0][1];
            pointsID[2]=intersectedEdge[1][1];
            edgeDirec=this->computeRestEdgeDirection(intersectedEdgeID[0])*-1;
        }
        else if(intersectedEdge[0][0]==intersectedEdge[1][1])
        {
            pointsID[0]=intersectedEdge[0][0];
            pointsID[1]=intersectedEdge[0][1];
            pointsID[2]=intersectedEdge[1][0];
            edgeDirec=this->computeRestEdgeDirection(intersectedEdgeID[0])*-1;
        }
        else if(intersectedEdge[0][1]==intersectedEdge[1][0])
        {
            pointsID[0]=intersectedEdge[0][1];
            pointsID[1]=intersectedEdge[0][0];
            pointsID[2]=intersectedEdge[1][1];
            edgeDirec=this->computeRestEdgeDirection(intersectedEdgeID[0]);
        }
        else
        {
            pointsID[0]=intersectedEdge[0][1];
            pointsID[1]=intersectedEdge[0][0];
            pointsID[2]=intersectedEdge[1][0];
            edgeDirec=this->computeRestEdgeDirection(intersectedEdgeID[0]);
        }

        //find the point index of tetrahedron which are not included to the intersected edge
        for(int i=0; i<4; i++)
        {
            bool flag=true;
            for(int j=0; j<3; j++)
            {
                if(intersectedTetra[i]==pointsID[j])
                {
                    flag=false;
                    break;
                }
            }
            if(flag)
            {
                pointsID[3]=intersectedTetra[i];
                break;
            }
        }

        //construct subdivided tetrahedrons
        Real dot=edgeDirec*planeNormal;
        Tetra subTetra[3];

        if(pointsID[3]>=m_intialNbPoints)
        {
            if((pointsID[3]-m_intialNbPoints)%2==0)//normal �ݴ�
            {
                if(dot>0)
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3]+1;
                    subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                    subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3];
                    subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3]+1;
                    subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3]+1;
                }
            }
            else
            {
                if(dot>0)
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3];
                    subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3]-1;
                    subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3]-1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3]-1;
                    subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                    subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
                }
            }
        }

        else
        {
            if(dot>0)
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=pointsID[3];
                subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
            }
            else
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=pointsID[3];
                subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1; subTetra[1][2]=pointsID[1];				subTetra[1][3]=pointsID[3];
                subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=pointsID[1];				subTetra[2][2]=pointsID[2];				subTetra[2][3]=pointsID[3];
            }
        }

        //add the sub divided tetrahedra to the to be added tetrahedra list
        for(int i=0; i<3; i++)
        {
            if(!(checkNodeSequence(subTetra[i])))
            {
                TetrahedronID temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=3;
        return nbAddedTetra;
    }

    //3. Number of intersected edge = 3
    if(intersectedEdgeID.size()==3)
    {
        int DIVISION_STATE=0;			//1: COMPLETE DIVISION, 2: PARTIAL DIVISION
        Edge intersectedEdge[3];

        //sorting
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            int temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }
        if(intersectedEdgeID[1]>intersectedEdgeID[2])
        {
            int temp=intersectedEdgeID[1];
            intersectedEdgeID[1]=intersectedEdgeID[2];
            intersectedEdgeID[2]=temp;

            temp=intersectedPointID[1];
            intersectedPointID[1]=intersectedPointID[2];
            intersectedPointID[2]=temp;
        }
        if(intersectedEdgeID[0]>intersectedEdgeID[1])
        {
            int temp=intersectedEdgeID[0];
            intersectedEdgeID[0]=intersectedEdgeID[1];
            intersectedEdgeID[1]=temp;

            temp=intersectedPointID[0];
            intersectedPointID[0]=intersectedPointID[1];
            intersectedPointID[1]=temp;
        }

        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);
        intersectedEdge[2]=this->m_container->getEdge(intersectedEdgeID[2]);

        sofa::type::vector<PointID> pointsID;
        pointsID.resize(4);

        for(int i=1; i<3; i++)
        {
            if(intersectedEdge[0][0]==intersectedEdge[i][0])
            {
                pointsID[0]=intersectedEdge[0][0];
                break;
            }
            if(intersectedEdge[0][0]==intersectedEdge[i][1])
            {
                pointsID[0]=intersectedEdge[0][0];
                break;
            }
            if(intersectedEdge[0][1]==intersectedEdge[i][0])
            {
                pointsID[0]=intersectedEdge[0][1];
                break;
            }
            if(intersectedEdge[0][1]==intersectedEdge[i][1])
            {
                pointsID[0]=intersectedEdge[0][1];
                break;
            }
        }

        //determine devision state
        int nbEdgeSharingPoint=0;
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<2; j++)
            {
                if(pointsID[0]==intersectedEdge[i][j])
                    nbEdgeSharingPoint++;
            }
        }
        if(nbEdgeSharingPoint==3)
            DIVISION_STATE=1;
        if(nbEdgeSharingPoint==2)
            DIVISION_STATE=2;

        //DIVISION STATE 1
        if(DIVISION_STATE==1)
        {
            if(pointsID[0]==intersectedEdge[0][0])
                edgeDirec=this->computeRestEdgeDirection(intersectedEdgeID[0])*-1;
            else
                edgeDirec=this->computeRestEdgeDirection(intersectedEdgeID[0]);
            for(int i=0; i<3; i++)
            {
                for(int j=0; j<2; j++)
                {
                    if(!(pointsID[0]==intersectedEdge[i][j]))
                        pointsID[i+1]=intersectedEdge[i][j];
                }
            }

            //construct subdivided tetrahedrons
            Real dot=edgeDirec*planeNormal;
            Tetra subTetra[4];
            if(dot>0)
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0]+1;	subTetra[0][2]=intersectedPointID[1]+1;	subTetra[0][3]=intersectedPointID[2]+1;
                subTetra[1][0]=intersectedPointID[0];	subTetra[1][1]=intersectedPointID[1];	subTetra[1][2]=intersectedPointID[2];	subTetra[1][3]=pointsID[1];
                subTetra[2][0]=intersectedPointID[1];	subTetra[2][1]=intersectedPointID[2];	subTetra[2][2]=pointsID[1];				subTetra[2][3]=pointsID[2];
                subTetra[3][0]=intersectedPointID[2];	subTetra[3][1]=pointsID[1];				subTetra[3][2]=pointsID[2];				subTetra[3][3]=pointsID[3];
            }
            else
            {
                subTetra[0][0]=pointsID[0];				subTetra[0][1]=intersectedPointID[0];	subTetra[0][2]=intersectedPointID[1];	subTetra[0][3]=intersectedPointID[2];
                subTetra[1][0]=intersectedPointID[0]+1;	subTetra[1][1]=intersectedPointID[1]+1;	subTetra[1][2]=intersectedPointID[2]+1;	subTetra[1][3]=pointsID[1];
                subTetra[2][0]=intersectedPointID[1]+1;	subTetra[2][1]=intersectedPointID[2]+1;	subTetra[2][2]=pointsID[1];				subTetra[2][3]=pointsID[2];
                subTetra[3][0]=intersectedPointID[2]+1;	subTetra[3][1]=pointsID[1]			;	subTetra[3][2]=pointsID[2];				subTetra[3][3]=pointsID[3];
            }

            //add the sub divided tetrahedra to the to be added tetrahedra list
            for(int i=0; i<4; i++)
            {
                if(!(checkNodeSequence(subTetra[i])))
                {
                    TetrahedronID temp=subTetra[i][1];
                    subTetra[i][1]=subTetra[i][2];
                    subTetra[i][2]=temp;
                }
                toBeAddedTetra.push_back(subTetra[i]);
            }
            nbAddedTetra=4;
            return nbAddedTetra;
        }

        //DIVISION STATE 2
        if(DIVISION_STATE==2)
        {
            Coord edgeDirec;
            if(pointsID[0]==intersectedEdge[0][0])
                edgeDirec=this->computeRestEdgeDirection(intersectedEdgeID[0])*-1;
            else
                edgeDirec=this->computeRestEdgeDirection(intersectedEdgeID[0]);

            int secondIntersectedEdgeIndex = 0, thirdIntersectedEdgeIndex = 0;
            int conectedIndex/*, nonConectedIndex*/;

            if(pointsID[0]==intersectedEdge[0][0])
                pointsID[1]=intersectedEdge[0][1];
            else
                pointsID[1]=intersectedEdge[0][0];

            if(pointsID[0]==intersectedEdge[1][0])
            {
                secondIntersectedEdgeIndex=1;
                thirdIntersectedEdgeIndex=2;
                pointsID[2]=intersectedEdge[1][1];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[2][0])
                    {
                        pointsID[3]=intersectedEdge[2][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[2][1])
                    {
                        pointsID[3]=intersectedEdge[2][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[1][1])
            {
                secondIntersectedEdgeIndex=1;
                thirdIntersectedEdgeIndex=2;
                pointsID[2]=intersectedEdge[1][0];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[2][0])
                    {
                        pointsID[3]=intersectedEdge[2][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[2][1])
                    {
                        pointsID[3]=intersectedEdge[2][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[2][0])
            {
                secondIntersectedEdgeIndex=2;
                thirdIntersectedEdgeIndex=1;
                pointsID[2]=intersectedEdge[2][1];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[1][0])
                    {
                        pointsID[3]=intersectedEdge[1][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[1][1])
                    {
                        pointsID[3]=intersectedEdge[1][0];
                        break;
                    }
                }
            }

            if(pointsID[0]==intersectedEdge[2][1])
            {
                secondIntersectedEdgeIndex=2;
                thirdIntersectedEdgeIndex=1;
                pointsID[2]=intersectedEdge[2][0];
                for(int i=0; i<3; i++)
                {
                    if(pointsID[i]==intersectedEdge[1][0])
                    {
                        pointsID[3]=intersectedEdge[1][1];
                        break;
                    }
                    if(pointsID[i]==intersectedEdge[1][1])
                    {
                        pointsID[3]=intersectedEdge[1][0];
                        break;
                    }
                }
            }

            if(pointsID[3]==intersectedEdge[thirdIntersectedEdgeIndex][0])
            {
                if(pointsID[1]==intersectedEdge[thirdIntersectedEdgeIndex][1])
                {
                    conectedIndex=1;
                    //nonConectedIndex=2;
                }
                else
                {
                    conectedIndex=2;
                    //nonConectedIndex=1;
                }
            }
            else
            {
                if(pointsID[1]==intersectedEdge[thirdIntersectedEdgeIndex][0])
                {
                    conectedIndex=1;
                    //nonConectedIndex=2;
                }
                else
                {
                    conectedIndex=2;
                    //nonConectedIndex=1;
                }
            }

            //construct subdivided tetrahedrons
            Real dot=edgeDirec*planeNormal;
            Tetra subTetra[5];

            if(dot>0)
            {
                if(secondIntersectedEdgeIndex==1)
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0]+1;								subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                }
                else
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0]+1;								subTetra[0][3]=intersectedPointID[secondIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[1];				subTetra[2][1]=pointsID[3];				subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];				subTetra[3][1]=intersectedPointID[0];	subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];				subTetra[4][1]=pointsID[2];				subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0]+1;		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0];		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                    }
                }
            }
            else
            {
                if(secondIntersectedEdgeIndex==1)
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0];								subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                }
                else
                {
                    if(conectedIndex==2)
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[0];								subTetra[0][3]=intersectedPointID[secondIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[1];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[0];								subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                    else
                    {
                        subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];					subTetra[0][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[0][3]=intersectedPointID[thirdIntersectedEdgeIndex];
                        subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[0];		subTetra[1][2]=intersectedPointID[secondIntersectedEdgeIndex];		subTetra[1][3]=intersectedPointID[thirdIntersectedEdgeIndex];

                        subTetra[2][0]=pointsID[2];			subTetra[2][1]=pointsID[3];					subTetra[2][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[2][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[3][0]=pointsID[1];			subTetra[3][1]=intersectedPointID[0]+1;		subTetra[3][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[3][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                        subTetra[4][0]=pointsID[1];			subTetra[4][1]=pointsID[2];					subTetra[4][2]=intersectedPointID[secondIntersectedEdgeIndex]+1;	subTetra[4][3]=intersectedPointID[thirdIntersectedEdgeIndex]+1;
                    }
                }
            }

            //add the sub divided tetrahedra to the to be added tetrahedra list
            for(int i=0; i<5; i++)
            {
                if(!(checkNodeSequence(subTetra[i])))
                {
                    TetrahedronID temp=subTetra[i][1];
                    subTetra[i][1]=subTetra[i][2];
                    subTetra[i][2]=temp;
                }
                toBeAddedTetra.push_back(subTetra[i]);
            }
            nbAddedTetra=5;
            return nbAddedTetra;
            return 0;
        }
    }

    //Sub-division STATE2 : 4 edges are intersected by the plane
    if(intersectedEdgeID.size()==4)
    {
        Edge intersectedEdge[4];
        intersectedEdge[0]=this->m_container->getEdge(intersectedEdgeID[0]);
        intersectedEdge[1]=this->m_container->getEdge(intersectedEdgeID[1]);
        intersectedEdge[2]=this->m_container->getEdge(intersectedEdgeID[2]);
        intersectedEdge[3]=this->m_container->getEdge(intersectedEdgeID[3]);

        sofa::type::vector<PointID> pointsID;
        pointsID.resize(4);

        sofa::type::vector<PointID> localIndex;
        localIndex.resize(4);
        localIndex[0]=0;

        pointsID[0]=intersectedEdge[0][0]; pointsID[1]=intersectedEdge[0][1];
        for(int j=1; j<4; j++)
        {
            while(1)
            {
                if(intersectedEdge[0][0]==intersectedEdge[j][0])
                    break;
                else if(intersectedEdge[0][0]==intersectedEdge[j][1])
                    break;
                else if(intersectedEdge[0][1]==intersectedEdge[j][0])
                    break;
                else if(intersectedEdge[0][1]==intersectedEdge[j][1])
                    break;
                else
                {
                    pointsID[2]=intersectedEdge[j][0]; pointsID[3]=intersectedEdge[j][1];
                    break;
                }

            }
        }
        for(int j=1; j<4; j++)
        {
            if((intersectedEdge[j][0]==pointsID[0])||(intersectedEdge[j][1]==pointsID[0]))
            {
                if((intersectedEdge[j][0]==pointsID[3])||(intersectedEdge[j][1]==pointsID[3]))
                {
                    int temp=pointsID[3];
                    pointsID[3]=pointsID[2];
                    pointsID[2]=temp;
                }
            }
        }
        for(int j=1; j<4; j++)
        {
            if((intersectedEdge[j][0]==pointsID[0]) || (intersectedEdge[j][1]==pointsID[0]))
                localIndex[1]=j;
            else if((intersectedEdge[j][0]==pointsID[1]) || (intersectedEdge[j][1]==pointsID[1]))
                localIndex[3]=j;
            else
                localIndex[2]=j;
        }

        Coord edgeDirec;
        edgeDirec=this->computeRestEdgeDirection(intersectedEdgeID[0])*-1;

        //construct subdivided tetrahedrons
        Real dot=edgeDirec*planeNormal;
        Tetra subTetra[6];
        if(dot>0)
        {
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[3]])
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]]+1;	subTetra[0][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]]+1;	subTetra[0][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[0]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[3]]+1;	subTetra[0][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[2]]+1;	subTetra[0][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]]+1;	subTetra[1][2]=intersectedPointID[localIndex[1]]+1;	subTetra[1][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[2][0]=pointsID[0];			subTetra[2][1]=intersectedPointID[localIndex[0]]+1;	subTetra[2][2]=intersectedPointID[localIndex[3]]+1;	subTetra[2][3]=intersectedPointID[localIndex[2]]+1;
                }
            }
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[1]])
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[2]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]];	subTetra[4][2]=intersectedPointID[localIndex[1]];	subTetra[4][3]=intersectedPointID[localIndex[2]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]];	subTetra[5][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[3][0]=pointsID[2];		subTetra[3][1]=intersectedPointID[localIndex[1]];	subTetra[3][2]=intersectedPointID[localIndex[2]];	subTetra[3][3]=intersectedPointID[localIndex[3]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]];	subTetra[4][2]=intersectedPointID[localIndex[1]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]];	subTetra[5][3]=intersectedPointID[localIndex[3]];
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[1]];
                    subTetra[4][0]=pointsID[1];		subTetra[4][1]=intersectedPointID[localIndex[1]];	subTetra[4][2]=intersectedPointID[localIndex[2]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]];	subTetra[5][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]];	subTetra[3][2]=intersectedPointID[localIndex[3]];	subTetra[3][3]=intersectedPointID[localIndex[1]];
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[1]];	subTetra[4][2]=intersectedPointID[localIndex[2]];	subTetra[4][3]=intersectedPointID[localIndex[3]];
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]];	subTetra[5][3]=intersectedPointID[localIndex[3]];
                }
            }
        }
        else
        {
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[3]])
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]];	subTetra[0][3]=intersectedPointID[localIndex[1]];
                    subTetra[1][0]=pointsID[3];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[3]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[0]];	subTetra[0][3]=intersectedPointID[localIndex[2]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[2]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[0]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[1]]>intersectedEdgeID[localIndex[2]])
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[3]];	subTetra[0][3]=intersectedPointID[localIndex[1]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[3]];
                    subTetra[2][0]=pointsID[3];			subTetra[2][1]=intersectedPointID[localIndex[1]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
                else
                {
                    subTetra[0][0]=pointsID[0];			subTetra[0][1]=pointsID[3];							subTetra[0][2]=intersectedPointID[localIndex[2]];	subTetra[0][3]=intersectedPointID[localIndex[3]];
                    subTetra[1][0]=pointsID[0];			subTetra[1][1]=intersectedPointID[localIndex[0]];	subTetra[1][2]=intersectedPointID[localIndex[1]];	subTetra[1][3]=intersectedPointID[localIndex[2]];
                    subTetra[2][0]=pointsID[0];			subTetra[2][1]=intersectedPointID[localIndex[0]];	subTetra[2][2]=intersectedPointID[localIndex[3]];	subTetra[2][3]=intersectedPointID[localIndex[2]];
                }
            }
            if(intersectedEdgeID[localIndex[0]]>intersectedEdgeID[localIndex[1]])
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]]+1;	subTetra[4][2]=intersectedPointID[localIndex[1]]+1;	subTetra[4][3]=intersectedPointID[localIndex[2]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]]+1;	subTetra[5][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[3][0]=pointsID[2];		subTetra[3][1]=intersectedPointID[localIndex[1]]+1;	subTetra[3][2]=intersectedPointID[localIndex[2]]+1;	subTetra[3][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[0]]+1;	subTetra[4][2]=intersectedPointID[localIndex[1]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[0]]+1;	subTetra[5][3]=intersectedPointID[localIndex[3]]+1;
                }
            }
            else
            {
                if(intersectedEdgeID[localIndex[2]]>intersectedEdgeID[localIndex[3]])
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[4][0]=pointsID[1];		subTetra[4][1]=intersectedPointID[localIndex[1]]+1;	subTetra[4][2]=intersectedPointID[localIndex[2]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]]+1;	subTetra[5][3]=intersectedPointID[localIndex[2]]+1;
                }
                else
                {
                    subTetra[3][0]=pointsID[1];		subTetra[3][1]=intersectedPointID[localIndex[0]]+1;	subTetra[3][2]=intersectedPointID[localIndex[3]]+1;	subTetra[3][3]=intersectedPointID[localIndex[1]]+1;
                    subTetra[4][0]=pointsID[2];		subTetra[4][1]=intersectedPointID[localIndex[1]]+1;	subTetra[4][2]=intersectedPointID[localIndex[2]]+1;	subTetra[4][3]=intersectedPointID[localIndex[3]]+1;
                    subTetra[5][0]=pointsID[1];		subTetra[5][1]=pointsID[2];							subTetra[5][2]=intersectedPointID[localIndex[1]]+1;	subTetra[5][3]=intersectedPointID[localIndex[3]]+1;
                }
            }
        }

        for(int i=0; i<6; i++)
        {
            if(!(checkNodeSequence(subTetra[i])))
            {
                TetrahedronID temp=subTetra[i][1];
                subTetra[i][1]=subTetra[i][2];
                subTetra[i][2]=temp;
            }
            toBeAddedTetra.push_back(subTetra[i]);
        }
        nbAddedTetra=6;
        return nbAddedTetra;
    }
    return 0;
}

template <class DataTypes>
bool TetrahedronSetGeometryAlgorithms<DataTypes>::mustComputeBBox() const
{
    return ((d_showTetrahedraIndices.getValue() || d_drawTetrahedra.getValue()) && this->m_topology->getNbTetrahedra() != 0) || Inherit1::mustComputeBBox();
}


template<class DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(this->d_componentState.getValue() == ComponentState::Invalid)
        return;

    TriangleSetGeometryAlgorithms<DataTypes>::draw(vparams);

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    const VecCoord& coords =(this->object->read(core::vec_id::read_access::position)->getValue());
    //Draw tetra indices
    if (d_showTetrahedraIndices.getValue() && this->m_topology->getNbTetrahedra() != 0)
    {
        const auto& color_tmp = d_drawColorTetrahedra.getValue();
        const sofa::type::RGBAColor color4(color_tmp[0] - 0.2f, color_tmp[1] - 0.2f, color_tmp[2] - 0.2f, 1.0);
        float scale = this->getIndicesScale();

        //for tetra:
        scale = scale/2;

        const sofa::type::vector<Tetrahedron> &tetraArray = this->m_topology->getTetrahedra();

        std::vector<type::Vec3> positions;
        for (size_t i =0; i<tetraArray.size(); i++)
        {

            const Tetrahedron the_tetra = tetraArray[i];
            Coord vertex1 = coords[ the_tetra[0] ];
            Coord vertex2 = coords[ the_tetra[1] ];
            Coord vertex3 = coords[ the_tetra[2] ];
            Coord vertex4 = coords[ the_tetra[3] ];
            const type::Vec3 center = type::toVec3((DataTypes::getCPos(vertex1)+DataTypes::getCPos(vertex2)+DataTypes::getCPos(vertex3)+DataTypes::getCPos(vertex4))/4);

            positions.push_back(center);

        }
        vparams->drawTool()->draw3DText_Indices(positions, scale, color4);
    }

    // Draw Tetra
    if (d_drawTetrahedra.getValue() && this->m_topology->getNbTetrahedra() != 0)
    {
        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, true);

        const auto& color_tmp = d_drawColorTetrahedra.getValue();
        const sofa::type::RGBAColor color4(color_tmp[0] - 0.2f, color_tmp[1] - 0.2f, color_tmp[2] - 0.2f, 1.0);

        const sofa::type::vector<Tetrahedron> &tetraArray = this->m_topology->getTetrahedra();
        std::vector<type::Vec3>   pos;
        pos.reserve(tetraArray.size() * 4u);

        for (size_t i = 0; i < tetraArray.size(); ++i)
        {
            const Tetrahedron& tet = tetraArray[i];
            for (unsigned int j = 0u; j < 4u; ++j)
            {
                pos.push_back(type::toVec3(DataTypes::getCPos(coords[tet[j]])));
            }
        }

        const float& scale = d_drawScaleTetrahedra.getValue();
        
        if (scale >= 1.0 || scale < 0.001)
            vparams->drawTool()->drawTetrahedra(pos, color4);
        else
            vparams->drawTool()->drawScaledTetrahedra(pos, color4, scale);

        // Draw Tetra border
        if (!vparams->displayFlags().getShowWireFrame())
        {
            vparams->drawTool()->setPolygonMode(0, true);
            //vparams->drawTool()->enablePolygonOffset(0.0, -1.0);
            if (scale >= 1.0 || scale < 0.001)
                vparams->drawTool()->drawTetrahedra(pos, sofa::type::RGBAColor::gray());
            else
                vparams->drawTool()->drawScaledTetrahedra(pos, sofa::type::RGBAColor::gray(), scale);
            //vparams->drawTool()->disablePolygonOffset();
            vparams->drawTool()->setPolygonMode(0, false);
        }

        // Draw bad tetra
        if (!m_badTetraIds.empty())
        {
            std::vector<type::Vec3> posBad;
            posBad.reserve(m_badTetraIds.size() * 4u);

            for (size_t i = 0; i < m_badTetraIds.size(); ++i)
            {
                const Tetrahedron& tet = tetraArray[m_badTetraIds[i]];
                for (unsigned int j = 0u; j < 4u; ++j)
                {
                    posBad.push_back(type::toVec3(DataTypes::getCPos(coords[tet[j]])));
                }
            }

            const float& scale = d_drawScaleTetrahedra.getValue();

            if (scale >= 1.0 || scale < 0.001)
                vparams->drawTool()->drawTetrahedra(posBad, sofa::type::RGBAColor::red());
            else
                vparams->drawTool()->drawScaledTetrahedra(posBad, sofa::type::RGBAColor::red(), scale);
        }        
       
        if (vparams->displayFlags().getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0, false);
    }


}



} //namespace sofa::component::topology::container::dynamic
