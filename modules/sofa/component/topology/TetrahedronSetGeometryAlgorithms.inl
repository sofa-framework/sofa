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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETGEOMETRYALGORITHMS_INL

#include <sofa/component/topology/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/topology/CommonAlgorithms.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;

template< class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::computeTetrahedronAABB(const TetraID i, Coord& minCoord, Coord& maxCoord) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p = *(this->object->getX());

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
    const typename DataTypes::VecCoord& p = *(this->object->getX());

    return (p[t[0]] + p[t[1]] + p[t[2]] + p[t[3]]) * (Real) 0.25;
}

template<class DataTypes>
typename DataTypes::Coord TetrahedronSetGeometryAlgorithms<DataTypes>::computeTetrahedronCircumcenter(const TetraID i) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p = *(this->object->getX());

    Coord center = p[t[0]];
    Coord t1 = p[t[1]] - p[t[0]];
    Coord t2 = p[t[2]] - p[t[0]];
    Coord t3 = p[t[3]] - p[t[0]];
    sofa::defaulttype::Vec<3,Real> a(t1[0], t1[1], t1[2]);
    sofa::defaulttype::Vec<3,Real> b(t2[0], t2[1], t2[2]);
    sofa::defaulttype::Vec<3,Real> c(t3[0], t3[1], t3[2]);

//		using namespace sofa::defaulttype;
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
    const typename DataTypes::VecCoord& p = *(this->object->getX());

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
    const typename DataTypes::VecCoord& p = *(this->object->getX());

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
            shapeFunctions[0] = v0/V;
            shapeFunctions[1] = v1/V;
            shapeFunctions[2] = v2/V;
            shapeFunctions[3] = v3/V;
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
    const typename DataTypes::VecCoord& p = *(this->object->getX());

    for(unsigned int i=0; i<4; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getRestTetrahedronVertexCoordinates(const TetraID i, Coord pnt[4]) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p = *(this->object->getX0());

    for(unsigned int i=0; i<4; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeTetrahedronVolume( const TetraID i) const
{
    const Tetrahedron t = this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p = *(this->object->getX());
    Real volume = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    if(volume<0)
        volume=-volume;
    return volume;
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeRestTetrahedronVolume( const TetraID i) const
{
    const Tetrahedron t=this->m_topology->getTetrahedron(i);
    const typename DataTypes::VecCoord& p = *(this->object->getX0());
    Real volume = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    if(volume<0)
        volume=-volume;
    return volume;
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeRestTetrahedronVolume( const Tetrahedron t) const
{
    const typename DataTypes::VecCoord& p = *(this->object->getX0());
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
    const typename DataTypes::VecCoord& p = *(this->object->getX());
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
    const typename DataTypes::VecCoord& vect_c = *(this->object->getX());

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
    const typename DataTypes::VecCoord& vect_c = *(this->object->getX());
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
            sofa::component::topology::TrianglesInTetrahedron adjacent_triangles = this->m_topology->getTrianglesInTetrahedron(ind_t);

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
//		const Tetrahedron &ta=this->m_topology->getTetrahedron(ind_ta);
    const typename DataTypes::VecCoord& vect_c = *(this->object->getX());

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
            sofa::component::topology::TrianglesInTetrahedron adjacent_triangles = this->m_topology->getTrianglesInTetrahedron(ind_t);

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
    const typename DataTypes::VecCoord& vect_c = *(this->object->getX0());
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
bool TetrahedronSetGeometryAlgorithms<DataTypes>::computeIntersectionEdgeWithPlane(Vec<3,Real>& p1, sofa::defaulttype::Vec<3,Real>& p2, sofa::defaulttype::Vec<3,Real>& c, sofa::defaulttype::Vec<3,Real>& normal, sofa::defaulttype::Vec<3,Real>& intersection)
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
    const typename DataTypes::VecCoord& vect_c = *(this->object->getX0());
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

    const typename DataTypes::VecCoord& vect_c = *(this->object->getX());

    const unsigned int numVertices = vect_c.size();

    myfile << "$NOD\n";
    myfile << numVertices <<"\n";

    for (unsigned int i=0; i<numVertices; ++i)
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

    //Draw tetra indices
    if (showTetrahedraIndices.getValue())
    {
        Mat<4,4, GLfloat> modelviewM;
        const VecCoord& coords = *(this->object->getX());
        const sofa::defaulttype::Vector3& color = _drawColor.getValue();
        glColor3f(color[0], color[1], color[2]);
        glDisable(GL_LIGHTING);
        float scale = this->getIndicesScale();

        //for tetra:
        scale = scale/2;

        const sofa::helper::vector<Tetrahedron> &tetraArray = this->m_topology->getTetrahedra();

        for (unsigned int i =0; i<tetraArray.size(); i++)
        {

            Tetrahedron the_tetra = tetraArray[i];
            Coord vertex1 = coords[ the_tetra[0] ];
            Coord vertex2 = coords[ the_tetra[1] ];
            Coord vertex3 = coords[ the_tetra[2] ];
            Coord vertex4 = coords[ the_tetra[3] ];
            sofa::defaulttype::Vec3f center; center = (DataTypes::getCPos(vertex1)+DataTypes::getCPos(vertex2)+DataTypes::getCPos(vertex3)+DataTypes::getCPos(vertex4))/4;

            std::ostringstream oss;
            oss << i;
            std::string tmp = oss.str();
            const char* s = tmp.c_str();
            glPushMatrix();

            glTranslatef(center[0], center[1], center[2]);
            glScalef(scale,scale,scale);

            // Makes text always face the viewer by removing the scene rotation
            // get the current modelview matrix
            glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
            modelviewM.transpose();

            sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

            //glLoadMatrixf(modelview);
            glLoadIdentity();

            glTranslatef(temp[0], temp[1], temp[2]);
            glScalef(scale,scale,scale);

            while(*s)
            {
                glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                s++;
            }

            glPopMatrix();

        }
    }


    // Draw Tetra
    if (_draw.getValue())
    {
        const sofa::helper::vector<Tetrahedron> &tetraArray = this->m_topology->getTetrahedra();

        if (!tetraArray.empty())
        {
            glDisable(GL_LIGHTING);
            const sofa::defaulttype::Vector3& color = _drawColor.getValue();
            glColor3f(color[0], color[1], color[2]);
            glBegin(GL_LINES);
            const VecCoord& coords = *(this->object->getX());

            for (unsigned int i = 0; i<tetraArray.size(); i++)
            {
                const Tetrahedron& tet = tetraArray[i];
                sofa::helper::vector <sofa::defaulttype::Vec3f> tetraCoord;

                for (unsigned int j = 0; j<4; j++)
                {
                    sofa::defaulttype::Vec3f p; p = DataTypes::getCPos(coords[tet[j]]);
                    tetraCoord.push_back(p);
                }

                for (unsigned int j = 0; j<4; j++)
                {
                    glVertex3f(tetraCoord[j][0], tetraCoord[j][1], tetraCoord[j][2]);
                    glVertex3f(tetraCoord[(j+1)%4][0], tetraCoord[(j+1)%4][1], tetraCoord[(j+1)%4][2]);
                }

                glVertex3f(tetraCoord[0][0], tetraCoord[0][1], tetraCoord[0][2]);
                glVertex3f(tetraCoord[2][0], tetraCoord[2][1], tetraCoord[2][2]);

                glVertex3f(tetraCoord[1][0], tetraCoord[1][1], tetraCoord[1][2]);
                glVertex3f(tetraCoord[3][0], tetraCoord[3][1], tetraCoord[3][2]);
            }
            glEnd();
        }
    }

}



} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TETEAHEDRONSETGEOMETRYALGORITHMS_INL
