/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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

    const unsigned int numVertices = (unsigned int)vect_c.size();

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
    Coord X; X[0]=0; X[1]=0; X[2]=0;

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

            for (unsigned int j = 0; j<3; j++)
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
#ifndef SOFA_NO_OPENGL
    PointSetGeometryAlgorithms<DataTypes>::draw(vparams);

    // Draw Edges indices
    if (showEdgeIndices.getValue())
    {
        sofa::defaulttype::Mat<4,4, GLfloat> modelviewM;
        const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
        const sofa::defaulttype::Vec3f& color = _drawColor.getValue();
        glColor3f(color[0], color[1], color[2]);
        glDisable(GL_LIGHTING);
        float scale = this->getIndicesScale();

        //for edges:
        scale = scale/2;

        const sofa::helper::vector <Edge>& edgeArray = this->m_topology->getEdges();

        for (unsigned int i =0; i<edgeArray.size(); i++)
        {

            Edge the_edge = edgeArray[i];
            Coord vertex1 = coords[ the_edge[0] ];
            Coord vertex2 = coords[ the_edge[1] ];
            sofa::defaulttype::Vec3f center; center = (DataTypes::getCPos(vertex1)+DataTypes::getCPos(vertex2))/2;

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


    // Draw edges
    if (_draw.getValue())
    {
        const sofa::helper::vector<Edge> &edgeArray = this->m_topology->getEdges();

        if (!edgeArray.empty())
        {
            glDisable(GL_LIGHTING);
            const sofa::defaulttype::Vec3f& color = _drawColor.getValue();
            glColor3f(color[0], color[1], color[2]);

            const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());

            glPointSize(4);
            for (unsigned int i = 0; i<edgeArray.size(); i++)
            {
                const Edge& e = edgeArray[i];
                sofa::defaulttype::Vec3f coordP1; coordP1 = DataTypes::getCPos(coords[e[0]]);
                sofa::defaulttype::Vec3f coordP2; coordP2 = DataTypes::getCPos(coords[e[1]]);
                glBegin(GL_LINES);
                glVertex3f(coordP1[0], coordP1[1], coordP1[2]);
                glVertex3f(coordP2[0], coordP2[1], coordP2[2]);
                glEnd();


                glBegin(GL_POINTS);
                glVertex3d(coordP1[0], coordP1[1], coordP1[2]);
                glVertex3d(coordP2[0], coordP2[1], coordP2[2]);
                glEnd();
            }
            glPointSize(1);
        }
    }

#endif /* SOFA_NO_OPENGL */
}



template< class DataTypes>
void EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights( sofa::helper::vector<unsigned>& numEdges,
                                                                           sofa::helper::vector<Edge>& vertexEdges,
                                                                           sofa::helper::vector<Vec3d>& weights ) const
{
    const VecCoord& pos =(this->object->read(core::ConstVecCoordId::position())->getValue()); // point positions

    sofa::helper::vector<Vector3> edgeVec;                  // 3D edges

    numEdges.clear();
    vertexEdges.clear();
    weights.clear();

    const SeqEdges& edges = this->m_topology->getEdges();

    for(unsigned pointId=0; pointId<pos.size(); pointId++ )
    {
        //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, point " << pointId << endl;
        EdgesAroundVertex ve = this->m_topology->getEdgesAroundVertex(pointId);
        edgeVec.resize(ve.size());
        numEdges.push_back(ve.size());            // number of edges attached to this point
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
            unsigned n = weights.size();     // start index for this vertex
            weights.resize( n + ve.size() ); // concatenate all the W of the nodes
            Vector3 a,u;

            // axis x
            a=Vector3(1,0,0);
            cholBksb(u,L,a); // solve EEt.u=x using the Cholesky decomposition
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, ux = " << u << endl;
            for(unsigned i=0; i<ve.size(); i++ )
            {
                weights[n+i][0] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to x = " << weights[n+i][0] << endl;
            }

            // axis y
            a=Vector3(0,1,0);
            cholBksb(u,L,a); // solve EEt.u=y using the Cholesky decomposition
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, uy = " << u << endl;
            for(unsigned i=0; i<ve.size(); i++ )
            {
                weights[n+i][1] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to y = " << weights[n+i][1] << endl;
            }

            // axis z
            a=Vector3(0,0,1);
            cholBksb(u,L,a); // solve EEt.u=z using the Cholesky decomposition
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, uz = " << u << endl;
            for(unsigned i=0; i<ve.size(); i++ )
            {
                weights[n+i][2] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to z = " << weights[n+i][2] << endl;
            }
        }
        else
        {
#ifdef SOFA_HAVE_EIGEN2   // use the SVD decomposition of Eigen
            unsigned n = weights.size();     // start index for this vertex
            weights.resize( n + ve.size() ); // concatenate all the W of the nodes
            Vector3 a,u;

            typedef Eigen::Matrix<SReal,3,3> EigenM33;
            EigenM33 emat = eigenMat(EEt);
//            Eigen::JacobiSVD<EigenM33> jacobi(emat, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::JacobiSVD<EigenM33> jacobi(emat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix<SReal,3,1> solution;

            // axis x
            a=Vector3(1,0,0);
            solution = jacobi.solve( eigenVec(a) );
            // least-squares solve EEt.u=x
            for(int i=0; i<3; i++)
                u[i] = solution(i);
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, ux = " << u << endl;
            for(unsigned i=0; i<ve.size(); i++ )
            {
                weights[n+i][0] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to x = " << weights[n+i][0] << endl;
            }

            // axis y
            a=Vector3(0,1,0);
            solution = jacobi.solve( eigenVec(a) );
            // least-squares solve EEt.u=y
            for(int i=0; i<3; i++)
                u[i] = solution(i);
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, uy = " << u << endl;
            for(unsigned i=0; i<ve.size(); i++ )
            {
                weights[n+i][1] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to y = " << weights[n+i][1] << endl;
            }

            // axis z
            a=Vector3(0,0,1);
            solution = jacobi.solve( eigenVec(a) );
            // least-squares solve EEt.u=z
            for(int i=0; i<3; i++)
                u[i] = solution(i);
            //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, uz = " << u << endl;
            for(unsigned i=0; i<ve.size(); i++ )
            {
                weights[n+i][2] = u * edgeVec[i];
                //cerr<<"EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, contribution of edge "<< i << " to z = " << weights[n+i][2] << endl;
            }

#else
            std::cerr << "EdgeSetGeometryAlgorithms< DataTypes >::computeLocalFrameEdgeWeights, cholDcmp failed, subsequent results are undefined " << std::endl;
#endif
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
