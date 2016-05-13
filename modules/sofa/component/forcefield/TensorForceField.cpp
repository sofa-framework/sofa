/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/component/forcefield/TensorForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/gl.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/defaulttype/Vec3Types.h>

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace forcefield
{

//dunnno how it was defined, but it works anyway...
static unsigned int vertexEdge[4][4]= {{0,0,1,2,},{0,0,3,4},{1,3,0,5},{2,4,5,0}};



template <class DataTypes>
TensorForceField<DataTypes>::TensorForceField(const char* filename)
{
    load(filename);
    initialize();
}



template <class DataTypes>
TensorForceField<DataTypes>::TensorForceField(
    component::MechanicalObject<DataTypes>* object, const char* filename
)
    : object_(object), alpha_(0.0), lambda_(2.80e5), mu_(3.1e4)
{
    load(filename);
    initialize();
}



template<class DataTypes>
void
TensorForceField<DataTypes>::load(const char *filename)
{
    // opening the wrapping file
    std::ifstream in(filename);

    // read the young Modulues E and  the Poisson coefficient nu and compute the
    // lambda and mu lame coefficients from them
    // TODO : change this to E and nu.
    Real E, nu;
    in >> E >> nu;
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu_ = E / (2 * (1 + nu));

    // Read the damping factor alpha
    in >> alpha_;

    // read the name of the file containing the forcefield geometry
    std::string filename2;
    in >> filename2;
    // we have all we need, so close the wrapping file
    in.close();
    std::ifstream input(filename2.c_str());

    // read nb vertices and tetrahedra
    unsigned int nbVertices,nbTetrahedra;
    unsigned int dummyUInt;
    input >> nbTetrahedra >> nbVertices;
    // skips next 15 integers
    for (int i = 0; i < 15; ++i)
        input >> dummyUInt;

    // read tetrahedra
    unsigned int **vertexTetrahedronTable = new unsigned int * [nbTetrahedra];
    for (unsigned int i = 0; i < nbTetrahedra; ++i)
        vertexTetrahedronTable[i] = new unsigned int [4];

    for (unsigned int i = 0; i < nbTetrahedra; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            input >> vertexTetrahedronTable[i][j];
            // the file indices are 1-based, so we convert them to a 0-base
            --vertexTetrahedronTable[i][j];
        }
    }

    // read vertices
    Real px, py, pz;
    for (unsigned int i = 0; i < nbVertices; ++i)
    {
        input >> px >> py >> pz;
        vertex_.push_back( Coord(px, py, pz) );
    }

    // we have read all we needed, so close the geometry file
    input.close();

    // defining tetrahedra
    for (unsigned int i = 0; i < nbTetrahedra; ++i)
    {
        Tetrahedron tetra;

        tetra.vertex[0] = vertexTetrahedronTable[i][0];
        tetra.vertex[1] = vertexTetrahedronTable[i][1];
        tetra.vertex[2] = vertexTetrahedronTable[i][2];
        tetra.vertex[3] = vertexTetrahedronTable[i][3];

        // defining the edges of the current tetrahedron
        for (int i = 1; i < 4; ++i)
        {
            int v0 = tetra.vertex[i];
            for (int j = 0; j < i; ++j)
            {
                int v1 = tetra.vertex[j];
                int e = getEdge(v0, v1);
                tetra.edge[ vertexEdge[i][j] ] = e;
            }
        }

        //defining the triangles of the current tetrahedron
        for (int i = 0; i < 4; ++i)
        {
            int v0 = tetra.vertex[(i+1)%4];
            int v1 = tetra.vertex[(i+2)%4];
            int v2 = tetra.vertex[(i+3)%4];
            int tr = getTriangle(v0,v1,v2);
            tetra.triangle[i] = tr;
        }

        tetra.index = tetrahedron_.size();
        tetrahedron_.push_back(tetra);

    }

    // freeing allocated ressources
    for (unsigned int i = 0; i < nbTetrahedra; ++i)
        delete vertexTetrahedronTable[i];
    delete vertexTetrahedronTable;

    return;
}



template<class DataTypes>
void
TensorForceField<DataTypes>::initialize()
{

    // setting all tensors to null

    for (unsigned int i = 0; i < vertex_.size(); ++i)
    {
        vertexTensor_.push_back(VertexTensor());
        vertexTensor_[i].resetToNull();
    }

    for (unsigned int i = 0; i < edge_.size(); ++i)
    {
        edgeTensor_.push_back(EdgeTensor());
        edgeTensor_[i].resetToNull();
    }

    // computing rest volume and triangles shapeVectors for all tetrahedrons
    for (unsigned int i = 0; i < tetrahedron_.size(); ++i)
    {
        Real a[3] = {vertex_[ tetrahedron_[i].vertex[0] ][0],
                vertex_[ tetrahedron_[i].vertex[0] ][1],
                vertex_[ tetrahedron_[i].vertex[0] ][2]
                    };
        Real b[3] = {vertex_[ tetrahedron_[i].vertex[1] ][0],
                vertex_[ tetrahedron_[i].vertex[1] ][1],
                vertex_[ tetrahedron_[i].vertex[1] ][2]
                    };
        Real c[3] = {vertex_[ tetrahedron_[i].vertex[2] ][0],
                vertex_[ tetrahedron_[i].vertex[2] ][1],
                vertex_[ tetrahedron_[i].vertex[2] ][2]
                    };
        Real d[3] = {vertex_[ tetrahedron_[i].vertex[3] ][0],
                vertex_[ tetrahedron_[i].vertex[3] ][1],
                vertex_[ tetrahedron_[i].vertex[3] ][2]
                    };

        Real ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        Real ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        Real ca[3] = {a[0] - c[0], a[1] - c[1], a[2] - c[2]};
        Real ad[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
        Real bc[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
        Real cb[3] = {b[0] - c[0], b[1] - c[1], b[2] - c[2]};
        Real bd[3] = {d[0] - b[0], d[1] - b[1], d[2] - b[2]};

        // restVolume = dot(cross(ab, ac), ad) / 6.0
        tetrahedron_[i].restVolume = ((ab[1] * ac[2] - ab[2] * ac[1]) * ad[0] +
                (ab[2] * ac[0] - ab[0] * ac[2]) * ad[1] +
                (ab[0] * ac[1] - ab[1] * ac[0]) * ad[2]) /
                6.0f;

        // tetrahedron_[i].triangleShapeVector[0] = cross (bc, bd)
        tetrahedron_[i].triangleShapeVector[0][0] = bc[1] * bd[2] - bc[2] * bd[1];
        tetrahedron_[i].triangleShapeVector[0][1] = bc[2] * bd[0] - bc[0] * bd[2];
        tetrahedron_[i].triangleShapeVector[0][2] = bc[0] * bd[1] - bc[1] * bd[0];

        // tetrahedron_[i].triangleShapeVector[0] = cross (ad, ac)
        tetrahedron_[i].triangleShapeVector[1][0] = ad[1] * ac[2] - ad[2] * ac[1];
        tetrahedron_[i].triangleShapeVector[1][1] = ad[2] * ac[0] - ad[0] * ac[2];
        tetrahedron_[i].triangleShapeVector[1][2] = ad[0] * ac[1] - ad[1] * ac[0];

        // tetrahedron_[i].triangleShapeVector[0] = cross (ab, ad)
        tetrahedron_[i].triangleShapeVector[2][0] = ab[1] * ad[2] - ab[2] * ad[1];
        tetrahedron_[i].triangleShapeVector[2][1] = ab[2] * ad[0] - ab[0] * ad[2];
        tetrahedron_[i].triangleShapeVector[2][2] = ab[0] * ad[1] - ab[1] * ad[0];

        // tetrahedron_[i].triangleShapeVector[0] = cross (cb, ca)
        tetrahedron_[i].triangleShapeVector[0][0] = cb[1] * ca[2] - cb[2] * ca[1];
        tetrahedron_[i].triangleShapeVector[0][1] = cb[2] * ca[0] - cb[0] * ca[2];
        tetrahedron_[i].triangleShapeVector[0][2] = cb[0] * ca[1] - cb[1] * ca[0];


    }

    // initializing elastic tensors;
    for (unsigned int i = 0; i < tetrahedron_.size(); ++i)
        addElasticTensors( tetrahedron_[i] );

}



template<class DataTypes>
void
TensorForceField<DataTypes>::addForce ()
{
    // getting the containing mechanical object's data
    VecDeriv& f = *object_->getF();
    const VecCoord& p = *object_->getX();
    const VecDeriv& v = *object_->getV();

    f.resize( p.size() ); // ??really needed??

    // computing difference between current position and rest position
    sofa::helper::vector< Coord > pos;
    for (unsigned int i = 0; i < p.size(); ++i )
    {
        Coord dif = p[i] - vertex_[i];
        pos.push_back(dif);
    }


    // la force au point i est egale a la matrice de raideur (tenseur) au
    // point i * le deplacement du point i par rapport a sa position de repos plus
    // la somme des matrices de raideur sur chacune des arretes partant du point i
    // * le deplacement du point a l'autre bout de l'arrete.

    // Adding vertices' contribution.
    // TODO Check for sign correctness
    for (unsigned int i = 0; i < vertex_.size(); ++i)
    {
        f[i][0] -= vertexTensor_[i].tensor[0][0] * (pos[i][0] + alpha_ * v[i][0]) +
                vertexTensor_[i].tensor[0][1] * (pos[i][1] + alpha_ * v[i][1]) +
                vertexTensor_[i].tensor[0][2] * (pos[i][2] + alpha_ * v[i][2]);

        f[i][1] -= vertexTensor_[i].tensor[1][0] * (pos[i][0] + alpha_ * v[i][0]) +
                vertexTensor_[i].tensor[1][1] * (pos[i][1] + alpha_ * v[i][1]) +
                vertexTensor_[i].tensor[1][2] * (pos[i][2] + alpha_ * v[i][2]);

        f[i][2] -= vertexTensor_[i].tensor[2][0] * (pos[i][0] + alpha_ * v[i][0]) +
                vertexTensor_[i].tensor[2][1] * (pos[i][1] + alpha_ * v[i][1]) +
                vertexTensor_[i].tensor[2][2] * (pos[i][2] + alpha_ * v[i][2]);
    }

    // Adding edges' contribution.
    // TODO Check for sign correctness
    // TODO Check for transposition correctness
    for (unsigned int i = 0; i < edge_.size(); ++i)
    {
        int v0 = edge_[i].vertex[0];
        int v1 = edge_[i].vertex[1];

        f[v0][0] -= edgeTensor_[i].tensor[0][0] * (pos[v1][0] + alpha_ * v[v1][0]) +
                edgeTensor_[i].tensor[1][0] * (pos[v1][1] + alpha_ * v[v1][1]) +
                edgeTensor_[i].tensor[2][0] * (pos[v1][2] + alpha_ * v[v1][2]);

        f[v0][1] -= edgeTensor_[i].tensor[0][1] * (pos[v1][0] + alpha_ * v[v1][0]) +
                edgeTensor_[i].tensor[1][1] * (pos[v1][1] + alpha_ * v[v1][1]) +
                edgeTensor_[i].tensor[2][1] * (pos[v1][2] + alpha_ * v[v1][2]);

        f[v0][2] -= edgeTensor_[i].tensor[0][2] * (pos[v1][0] + alpha_ * v[v1][0]) +
                edgeTensor_[i].tensor[1][2] * (pos[v1][1] + alpha_ * v[v1][1]) +
                edgeTensor_[i].tensor[2][2] * (pos[v1][2] + alpha_ * v[v1][2]);

        f[v1][0] -= edgeTensor_[i].tensor[0][0] * (pos[v0][0] + alpha_ * v[v0][0]) +
                edgeTensor_[i].tensor[0][1] * (pos[v0][1] + alpha_ * v[v0][1]) +
                edgeTensor_[i].tensor[0][2] * (pos[v0][2] + alpha_ * v[v0][2]);

        f[v1][1] -= edgeTensor_[i].tensor[1][0] * (pos[v0][0] + alpha_ * v[v0][0]) +
                edgeTensor_[i].tensor[1][1] * (pos[v0][1] + alpha_ * v[v0][1]) +
                edgeTensor_[i].tensor[1][2] * (pos[v0][2] + alpha_ * v[v0][2]);

        f[v1][2] -= edgeTensor_[i].tensor[2][0] * (pos[v0][0] + alpha_ * v[v0][0]) +
                edgeTensor_[i].tensor[2][1] * (pos[v0][1] + alpha_ * v[v0][1]) +
                edgeTensor_[i].tensor[2][2] * (pos[v0][2] + alpha_ * v[v0][2]);

    }

}



template<class DataTypes>
void
TensorForceField<DataTypes>::addDForce()
{
    // getting the containing mechanical object's data
    VecDeriv& f = *object_->getF();
    // use Dx instead of X
    const VecCoord& p = *object_->getDx();
    const VecDeriv& v = *object_->getV();

    f.resize( p.size() ); // ??really needed??

    // computing difference between current position and rest position
    /*sofa::helper::vector< Coord > pos;
    for (unsigned int i = 0; i < p.size(); ++i ) {
      Coord dif = p[i] - vertex_[i];
      pos.push_back(dif);
    }*/


    // la force au point i est egale a la matrice de raideur (tenseur) au
    // point i * le deplacement du point i par rapport a sa position de repos plus
    // la somme des matrices de raideur sur chacune des arretes partant du point i
    // * le deplacement du point a l'autre bout de l'arrete.

    // Adding vertices' contribution.
    // TODO Check for sign correctness
    for (unsigned int i = 0; i < vertex_.size(); ++i)
    {
        f[i][0] -= vertexTensor_[i].tensor[0][0] * (p[i][0] + alpha_ * v[i][0]) +
                vertexTensor_[i].tensor[0][1] * (p[i][1] + alpha_ * v[i][1]) +
                vertexTensor_[i].tensor[0][2] * (p[i][2] + alpha_ * v[i][2]);

        f[i][1] -= vertexTensor_[i].tensor[1][0] * (p[i][0] + alpha_ * v[i][0]) +
                vertexTensor_[i].tensor[1][1] * (p[i][1] + alpha_ * v[i][1]) +
                vertexTensor_[i].tensor[1][2] * (p[i][2] + alpha_ * v[i][2]);

        f[i][2] -= vertexTensor_[i].tensor[2][0] * (p[i][0] + alpha_ * v[i][0]) +
                vertexTensor_[i].tensor[2][1] * (p[i][1] + alpha_ * v[i][1]) +
                vertexTensor_[i].tensor[2][2] * (p[i][2] + alpha_ * v[i][2]);
    }

    // Adding edges' contribution.
    // TODO Check for sign correctness
    // TODO Check for transposition correctness
    for (unsigned int i = 0; i < edge_.size(); ++i)
    {
        int v0 = edge_[i].vertex[0];
        int v1 = edge_[i].vertex[1];

        f[v0][0] -= edgeTensor_[i].tensor[0][0] * (p[v1][0] + alpha_ * v[v1][0]) +
                edgeTensor_[i].tensor[1][0] * (p[v1][1] + alpha_ * v[v1][1]) +
                edgeTensor_[i].tensor[2][0] * (p[v1][2] + alpha_ * v[v1][2]);

        f[v0][1] -= edgeTensor_[i].tensor[0][1] * (p[v1][0] + alpha_ * v[v1][0]) +
                edgeTensor_[i].tensor[1][1] * (p[v1][1] + alpha_ * v[v1][1]) +
                edgeTensor_[i].tensor[2][1] * (p[v1][2] + alpha_ * v[v1][2]);

        f[v0][2] -= edgeTensor_[i].tensor[0][2] * (p[v1][0] + alpha_ * v[v1][0]) +
                edgeTensor_[i].tensor[1][2] * (p[v1][1] + alpha_ * v[v1][1]) +
                edgeTensor_[i].tensor[2][2] * (p[v1][2] + alpha_ * v[v1][2]);

        f[v1][0] -= edgeTensor_[i].tensor[0][0] * (p[v0][0] + alpha_ * v[v0][0]) +
                edgeTensor_[i].tensor[0][1] * (p[v0][1] + alpha_ * v[v0][1]) +
                edgeTensor_[i].tensor[0][2] * (p[v0][2] + alpha_ * v[v0][2]);

        f[v1][1] -= edgeTensor_[i].tensor[1][0] * (p[v0][0] + alpha_ * v[v0][0]) +
                edgeTensor_[i].tensor[1][1] * (p[v0][1] + alpha_ * v[v0][1]) +
                edgeTensor_[i].tensor[1][2] * (p[v0][2] + alpha_ * v[v0][2]);

        f[v1][2] -= edgeTensor_[i].tensor[2][0] * (p[v0][0] + alpha_ * v[v0][0]) +
                edgeTensor_[i].tensor[2][1] * (p[v0][1] + alpha_ * v[v0][1]) +
                edgeTensor_[i].tensor[2][2] * (p[v0][2] + alpha_ * v[v0][2]);

    }

}

template <class DataTypes>
double TensorForceField<DataTypes>::getPotentialEnergy()
{
    cerr<<"TensorForceField::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}


template<class DataTypes>
void TensorForceField<DataTypes>::draw()
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    const VecCoord& p1 = *object_->getX();
    glDisable(GL_LIGHTING);
    glColor4f(1,1,1,1);
    glBegin(GL_LINES);
    for (unsigned int i = 0; i < edge_.size(); ++i)
    {
        glVertex3d(p1[edge_[i].vertex[0]][0],
                p1[edge_[i].vertex[0]][1],
                p1[edge_[i].vertex[0]][2]);

        glVertex3d(p1[edge_[i].vertex[1]][0],
                p1[edge_[i].vertex[1]][1],
                p1[edge_[i].vertex[1]][2]);
    }
    glEnd();
}


template<class DataTypes>
void TensorForceField<DataTypes>::initTextures()
{

}


template<class DataTypes>
void TensorForceField<DataTypes>::update()
{

}


// search for the edge connecting the given vertices, create it if not found
template<class DataTypes>
int
TensorForceField<DataTypes>::getEdge(const int v0, const int v1)
{
    for (unsigned int i = 0; i < edge_.size(); ++i)
    {
        if (
            (edge_[i].vertex[0] == v0 && edge_[i].vertex[1] == v1) ||
            (edge_[i].vertex[0] == v1 && edge_[i].vertex[1] == v0)
        )
            return i;
    }

    // edge wasn't found, we have to create it
    Edge e;

    e.index = edge_.size();

    e.vertex[0] = v0;
    e.vertex[1] = v1;

    edge_.push_back(e);

    return e.index;
}



// search for the triangle connecting the given vertices, create it if not found
template<class DataTypes>
int
TensorForceField<DataTypes>::getTriangle(const int v0, const int v1,
        const int v2)
{
    for (unsigned int i = 0; i < triangle_.size(); ++i)
    {
        int tv0 = triangle_[i].vertex[0];
        int tv1 = triangle_[i].vertex[1];
        int tv2 = triangle_[i].vertex[2];

        if (
            (tv0 == v0 && tv1 == v1 && tv2 == v2) ||
            (tv0 == v0 && tv1 == v2 && tv2 == v1) ||
            (tv0 == v1 && tv1 == v0 && tv2 == v2) ||
            (tv0 == v1 && tv1 == v2 && tv2 == v0) ||
            (tv0 == v2 && tv1 == v0 && tv2 == v1) ||
            (tv0 == v2 && tv1 == v1 && tv2 == v0)
        )
            return i;
    }

    // triangle wasn't found, we have to create it
    Triangle t;

    t.index = triangle_.size();

    t.vertex[0] = v0;
    t.vertex[1] = v1;
    t.vertex[2] = v2;

    triangle_.push_back(t);

    return t.index;

}



// add the elastic tensors for the given tetrahedron
template<class DataTypes>
void TensorForceField<DataTypes>::addElasticTensors(Tetrahedron& tetra)
{
    Real si[3];
    Real sj[3];

    Real t[3][3];
    Real id[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    for (int i = 0; i < 4; ++i)
    {
        for (int j = i; j < 4; ++j)
        {
            // si is the shape vector of triangle[i]
            si[0] = tetra.triangleShapeVector[i][0];
            si[1] = tetra.triangleShapeVector[i][1];
            si[2] = tetra.triangleShapeVector[i][2];
            // sj is the shape vector of triangle[j]
            sj[0] = tetra.triangleShapeVector[j][0];
            sj[1] = tetra.triangleShapeVector[j][1];
            sj[2] = tetra.triangleShapeVector[j][2];

            Real dot = (si[0] * sj[0]) + (si[1] * sj[1]) + (si[2] * sj[2]);

            // mij is the tensor product of si sj
            Real mij[3][3]       = {{si[0] * sj[0], si[0] * sj[1], si[0] * sj[2]},
                {si[1] * sj[0], si[1] * sj[1], si[1] * sj[2]},
                {si[2] * sj[0], si[2] * sj[1], si[2] * sj[2]}
            };

            Real mijTransp[3][3] = {{si[0] * sj[0], si[1] * sj[0], si[2] * sj[0]},
                {si[0] * sj[1], si[1] * sj[1], si[2] * sj[1]},
                {si[0] * sj[2], si[1] * sj[2], si[2] * sj[2]}
            };

            // t is the edge or vertex tensor
            for (int k = 0; k < 3; ++k)
            {
                for (int l = 0; l < 3; ++l)
                {
                    t[k][l] = lambda_ * mij[k][l] +
                            mu_ * ( mijTransp[k][l] + dot * id[k][l] );
                    // divide by (6.0 * volume)^2 to get the real shape vectors in the
                    // products ( (x) -> ^2) and multiplied by volume from the volumetric
                    // integration
                    t[k][l] /= 36.0f * tetra.restVolume;
                }
            }

            // add t to the tensor stored in edges or vertices
            if (i==j)
            {
                // t is the tensor for the vertex
                for (int k = 0; k < 3; ++k)
                {
                    for (int l = 0; l < 3; ++l)
                    {
                        vertexTensor_[tetra.vertex[i]].tensor[k][l] += t[k][l];
                    }
                }
            }
            else
            {
                // t (or its transposed, depending on the tetrahedron orientation) is
                // the tensor for the edge.
                if ( edge_[ tetra.edge[ vertexEdge[i][j] ] ].vertex[0] ==
                        tetra.vertex[i] )
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        for (int l = 0; l < 3; ++l)
                        {
                            edgeTensor_[tetra.edge[vertexEdge[i][j]]].tensor[k][l]+= t[k][l];
                        }
                    }
                }
                else
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        for (int l = 0; l < 3; ++l)
                        {
                            edgeTensor_[tetra.edge[vertexEdge[i][j]]].tensor[k][l]+= t[l][k];
                        }
                    }
                }
            }
        }
    }
}



//--- the following seems to be needed for factory registering


SOFA_DECL_CLASS(TensorForceField)

using namespace sofa::defaulttype;


template<class DataTypes>
void create(TensorForceField<DataTypes>*& obj,
        simulation::xml::ObjectDescription* arg)
{
    simulation::xml::createWithParentAndFilename<
    TensorForceField<DataTypes>, component::MechanicalObject<DataTypes>
    > (obj, arg);
}

#ifndef SOFA_FLOAT
Creator<simulation::xml::ObjectFactory, TensorForceField<Vec3dTypes> >
TensorForceFieldVec3dClass("TensorForceField", true);
template class TensorForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
Creator<simulation::xml::ObjectFactory, TensorForceField<Vec3fTypes> >
TensorForceFieldVec3fClass("TensorForceField", true);
template class TensorForceField<Vec3fTypes>; // doesn't work for now
#endif

} // namespace  forcefield

} // namespace  component

} // namespace sofa

