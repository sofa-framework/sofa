/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_ENGINE_GENERATEGRID_INL
#define SOFA_COMPONENT_ENGINE_GENERATEGRID_INL

#include "GenerateGrid.h"
#include <SofaBaseMechanics/IdentityMapping.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
GenerateGrid<DataTypes>::GenerateGrid()
    : d_outputX ( initData (&d_outputX, "output_position", "output array of 3d points") )
    , d_tetrahedron( initData (&d_tetrahedron, "tetrahedra", "output mesh tetrahedra") )
    , d_quad( initData (&d_quad, "quads", "output mesh quads") )
    , d_triangle( initData (&d_triangle, "triangles", "output mesh triangles") )
    , d_hexahedron( initData (&d_hexahedron, "hexahedra", "output mesh hexahedra") )
    , d_minCorner( initData (&d_minCorner,Vec3(), "min", "the 3 coordinates of the minimum corner") )
    , d_maxCorner( initData (&d_maxCorner,Vec3(), "max", "the 3 coordinates of the maximum corner") )
    , d_resolution( initData (&d_resolution,Vec3Int(3,3,3), "resolution", "the number of cubes in the x,y,z directions. If resolution in the z direction is  0 then a 2D grid is generated") )
{
    addAlias(&d_outputX,"position");
}


template <class DataTypes>
void GenerateGrid<DataTypes>::init()
{
    addInput(&d_minCorner);
    addInput(&d_maxCorner);
    addInput(&d_resolution);
    addOutput(&d_outputX);
    addOutput(&d_tetrahedron);
    addOutput(&d_hexahedron);
    addOutput(&d_quad);
    addOutput(&d_triangle);
    setDirtyValue();
}

template <class DataTypes>
void GenerateGrid<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void GenerateGrid<DataTypes>::update()
{
    cleanDirty();

    helper::WriteAccessor<Data<VecCoord> > out = d_outputX;

    Vec3 size=d_maxCorner.getValue()-d_minCorner.getValue();

    size_t freqL=d_resolution.getValue()[0];
    size_t freqH=d_resolution.getValue()[2];
    size_t freqW=d_resolution.getValue()[1];

    if (freqL==0) {
        serr<<" Number of cubes in the x direction cannot be 0; Changed to 1"<<sendl;
        freqL=1;
    }
    if (freqW==0) {
        serr<<" Number of cubes in the y direction cannot be 0; Changed to 1"<<sendl;
        freqW=1;
    }
    const Real length = size[0]/freqL;
    const Real width = size[1]/freqW;
    Real height;
    if (freqH==0)
        height=0;
    else
        height = size[2]/freqH;
    Coord origin;
    helper::eq(origin,Vec3(d_minCorner.getValue()));




    size_t  nbVertices= (freqL+1)*(freqH+1)*(freqW+1);
    out.resize(nbVertices);

    size_t i,j,k,index;
    Coord pos;

    for(index=0,k=0;k<=freqH;++k) {
        for(j=0;j<=freqW;++j) {
            for(i=0;i<=freqL;i++) {
                // handle Vec2D case
                helper::eq(pos,Vec3(i*length,j*width,k*height));
                pos+=origin;
                out[index++]=pos;
            }
        }
    }

    if (freqH==0) {
        // only output quads & triangles
        size_t nbQuads=(freqL)*(freqW);
        SeqTriangles  &triangles = *(d_triangle.beginEdit());
        SeqQuads  &quads = *(d_quad.beginEdit());
        quads.resize(nbQuads);
        triangles.resize(nbQuads*2);


        Quad quad;
        for(index=0,i=0;i<freqL;i++) {
            for(j=0;j<freqW;++j) {
                quad[0]=(PointID)(i+j*(freqL+1));
                quad[1]=(PointID)(i+1+j*(freqL+1));
                quad[2]=(PointID)(i+1+(j+1)*(freqL+1));
                quad[3]=(PointID)(i+(j+1)*(freqL+1));
                quads[index]=quad;
                /// decompose quad into 2 triangles tetra
                triangles[2*index]=Triangle(quad[0],quad[1],quad[3]);
                triangles[2*index+1]=Triangle(quad[3],quad[1],quad[2]);

                index++;

            }
        }

    } else {
        // outputs hexahedra & tetrahedra
        SeqTetrahedra  &tetras = *(d_tetrahedron.beginEdit());
        SeqHexahedra  &hexas = *(d_hexahedron.beginEdit());
        size_t nbHexahedra=(freqL)*(freqH)*(freqW);
        hexas.resize(nbHexahedra);
        tetras.resize(nbHexahedra*6);

        typedef sofa::core::topology::Topology::PointID PointID;
        Hexahedron hexahedron;
        for(index=0,i=0;i<freqL;i++) {
            for(j=0;j<freqW;++j) {
                for(k=0;k<freqH;++k) {
                    hexahedron[0]=(PointID)(i+j*(freqL+1)+k*(freqL+1)*(freqW+1));
                    hexahedron[1]=(PointID)(i+1+j*(freqL+1)+k*(freqL+1)*(freqW+1));
                    hexahedron[2]=(PointID)(i+1+(j+1)*(freqL+1)+k*(freqL+1)*(freqW+1));
                    hexahedron[3]=(PointID)(i+(j+1)*(freqL+1)+k*(freqL+1)*(freqW+1));
                    hexahedron[4]=(PointID)(i+j*(freqL+1)+(k+1)*(freqL+1)*(freqW+1));
                    hexahedron[5]=(PointID)(i+1+j*(freqL+1)+(k+1)*(freqL+1)*(freqW+1));
                    hexahedron[6]=(PointID)(i+1+(j+1)*(freqL+1)+(k+1)*(freqL+1)*(freqW+1));
                    hexahedron[7]=(PointID)(i+(j+1)*(freqL+1)+(k+1)*(freqL+1)*(freqW+1));
                    hexas[index]=hexahedron;
                    /// decompose hexahedron into 6 tetra
                    tetras[6*index]=Tetrahedron(hexahedron[0],hexahedron[5],hexahedron[1],hexahedron[6]);
                    tetras[6*index+1]=Tetrahedron(hexahedron[0],hexahedron[1],hexahedron[3],hexahedron[6]);
                    tetras[6*index+2]=Tetrahedron(hexahedron[1],hexahedron[3],hexahedron[6],hexahedron[2]);
                    tetras[6*index+3]=Tetrahedron(hexahedron[6],hexahedron[3],hexahedron[0],hexahedron[7]);
                    tetras[6*index+4]=Tetrahedron(hexahedron[6],hexahedron[7],hexahedron[0],hexahedron[5]);
                    tetras[6*index+5]=Tetrahedron(hexahedron[7],hexahedron[5],hexahedron[4],hexahedron[0]);
                    index++;
                }
            }
        }
    }





    d_tetrahedron.endEdit();
    d_hexahedron.endEdit();
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
