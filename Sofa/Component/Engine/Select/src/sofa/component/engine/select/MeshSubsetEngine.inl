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
#include <sofa/component/engine/select/MeshSubsetEngine.h>

namespace sofa::component::engine::select
{

template <class DataTypes>
MeshSubsetEngine<DataTypes>::MeshSubsetEngine()
    : Inherited()
    , inputPosition(initData(&inputPosition,"inputPosition","input vertices"))
    , inputEdges(initData(&inputEdges,"inputEdges","input edges"))
    , inputTriangles(initData(&inputTriangles,"inputTriangles","input triangles"))
    , inputQuads(initData(&inputQuads,"inputQuads","input quads"))
    , indices(initData(&indices,"indices","Index lists of the selected vertices"))
    , position(initData(&position,"position","Vertices of mesh subset"))
    , edges(initData(&edges,"edges","edges of mesh subset"))
    , triangles(initData(&triangles,"triangles","Triangles of mesh subset"))
    , quads(initData(&quads,"quads","Quads of mesh subset"))
{
    addInput(&inputPosition);
    addInput(&inputEdges);
    addInput(&inputTriangles);
    addInput(&inputQuads);
    addInput(&indices);
    addOutput(&position);
    addOutput(&edges);
    addOutput(&triangles);
    addOutput(&quads);
}

template <class DataTypes>
MeshSubsetEngine<DataTypes>::~MeshSubsetEngine()
{
}

template <class DataTypes>
void MeshSubsetEngine<DataTypes>::doUpdate()
{
    helper::ReadAccessor<Data< SeqPositions > > pos(this->inputPosition);
    const helper::ReadAccessor<Data< SeqEdges > > edg(this->inputEdges);
    const helper::ReadAccessor<Data< SeqTriangles > > tri(this->inputTriangles);
    const helper::ReadAccessor<Data< SeqQuads > > qd(this->inputQuads);
    const helper::ReadAccessor<Data< SetIndices > >  ind(this->indices);

    helper::WriteOnlyAccessor<Data< SeqPositions > > opos(this->position);
    helper::WriteOnlyAccessor<Data< SeqEdges > >  oedg(this->edges);
    helper::WriteOnlyAccessor<Data< SeqTriangles > >  otri(this->triangles);
    helper::WriteOnlyAccessor<Data< SeqQuads > > oqd(this->quads);

    opos.resize(ind.size());
    std::map<PointID,PointID> FtoS;
    for(size_t i=0; i<ind.size(); i++)
    {
        opos[i]=pos[ind[i]];
        FtoS[ind[i]]=i;
    }
    oedg.clear();
    for(size_t i=0; i<edg.size(); i++)
    {
        bool inside=true;
        Edge cell;
        for(size_t j=0; j<2; j++) if(FtoS.find(edg[i][j])==FtoS.end()) { inside=false; break; } else cell[j]=FtoS[edg[i][j]];
        if(inside) oedg.push_back(cell);
    }
    otri.clear();
    for(size_t i=0; i<tri.size(); i++)
    {
        bool inside=true;
        Triangle cell;
        for(size_t j=0; j<3; j++) if(FtoS.find(tri[i][j])==FtoS.end()) { inside=false; break; } else cell[j]=FtoS[tri[i][j]];
        if(inside) otri.push_back(cell);
    }
    oqd.clear();
    for(size_t i=0; i<qd.size(); i++)
    {
        bool inside=true;
        Quad cell;
        for(size_t j=0; j<4; j++) if(FtoS.find(qd[i][j])==FtoS.end()) { inside=false; break; } else cell[j]=FtoS[qd[i][j]];
        if(inside) oqd.push_back(cell);
    }
}


} //namespace sofa::component::engine::select
