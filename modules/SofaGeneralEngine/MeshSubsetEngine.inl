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
#ifndef SOFA_COMPONENT_ENGINE_MeshSubsetEngine_INL
#define SOFA_COMPONENT_ENGINE_MeshSubsetEngine_INL

#include "MeshSubsetEngine.h"

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
void MeshSubsetEngine<DataTypes>::update()
{
    helper::ReadAccessor<Data< SeqPositions > > pos(this->inputPosition);
    helper::ReadAccessor<Data< SeqEdges > > edg(this->inputEdges);
    helper::ReadAccessor<Data< SeqTriangles > > tri(this->inputTriangles);
    helper::ReadAccessor<Data< SeqQuads > > qd(this->inputQuads);
    helper::ReadAccessor<Data< SetIndices > >  ind(this->indices);

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

    this->cleanDirty();
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
