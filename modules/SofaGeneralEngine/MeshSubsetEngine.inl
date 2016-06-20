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
    helper::ReadAccessor<Data< SeqTriangles > > tri(this->inputTriangles);
    helper::ReadAccessor<Data< SeqQuads > > qd(this->inputQuads);
    helper::ReadAccessor<Data< SetIndices > >  ind(this->indices);

    helper::WriteOnlyAccessor<Data< SeqPositions > > opos(this->position);
    helper::WriteOnlyAccessor<Data< SeqTriangles > >  otri(this->triangles);
    helper::WriteOnlyAccessor<Data< SeqQuads > > oqd(this->quads);

    opos.resize(ind.size());
    std::map<PointID,PointID> FtoS;
    for(size_t i=0; i<ind.size(); i++)
    {
        opos[i]=pos[ind[i]];
        FtoS[ind[i]]=i;
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
