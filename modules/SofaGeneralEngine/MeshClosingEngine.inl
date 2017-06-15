/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_ENGINE_MeshClosingEngine_INL
#define SOFA_COMPONENT_ENGINE_MeshClosingEngine_INL

#include "MeshClosingEngine.h"

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
void MeshClosingEngine<DataTypes>::update()
{
    helper::ReadAccessor<Data< SeqPositions > > pos(this->inputPosition);
    helper::ReadAccessor<Data< SeqTriangles > > tri(this->inputTriangles);
    helper::ReadAccessor<Data< SeqQuads > > qd(this->inputQuads);

    cleanDirty();

    helper::WriteOnlyAccessor<Data< SeqPositions > > opos(this->position);
    helper::WriteOnlyAccessor<Data< SeqTriangles > >  otri(this->triangles);
    helper::WriteOnlyAccessor<Data< SeqQuads > > oqd(this->quads);
    helper::WriteOnlyAccessor<Data< VecSeqIndex > >  oindices(this->indices);
    helper::WriteOnlyAccessor<Data< SeqPositions > > clpos(this->closingPosition);
    helper::WriteOnlyAccessor<Data< SeqTriangles > >  cltri(this->closingTriangles);

    // copy input mesh to output
    opos.resize(pos.size()); for(size_t i=0; i<pos.size(); i++) opos[i]=pos[i];
    otri.resize(tri.size()); for(size_t i=0; i<tri.size(); i++) otri[i]=tri[i];
    oqd.resize(qd.size()); for(size_t i=0; i<qd.size(); i++) oqd[i]=qd[i];
    oindices.clear();
    clpos.clear();
    cltri.clear();

    // get list of border edges
    typedef std::pair<unsigned int,unsigned int> edge; // orinted edges
    typedef std::set< edge > edgeset;
    typedef typename edgeset::iterator edgesetit;

    edgeset edges;
    for(size_t i=0; i<tri.size(); i++)
        for(size_t j=0; j<3; j++)
        {
            PointID p1=tri[i][(j==0)?2:j-1],p2=tri[i][j];
            edgesetit it=edges.find(edge(p2,p1));
            if(it==edges.end()) edges.insert(edge(p1,p2));
            else edges.erase(it);
        }
    for(size_t i=0; i<qd.size(); i++)
        for(size_t j=0; j<4; j++)
        {
            PointID p1=qd[i][(j==0)?3:j-1],p2=qd[i][j];
            edgesetit it=edges.find(edge(p2,p1));
            if(it==edges.end()) edges.insert(edge(p1,p2));
            else edges.erase(it);
        }
    if(edges.empty()) return; // no hole

    // get loops
    typedef std::map<PointID,PointID> edgemap;
    edgemap emap;
    for(edgesetit it=edges.begin(); it!=edges.end(); it++)  emap[it->first]=it->second;

    typename edgemap::iterator it=emap.begin();
    VecSeqIndex loops; loops.resize(1);
    loops.back().push_back(it->first);
    while(!emap.empty())
    {
        PointID i=it->second;
        loops.back().push_back(i);  // insert point in loop
        emap.erase(it);
        if(!emap.empty())
        {
            if(i==loops.back().front())  loops.push_back(SeqIndex());  //  loop termination
            it=emap.find(i);
            if(it==emap.end())
            {
                it=emap.begin();
                loops.back().push_back(it->first);
            }
        }
    }
    sout<<"Found "<< loops.size()<<" loops"<<sendl;
//    for(size_t i=0;i<loops.size();i++) for(size_t j=0;j<loops[i].size();j++) std::cout<<"loop "<<i<<","<<j<<":"<<loops[i][j]<<std::endl;

    // insert points at loop centroids and triangles connecting loop edges and centroids
    for(size_t i=0; i<loops.size(); i++)
        if(loops[i].size()>2)
        {
            SeqIndex ind;
            Coord centroid;
            size_t indexCentroid=clpos.size()+loops[i].size()-1;
            for(size_t j=0; j<loops[i].size()-1; j++)
            {
                PointID posIdx = loops[i][j];
                ind.push_back(posIdx);
                clpos.push_back(pos[posIdx]);
                centroid+=pos[posIdx];
                cltri.push_back(Triangle(indexCentroid,clpos.size()-1,j?clpos.size()-2:indexCentroid-1));
                otri.push_back(Triangle(opos.size(),posIdx,j?loops[i][j-1]:loops[i][loops[i].size()-2]));
            }
            centroid/=(Real)(loops[i].size()-1);
            clpos.push_back(centroid);
            ind.push_back(opos.size());
            opos.push_back(centroid);
            oindices.push_back(ind);
        }
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
