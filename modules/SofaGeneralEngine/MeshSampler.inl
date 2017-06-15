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
#ifndef SOFA_COMPONENT_ENGINE_MESHSAMPLER_INL
#define SOFA_COMPONENT_ENGINE_MESHSAMPLER_INL

#include <SofaGeneralEngine/MeshSampler.h>
#include <sofa/helper/gl/template.h>
#include <iostream>


namespace sofa
{

namespace component
{

namespace engine
{


using namespace sofa::helper;
using namespace sofa::defaulttype;
using namespace core::objectmodel;

template <class DataTypes>
MeshSampler<DataTypes>::MeshSampler()
    : DataEngine()
    , number(initData(&number, (unsigned int)1, "number", "Sample number"))
    , position(initData(&position,"position","Input positions."))
    , f_edges(initData(&f_edges,"edges","Input edges for geodesic sampling (Euclidean distances are used if not specified)."))
    , maxIter(initData(&maxIter, (unsigned int)100, "maxIter", "Max number of Lloyd iterations."))
    , outputIndices(initData(&outputIndices,"outputIndices","Computed sample indices."))
    , outputPosition(initData(&outputPosition,"outputPosition","Computed sample coordinates."))
{
}

template <class DataTypes>
void MeshSampler<DataTypes>::init()
{
    addInput(&number);
    addInput(&position);
    addInput(&f_edges);
    addInput(&maxIter);
    addOutput(&outputIndices);
    addOutput(&outputPosition);
    setDirtyValue();
}



template <class DataTypes>
void MeshSampler<DataTypes>::update()
{
    sofa::helper::ReadAccessor< Data< VecCoord > > pos = this->position;

    number.updateIfDirty();
    f_edges.updateIfDirty();
    maxIter.updateIfDirty();

    cleanDirty();

    VVI ngb;    if(this->f_edges.getValue().size()!=0) computeNeighbors(ngb); // one ring neighbors from edges
    VI voronoi;
    VD distances;

    // farthestPointSampling
    farthestPointSampling(distances,voronoi,ngb);

    // Lloyd
    unsigned int count=0;
    while(count<this->maxIter.getValue())
    {
        if(!LLoyd(distances,voronoi,ngb)) break;
        count++;
    }
    dmsg_info() <<this->getPathName()<<": Lloyd relaxation done in "<<count<<" iterations" ;

    // get export position from indices
    sofa::helper::WriteOnlyAccessor< Data< VI > > ind = this->outputIndices;
    sofa::helper::WriteOnlyAccessor< Data< VecCoord > > outPos = this->outputPosition;
    outPos.resize(ind.size());		for (unsigned int i=0; i<ind.size(); ++i)  outPos[i]=pos[ind[i]];

}

template <class DataTypes>
void MeshSampler<DataTypes>::computeNeighbors(VVI& ngb)
{
    sofa::helper::ReadAccessor< Data< SeqEdges > > edges = this->f_edges;
    unsigned int nbp=this->position.getValue().size();
    ngb.resize(nbp);    for(unsigned int i=0;i<nbp;i++) ngb[i].clear();
    for (unsigned int j = 0; j<edges.size(); j++)
    {
        ngb[edges[j][0]].push_back ( edges[j][1] );
        ngb[edges[j][1]].push_back ( edges[j][0] );
    }
}


template <class DataTypes>
void MeshSampler<DataTypes>::farthestPointSampling(VD& distances,VI& voronoi,const VVI& ngb)
{
    sofa::helper::WriteOnlyAccessor< Data< VI > > ind = this->outputIndices;

    ind.clear();
    ind.push_back(0); // add first point

    computeDistances( distances, voronoi, ngb);

    unsigned int nbp = this->position.getValue().size(), nbc=this->number.getValue();
    if(nbc>nbp) nbc=nbp;

    while(ind.size()<nbc)
    {
        Real dmax=0; ID imax;
        for (unsigned int i=0; i<distances.size(); i++) if(distances[i]>dmax) {dmax=distances[i]; imax=(ID)i;}
        if(dmax==0) break;
        else ind.push_back(imax);
        computeDistances( distances, voronoi, ngb);
    }

    dmsg_info() <<this->getPathName()<<": farthestPointSampling done" ;
}

template <class DataTypes>
bool MeshSampler<DataTypes>::LLoyd(VD& distances,VI& voronoi,const VVI& ngb)
{
    sofa::helper::WriteOnlyAccessor< Data< VI > > ind = this->outputIndices;
    sofa::helper::ReadAccessor< Data< VecCoord > > pos = this->position;

    unsigned int nbp = pos.size(), nbs = ind.size();

    // update voronoi region centers
    VecCoord center(nbs);
    VI nb(nbs);
    for(unsigned int i=0;i<nbs;i++) { center[i]=Coord(); nb[i]=0;    }
    for(unsigned int i=0;i<nbp;i++) { center[voronoi[i]]+=pos[i]; nb[voronoi[i]]+=1; }
    for(unsigned int i=0;i<nbs;i++) center[i]/=(Real)nb[i];

    // replace each sample by the closest point from the region center
    VD d(nbp);
    for(unsigned int i=0;i<nbp;i++) d[i]=(pos[i]-center[voronoi[i]]).norm2();
    VD dmin(nbs);
    for(unsigned int i=0;i<nbs;i++) dmin[i]=d[ind[i]];
    bool changed=false;
    for(unsigned int i=0;i<nbp;i++) if(d[i]<dmin[voronoi[i]]) { dmin[voronoi[i]]=d[i]; ind[voronoi[i]]=i; changed=true;}

    // update distances
    computeDistances( distances, voronoi, ngb);

    return changed;
}

template <class DataTypes>
void MeshSampler<DataTypes>::computeDistances(VD& distances, VI& voronoi,const VVI& ngb)
{
    sofa::helper::WriteOnlyAccessor< Data< VI > > ind = this->outputIndices;
    sofa::helper::ReadAccessor< Data< VecCoord > > pos = this->position;

    unsigned int nbp = pos.size();
    distances.resize(nbp); for (unsigned int i=0; i<nbp; i++)  distances[i]=std::numeric_limits<Real>::max();
    voronoi.resize(nbp);

    if(this->f_edges.getValue().size()==0)  // use Euclidean distances
    {
        for (unsigned int i=0; i<nbp; i++)
        {
            for (unsigned int j=0; j<ind.size(); j++)
            {
                Real d=(pos[i] - pos[ind[j]]).norm2();
                if(d<distances[i]) { distances[i]=d; voronoi[i]=j;}
            }
        }
    }
    else
    {
        typedef std::pair<Real,ID> DistanceToPoint;
        std::set<DistanceToPoint> q; // priority queue
        typename std::set<DistanceToPoint>::iterator qit;

        for (unsigned int i=0; i<ind.size(); i++)
        {
            q.insert( DistanceToPoint((Real)0,ind[i]) );
            distances[ind[i]]=0;
            voronoi[ind[i]]=i;
        }

        while( !q.empty() )
        {
            DistanceToPoint top = *q.begin();
            q.erase(q.begin());
            ID v = top.second;

            for (unsigned int i=0 ; i<ngb[v].size(); ++i)
            {
                ID v2 = ngb[v][i];

                Real d = distances[v] + (pos[v] - pos[v2]).norm();
                if(distances[v2] > d )
                {
                    qit=q.find(DistanceToPoint(distances[v2],v2));
                    if(qit != q.end()) q.erase(qit);
                    voronoi[v2]=voronoi[v];
                    distances[v2] = d;
                    q.insert( DistanceToPoint(d,v2) );
                }
            }
        }
    }
}



template <class DataTypes>
void MeshSampler<DataTypes>::draw(const core::visual::VisualParams* /*vparams*/)
{

}




} // namespace engine

} // namespace component

} // namespace sofa

#endif
