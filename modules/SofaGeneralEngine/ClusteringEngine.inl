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
#ifndef SOFA_COMPONENT_ENGINE_CLUSTERING_INL
#define SOFA_COMPONENT_ENGINE_CLUSTERING_INL

#include <SofaGeneralEngine/ClusteringEngine.h>
#include <sofa/helper/gl/template.h>
#include <iostream>
#include <sofa/core/visual/VisualParams.h>
#include <fstream>

#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/FileRepository.h>

namespace sofa
{

namespace component
{

namespace engine
{

using std::pair;
using std::set;
using std::string;
using std::ofstream;
using std::ifstream;
using std::numeric_limits;

using sofa::helper::ReadAccessor;
using sofa::helper::WriteOnlyAccessor;

template <class DataTypes>
ClusteringEngine<DataTypes>::ClusteringEngine()
    : d_useTopo(initData(&d_useTopo, true, "useTopo", "Use avalaible topology to compute neighborhood."))
    , d_radius(initData(&d_radius, (Real)1.0, "radius", "Neighborhood range."))
    , d_fixedRadius(initData(&d_fixedRadius, (Real)1.0, "fixedRadius", "Neighborhood range (for non mechanical particles)."))
    , d_nbClusters(initData(&d_nbClusters, (int)-1, "number", "Number of clusters (-1 means that all input points are selected)."))
    , d_fixedPosition(initData(&d_fixedPosition,"fixedPosition","Input positions of fixed (non mechanical) particles."))
    , d_position(initData(&d_position,"position","Input rest positions."))
    , d_cluster(initData(&d_cluster,"cluster","Computed clusters."))
    , input_filename(initData(&input_filename,"inFile","import precomputed clusters"))
    , output_filename(initData(&output_filename,"outFile","export clusters"))
    , topo(NULL)
{
}

template <class DataTypes>
void ClusteringEngine<DataTypes>::init()
{
    this->mstate = dynamic_cast< sofa::core::behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());

    if(this->mstate==NULL)
        msg_info(this) << "This component requires a mechanical state in its context for output visualization.";

    addInput(&d_radius);
    addInput(&d_fixedRadius);
    addInput(&d_nbClusters);
    addInput(&d_fixedPosition);
    addInput(&d_position);
    addInput(&input_filename);
    addOutput(&d_cluster);
    setDirtyValue();

    //- Topology Container
    this->getContext()->get(topo);
}



template <class DataTypes>
void ClusteringEngine<DataTypes>::update()
{
    if(load()) return;

    ReadAccessor< Data< VecCoord > > fixedPositions = this->d_fixedPosition;
    ReadAccessor< Data< VecCoord > > restPositions = this->d_position;
    WriteOnlyAccessor< Data< VVI > > clust = this->d_cluster;
    const unsigned int nbPoints =  restPositions.size(), nbFixed = fixedPositions.size();

    // get cluster centers
    VI ptIndices,voronoi;
    if(this->d_nbClusters.getValue() == -1)
    {
        ptIndices.clear();	voronoi.clear();
        for (unsigned int i=0; i<nbPoints; ++i) { ptIndices.push_back(i); voronoi.push_back(i); }
    }
    else
    {
        farthestPointSampling(ptIndices,voronoi);
        LLoyd();
    }

    // get points in clusters
    clust.resize(ptIndices.size());		for (unsigned int i=0; i<ptIndices.size(); ++i)  { clust[i].clear(); clust[i].push_back(ptIndices[i]); }

    if(this->topo && this->d_useTopo.getValue())
    {
        for (unsigned int i=0; i<ptIndices.size(); ++i)
        {
            VI lastN; lastN.push_back(ptIndices[i]);
            AddNeighborhoodFromNeighborhood(lastN,i,voronoi);
        }
    }
    else
    {
        // add mechanical points
        for (unsigned int j=0; j<nbPoints; ++j)
        {
            bool inserted =false;
            for (unsigned int i=0; i<ptIndices.size(); ++i)
                if(j != ptIndices[i])
                    if ( ((restPositions[j] - restPositions[ptIndices[i]]).norm() < this->d_radius.getValue()) )
                    {
                        clust[i].push_back(j);
                        inserted=true;
                    }
            if(!inserted) // add point to closest cluster to avoid free points
            {
                Real d,dmin=std::numeric_limits<Real>::max(); int imin=-1;
                for (unsigned int i=0; i<ptIndices.size(); ++i)
                    if(j != ptIndices[i])
                    {
                        d= (restPositions[j] - restPositions[ptIndices[i]]).norm();
                        if ( d < dmin ) { dmin=d; imin=i; }
                    }
                if(imin!=-1) clust[imin].push_back(j);
            }
        }

        // add non mechanical points
        for (unsigned int j=0; j<nbFixed; ++j)
        {
            for (unsigned int i=0; i<ptIndices.size(); ++i)
                if ( ((fixedPositions[j] - restPositions[ptIndices[i]]).norm() < this->d_fixedRadius.getValue()) )
                    clust[i].push_back(j+nbPoints);
        }
    }

    save();
    cleanDirty();
}



template <class DataTypes>
void ClusteringEngine<DataTypes>::AddNeighborhoodFromNeighborhood(VI& lastNgb, const unsigned int i,const VI& voronoi)
{
    ReadAccessor< Data< VecCoord > > restPositions = this->d_position;
    WriteOnlyAccessor< Data< VVI > > clust = this->d_cluster;

    VI newNgb;
    VI *Ngb=&clust[i];
    const Coord &p=restPositions[clust[i][0]];

    bool inserted=false;
    for (VI::const_iterator it = lastNgb.begin() ; it != lastNgb.end() ; ++it)
    {
        const helper::vector<ID>& ngb = this->topo->getVerticesAroundVertex(*it);
        for (unsigned int j=0 ; j<ngb.size(); ++j)
        {
            ID pt = ngb[j];
            // insert if dist<radius, but insert at least the voronoi+ one ring
            if ((voronoi[*it]==i) || ( (p - restPositions[pt]).norm() < this->d_radius.getValue()) )
                if(find(Ngb->begin(),Ngb->end(),pt) == Ngb->end())
                {
                    Ngb->push_back(pt);
                    newNgb.push_back(pt);
                    inserted=true;
                }
        }
    }
    lastNgb.assign(newNgb.begin(),newNgb.end());
    if(inserted) AddNeighborhoodFromNeighborhood(lastNgb,i,voronoi);
}


template <class DataTypes>
void ClusteringEngine<DataTypes>::farthestPointSampling(VI& ptIndices,VI& voronoi)
{
    ReadAccessor< Data< VecCoord > > restPositions = this->d_position;
    const Real distMax = numeric_limits<Real>::max();

    unsigned int nbp=restPositions.size();
    unsigned int nbc=(unsigned int)this->d_nbClusters.getValue();
    if(nbc>nbp) nbc=nbp;

    ptIndices.clear(); ptIndices.push_back(0);
    voronoi.resize(nbp); voronoi.fill(0);
    VD distances((int)nbp,distMax);

    while(ptIndices.size()<nbc)
    {
        if(this->topo && this->d_useTopo.getValue()) 	dijkstra(ptIndices , distances, voronoi);
        else Voronoi(ptIndices , distances, voronoi);

        Real dmax=0; ID imax;
        for (unsigned int i=0; i<nbp; i++) if(distances[i]>dmax) {dmax=distances[i]; imax=(ID)i;}
        if(dmax==0) break;
        else ptIndices.push_back(imax);
        sout<<"ClusteringEngine :"<<(int)floor(100.*(double)ptIndices.size()/(double)nbc)<<" % done\r";
    }
    sout<<"ClusteringEngine :100 % done\n";

    if(this->topo && this->d_useTopo.getValue()) 	dijkstra(ptIndices , distances, voronoi);
    else Voronoi(ptIndices , distances, voronoi);


    if (notMuted())
    {
        std::stringstream tmp;
        for (unsigned int i=0; i<nbp; i++)
            tmp<<"["<<i<<":"<<ptIndices[voronoi[i]]<<","<<distances[i]<<"]";
        dmsg_info() << tmp.str() ;
    }
}

template <class DataTypes>
void ClusteringEngine<DataTypes>::LLoyd()
{
    // not yet implemented
}

template <class DataTypes>
void ClusteringEngine<DataTypes>::dijkstra(const VI& ptIndices , VD& distances, VI& voronoi)
{
    ReadAccessor< Data< VecCoord > > restPositions = this->d_position;

    unsigned int i,nbi=ptIndices.size();

    typedef pair<Real,ID> DistanceToPoint;
    set<DistanceToPoint> q; // priority queue
    typename set<DistanceToPoint>::iterator qit;

    for (i=0; i<nbi; i++)
    {
        q.insert( DistanceToPoint((Real)0,ptIndices[i]) );
        distances[ptIndices[i]]=0;
        voronoi[ptIndices[i]]=i;
    }

    while( !q.empty() )
    {
        DistanceToPoint top = *q.begin();
        q.erase(q.begin());
        ID v = top.second;

        const helper::vector<ID>& ngb = this->topo->getVerticesAroundVertex(v);
        for (i=0 ; i<ngb.size(); ++i)
        {
            ID v2 = ngb[i];

            Real d = distances[v] + (restPositions[v] - restPositions[v2]).norm();
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

template <class DataTypes>
void ClusteringEngine<DataTypes>::Voronoi(const VI& ptIndices , VD& distances, VI& voronoi)
{
    ReadAccessor< Data< VecCoord > > restPositions = this->d_position;
    for (unsigned int i=0; i<restPositions.size(); i++)
    {
        for (unsigned int j=0; j<ptIndices.size(); j++)
        {
            Real d=(restPositions[i] - restPositions[ptIndices[j]]).norm();
            if(d<distances[i]) { distances[i]=d; voronoi[i]=j;}
        }
    }
}



template <class DataTypes>
bool ClusteringEngine<DataTypes>::load()
{
    if (!this->input_filename.isSet()) return false;

    input_filename.update();
    string fname(this->input_filename.getFullPath());
    if(!fname.compare(loadedFilename)) return true;

    if (!fname.size()) return false;
    if (!helper::system::DataRepository.findFile(fname))  { serr << "ClusteringEngine: cannot find "<<fname<<sendl;  return false;	}
    fname=helper::system::DataRepository.getFile(fname);

    ifstream fileStream (fname.c_str(), std::ifstream::in);
    if (!fileStream.is_open())	{ serr << "ClusteringEngine: cannot open "<<fname<<sendl;  return false;	}

    WriteOnlyAccessor< Data< VVI > > clust = this->d_cluster;
    clust.clear();

    bool usetopo; fileStream >> usetopo;	this->d_useTopo.setValue(usetopo);
    Real rad; fileStream >> rad;		this->d_radius.setValue(usetopo);
    fileStream >> rad;		this->d_fixedRadius.setValue(usetopo);
    unsigned int nb; fileStream >> nb;			clust.resize(nb);
    int numb; fileStream >> numb;		this->d_nbClusters.setValue(usetopo);

    for (unsigned int i=0; i<nb; ++i)
    {
        unsigned int nbj; fileStream >> nbj;
        for (unsigned int j=0; j<nbj; ++j) {int k; fileStream >> k; clust[i].push_back(k);}
    }

    loadedFilename = fname;
    sout << "ClusteringEngine: loaded clusters from "<<fname<<sendl;
    return true;
}

template <class DataTypes>
bool ClusteringEngine<DataTypes>::save()
{
    if (!this->output_filename.isSet()) return false;

    string fname(this->output_filename.getFullPath());
    if (!fname.size()) return false;

    ofstream fileStream (fname.c_str(), ofstream::out);
    if (!fileStream.is_open())	{ serr << "ClusteringEngine: cannot open "<<fname<<sendl;  return false;	}

    ReadAccessor< Data< VVI > > clust = this->d_cluster;

    fileStream << this->d_useTopo.getValue() << " ";
    fileStream << this->d_radius.getValue() << " ";
    fileStream << this->d_fixedRadius.getValue() << " ";
    fileStream << clust.size() << " ";
    fileStream << this->d_nbClusters.getValue() << " ";
    fileStream << std::endl;

    for (unsigned int i=0; i<clust.size(); ++i)
    {
        fileStream << clust[i].size() << " ";
        for (unsigned int j=0; j< clust[i].size(); ++j) fileStream << clust[i][j] << " ";
        fileStream << std::endl;
    }

    sout << "ClusteringEngine: saved clusters in "<<fname<<sendl;

    return true;
}

template <class DataTypes>
void ClusteringEngine<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (vparams->displayFlags().getShowBehaviorModels())
    {
        if(this->mstate==NULL)
            return;

        const VecCoord& currentPositions = this->mstate->read(core::ConstVecCoordId::position())->getValue();
        ReadAccessor< Data< VVI > > clust = this->d_cluster;
        const unsigned int nbp = currentPositions.size();

        glPushAttrib( GL_LIGHTING_BIT);

        glDisable(GL_LIGHTING);

        glBegin(GL_LINES);

        float r, g, b;

        for (unsigned int i=0 ; i<clust.size() ; ++i)
        {
            r = (float)((i*7543)%11)/11;
            g = (float)((i*1357)%13)/13;
            b = (float)((i*4829)%17)/17;

            glColor3f(r,g,b);

            VI::const_iterator it, itEnd;
            for (it = clust[i].begin()+1, itEnd = clust[i].end(); it != itEnd ; ++it)
                if(*it<nbp) // discard visualization of fixed particles (as their current positions is unknown)
                {
                    helper::gl::glVertexT(currentPositions[clust[i].front()]);
                    helper::gl::glVertexT(currentPositions[*it]);
                }
        }
        glEnd();

        glPopAttrib();
    }
#endif /* SOFA_NO_OPENGL */
}




} // namespace engine

} // namespace component

} // namespace sofa

#endif
