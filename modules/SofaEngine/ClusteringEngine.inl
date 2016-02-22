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
#ifndef SOFA_COMPONENT_ENGINE_CLUSTERING_INL
#define SOFA_COMPONENT_ENGINE_CLUSTERING_INL

#include <SofaEngine/ClusteringEngine.h>
#include <sofa/helper/gl/template.h>
#include <iostream>
#include <sofa/core/visual/VisualParams.h>
#include <fstream>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
ClusteringEngine<DataTypes>::ClusteringEngine()
    : useTopo(initData(&useTopo, true, "useTopo", "Use avalaible topology to compute neighborhood."))
    //,maxIter(initData(&maxIter, unsigned(500), "maxIter", "Max number of Lloyd iterations."))
    , radius(initData(&radius, (Real)1.0, "radius", "Neighborhood range."))
    , fixedRadius(initData(&fixedRadius, (Real)1.0, "fixedRadius", "Neighborhood range (for non mechanical particles)."))
    , number(initData(&number, (int)-1, "number", "Number of clusters (-1 means that all input points are selected)."))
    , fixedPosition(initData(&fixedPosition,"fixedPosition","Input positions of fixed (non mechanical) particles."))
    , position(initData(&position,"position","Input rest positions."))
    , cluster(initData(&cluster,"cluster","Computed clusters."))
    , input_filename(initData(&input_filename,"inFile","import precomputed clusters"))
    , output_filename(initData(&output_filename,"outFile","export clusters"))
    , topo(NULL)
{
}

template <class DataTypes>
void ClusteringEngine<DataTypes>::init()
{
    this->mstate = dynamic_cast< sofa::core::behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    addInput(&radius);
    addInput(&fixedRadius);
    addInput(&number);
    addInput(&fixedPosition);
    addInput(&position);
    addInput(&input_filename);
    addOutput(&cluster);
    setDirtyValue();

    //- Topology Container
    this->getContext()->get(topo);
}



template <class DataTypes>
void ClusteringEngine<DataTypes>::update()
{
    if(load()) return;

    sofa::helper::ReadAccessor< Data< VecCoord > > fixedPositions = this->fixedPosition;
    sofa::helper::ReadAccessor< Data< VecCoord > > restPositions = this->position;
    sofa::helper::WriteOnlyAccessor< Data< VVI > > clust = this->cluster;
    const unsigned int nbPoints =  restPositions.size(), nbFixed = fixedPositions.size();

    // get cluster centers
    VI ptIndices,voronoi;
    if(this->number.getValue() == -1)
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

    if(this->topo && this->useTopo.getValue())
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
                    if ( ((restPositions[j] - restPositions[ptIndices[i]]).norm() < this->radius.getValue()) )
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
                if ( ((fixedPositions[j] - restPositions[ptIndices[i]]).norm() < this->fixedRadius.getValue()) )
                    clust[i].push_back(j+nbPoints);
        }
    }

    save();
    cleanDirty();
}



template <class DataTypes>
void ClusteringEngine<DataTypes>::AddNeighborhoodFromNeighborhood(VI& lastNgb, const unsigned int i,const VI& voronoi)
{
    sofa::helper::ReadAccessor< Data< VecCoord > > restPositions = this->position;
    sofa::helper::WriteOnlyAccessor< Data< VVI > > clust = this->cluster;

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
            if ((voronoi[*it]==i) || ( (p - restPositions[pt]).norm() < this->radius.getValue()) )
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
    sofa::helper::ReadAccessor< Data< VecCoord > > restPositions = this->position;
    const Real distMax =std::numeric_limits<Real>::max();

    unsigned int nbp=restPositions.size();
    unsigned int nbc=(unsigned int)this->number.getValue();
    if(nbc>nbp) nbc=nbp;

    ptIndices.clear(); ptIndices.push_back(0);
    voronoi.resize(nbp); voronoi.fill(0);
    VD distances((int)nbp,distMax);

    while(ptIndices.size()<nbc)
    {
        if(this->topo && this->useTopo.getValue()) 	dijkstra(ptIndices , distances, voronoi);
        else Voronoi(ptIndices , distances, voronoi);

        Real dmax=0; ID imax;
        for (unsigned int i=0; i<nbp; i++) if(distances[i]>dmax) {dmax=distances[i]; imax=(ID)i;}
        if(dmax==0) break;
        else ptIndices.push_back(imax);
        sout<<"ClusteringEngine :"<<(int)floor(100.*(double)ptIndices.size()/(double)nbc)<<" % done\r";
    }
    sout<<"ClusteringEngine :100 % done\n";

    if(this->topo && this->useTopo.getValue()) 	dijkstra(ptIndices , distances, voronoi);
    else Voronoi(ptIndices , distances, voronoi);

    if (this->f_printLog.getValue()) for (unsigned int i=0; i<nbp; i++) std::cout<<"["<<i<<":"<<ptIndices[voronoi[i]]<<","<<distances[i]<<"]";
}

template <class DataTypes>
void ClusteringEngine<DataTypes>::LLoyd()
{
    // not yet implemented
}

template <class DataTypes>
void ClusteringEngine<DataTypes>::dijkstra(const VI& ptIndices , VD& distances, VI& voronoi)
{
    sofa::helper::ReadAccessor< Data< VecCoord > > restPositions = this->position;

    unsigned int i,nbi=ptIndices.size();

    typedef std::pair<Real,ID> DistanceToPoint;
    std::set<DistanceToPoint> q; // priority queue
    typename std::set<DistanceToPoint>::iterator qit;

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
    sofa::helper::ReadAccessor< Data< VecCoord > > restPositions = this->position;
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
    std::string fname(this->input_filename.getFullPath());
    if(!fname.compare(loadedFilename)) return true;

    if (!fname.size()) return false;
    if (!sofa::helper::system::DataRepository.findFile(fname))  { serr << "ClusteringEngine: cannot find "<<fname<<sendl;  return false;	}
    fname=sofa::helper::system::DataRepository.getFile(fname);

    std::ifstream fileStream (fname.c_str(), std::ifstream::in);
    if (!fileStream.is_open())	{ serr << "ClusteringEngine: cannot open "<<fname<<sendl;  return false;	}

    sofa::helper::WriteOnlyAccessor< Data< VVI > > clust = this->cluster;
    clust.clear();

    bool usetopo; fileStream >> usetopo;	this->useTopo.setValue(usetopo);
    Real rad; fileStream >> rad;		this->radius.setValue(usetopo);
    fileStream >> rad;		this->fixedRadius.setValue(usetopo);
    unsigned int nb; fileStream >> nb;			clust.resize(nb);
    int numb; fileStream >> numb;		this->number.setValue(usetopo);

    for (unsigned int i=0; i<nb; ++i)
    {
        unsigned int nbj; fileStream >> nbj;
        for (unsigned int j=0; j<nbj; ++j) {int k; fileStream >> k; clust[i].push_back(k);}
    }

    loadedFilename = fname;
    sout << "ClusteringEngine: loaded clusters from "<<fname<<sendl;
    //if (this->f_printLog.getValue())
    std::cout << "ClusteringEngine: loaded clusters from "<<fname<<std::endl;
    return true;
}

template <class DataTypes>
bool ClusteringEngine<DataTypes>::save()
{
    if (!this->output_filename.isSet()) return false;

    std::string fname(this->output_filename.getFullPath());
    if (!fname.size()) return false;

    std::ofstream fileStream (fname.c_str(), std::ofstream::out);
    if (!fileStream.is_open())	{ serr << "ClusteringEngine: cannot open "<<fname<<sendl;  return false;	}

    sofa::helper::ReadAccessor< Data< VVI > > clust = this->cluster;

    fileStream << this->useTopo.getValue() << " ";
    fileStream << this->radius.getValue() << " ";
    fileStream << this->fixedRadius.getValue() << " ";
    fileStream << clust.size() << " ";
    fileStream << this->number.getValue() << " ";
    fileStream << std::endl;

    for (unsigned int i=0; i<clust.size(); ++i)
    {
        fileStream << clust[i].size() << " ";
        for (unsigned int j=0; j< clust[i].size(); ++j) fileStream << clust[i][j] << " ";
        fileStream << std::endl;
    }

    sout << "ClusteringEngine: saved clusters in "<<fname<<sendl;
    //if (this->f_printLog.getValue())
    std::cout << "ClusteringEngine: saved clusters in "<<fname<<std::endl;

    return true;
}

template <class DataTypes>
void ClusteringEngine<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (vparams->displayFlags().getShowBehaviorModels())
    {
        const VecCoord& currentPositions = this->mstate->read(core::ConstVecCoordId::position())->getValue();
//        sofa::helper::ReadAccessor< Data< VecCoord > > fixedPositions = this->fixedPosition;
        sofa::helper::ReadAccessor< Data< VVI > > clust = this->cluster;
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
                    // helper::gl::glVertexT(Xcm[i]);
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
