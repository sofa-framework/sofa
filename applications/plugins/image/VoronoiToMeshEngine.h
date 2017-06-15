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
#ifndef SOFA_IMAGE_VoronoiToMeshENGINE_H
#define SOFA_IMAGE_VoronoiToMeshENGINE_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace engine
{


/**
 * This class generates flat faces between adjacent regions of an image
 */


template <class _ImageTypes>
class VoronoiToMeshEngine : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(VoronoiToMeshEngine,_ImageTypes),Inherited);

    typedef SReal Real;

    Data< bool > showMesh;

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;
    Data< ImageTypes > background;

    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    typedef helper::vector<Coord > SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    typedef helper::WriteOnlyAccessor<Data< SeqPositions > > waPositions;
    Data< SeqPositions > position;

    typedef typename core::topology::BaseMeshTopology::Edge Edge;
    typedef typename core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef helper::ReadAccessor<Data< SeqEdges > > raEdges;
    typedef helper::WriteOnlyAccessor<Data< SeqEdges > > waEdges;
    Data< SeqEdges > edges;

    typedef typename core::topology::BaseMeshTopology::Triangle Triangle;
    typedef typename core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef helper::WriteOnlyAccessor<Data< SeqTriangles > > waTriangles;
    Data< SeqTriangles > triangles;

    Data< Real > minLength;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const VoronoiToMeshEngine<ImageTypes>* = NULL) { return ImageTypes::Name();    }

    VoronoiToMeshEngine()    :   Inherited()
      , showMesh(initData(&showMesh,false,"showMesh","show reconstructed mesh"))
      , image(initData(&image,ImageTypes(),"image","Voronoi image"))
      , background(initData(&background,ImageTypes(),"background","Optional Voronoi image of the background to surface details"))
      , transform(initData(&transform,TransformType(),"transform",""))
      , position(initData(&position,SeqPositions(),"position","output positions"))
      , edges(initData(&edges,SeqEdges(),"edges","output edges"))
      , triangles(initData(&triangles,SeqTriangles(),"triangles","output triangles"))
      , minLength(initData(&minLength,(Real)2.,"minLength","minimun edge length in pixels"))
      , time((unsigned int)0)
    {
        image.setReadOnly(true);
        transform.setReadOnly(true);
        f_listening.setValue(true);
    }

    virtual void init()
    {
        addInput(&image);
        addInput(&background);
        addInput(&transform);
        addInput(&minLength);
        addOutput(&position);
        addOutput(&triangles);
        addOutput(&edges);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:

    unsigned int time;

    typedef typename std::set<unsigned int> indSet;  ///< list of indices
    typedef typename indSet::iterator indSetIt;
    typedef std::map<unsigned int, indSet > IDtoInd;  ///< map from point index to a list of indices
    typedef typename IDtoInd::iterator IDtoIndIt;
    typedef std::map<indSet, indSet > indtoInd;  ///< map from a list of indices to a list of indices
    typedef typename indtoInd::iterator indtoIndIt;
    typedef std::map<unsigned int, Coord > IDtoCoord;  ///< map from point index to a 3D coordinate
    typedef typename IDtoCoord::iterator IDtoCoordIt;
    typedef std::map<unsigned int, std::pair<Coord,unsigned int> > IDtoCoordAndUI;


    // count the number of identical values in two sets
    inline unsigned int numIdentical(const indSet& s1,const indSet& s2) const
    {
        unsigned int count=0;
        for(indSetIt v=s1.begin();v!=s1.end();++v)
            if(s2.find(*v)!=s2.end())
                count++;
        return count;
    }


    // remove vertices with one or two edges
    inline bool cleanGraph(IDtoInd& neighbors,IDtoInd& regions) const
    {
        for(IDtoIndIt p=neighbors.begin();p!=neighbors.end();++p)
            if(p->second.size()==1)
            {
                unsigned int n=*p->second.begin();
                neighbors[n].erase(p->first);
                regions.erase(p->first);
                neighbors.erase(p->first);
                return false;
            }
            else if(p->second.size()==2)
            {
                unsigned int n1=*p->second.begin(),n2=*p->second.rbegin();
                neighbors[n1].erase(p->first); neighbors[n1].insert(n2);
                neighbors[n2].erase(p->first); neighbors[n2].insert(n1);
                regions.erase(p->first);
                neighbors.erase(p->first);
                return false;
            }
        return true;
    }

    // remove edges bellow a certain length
    inline bool removeSmallEdges(IDtoInd& neighbors,IDtoInd& regions,IDtoCoord& coords, IDtoCoordAndUI& sums,const Real& tol) const
    {
        Real tol2=tol*tol;
        for(IDtoIndIt p=neighbors.begin();p!=neighbors.end();++p)
            for(indSetIt n=p->second.begin();n!=p->second.end();++n)
                if(p->first<*n)
                {
                    Real d = (coords[p->first] - coords[*n]).norm2();
                    if(d<tol2)
                    {
                        for(indSetIt n2=neighbors[*n].begin();n2!=neighbors[*n].end();++n2) if(p->first!=*n2) { p->second.insert(*n2); neighbors[*n2].insert(p->first); neighbors[*n2].erase(*n);}
                        p->second.erase(*n);
                        neighbors.erase(*n);
                        for(indSetIt r=regions[*n].begin();r!=regions[*n].end();++r) regions[p->first].insert(*r);
                        regions.erase(*n);
                        // accumulate position for a further averaging
                        sums[p->first].first += sums[*n].first;
                        sums[p->first].second += sums[*n].second;
                        return false;
                    }
                }
        return true;
    }


    // order face from a point list and neighbors
    inline bool orderFace(indSet& pts, IDtoInd& neighbors, std::vector<unsigned int>& face) const
    {
        std::map<unsigned int,bool> inserted;
        for(indSetIt p=pts.begin();p!=pts.end();++p) inserted[*p]=false;

        unsigned int currentp = *pts.begin();
        face.push_back(currentp); inserted[currentp]=true;

        bool ok=false;
        while(!ok)
        {
            bool ok2=true;
            for(indSetIt p=pts.begin();p!=pts.end();++p)
                if(!inserted[*p])
                    if(neighbors[currentp].find(*p)!=neighbors[currentp].end())
                    {
                        ok2=false;
                        face.push_back(*p); inserted[*p]=true;
                        currentp=*p;
                    }
            ok=ok2;
        }

        if(/*face.size()==pts.size() &&*/ neighbors[face.front()].find(face.back())!=neighbors[face.front()].end()) return true;
        else return false; // the face is singular (it does not make a loop)
    }


    virtual void update()
    {
        raImage in(this->image);
        raImage inb(this->background);
        raTransform inT(this->transform);

        // get image at time t
        cimg_library::CImg<T> img = in->getCImg(this->time);

        T mx = img.max()+1;

        if(inb->isEmpty())         //fill background with 6 different colors to allow detection of corner points
        {
            cimg_forXYZ(img,x,y,z)
                    if(img(x,y,z)==0)
            {
                int dists[6]={x,img.width()-1-x,y,img.height()-1-y,z,img.depth()-1-z};
                int mn=dists[0]; img(x,y,z)=mx;
                for(unsigned int i=1;i<6;i++) if(dists[i]<mn) { mn=dists[i]; img(x,y,z)=mx+i; }
            }
        }
        else        // use voronoi of the background to add surface details
        {
            cimg_library::CImg<T> bkg = inb->getCImg(this->time);
            cimg_forXYZ(img,x,y,z)
                    if(img(x,y,z)==0)
                        img(x,y,z)=mx+bkg(x,y,z)-1;
        }

        // identify special voxel corners (with more than three neighboring regions)
        cimg_library::CImg<unsigned int> UIimg(img.width(),img.height(),img.depth());
        UIimg.fill(0);

        IDtoInd regions;
        IDtoCoord coords;

        unsigned int count=0;
        cimg_for_insideXYZ(img,x,y,z,1)
        {
            indSet l; for (unsigned int dz=0; dz<2; ++dz) for (unsigned int dy=0; dy<2; ++dy) for (unsigned int dx=0; dx<2; ++dx) l.insert(img(x+dx,y+dy,z+dz));
            if(l.size()>=3)
            {
                regions[count]=l;
                coords[count]=Coord(x+0.5,y+0.5,z+0.5);
                UIimg(x,y,z)=count+1;
                count++;
            }
        }

        // link neighboring vertices sharing 3 regions
        IDtoInd neighbors;
        cimg_for_insideXYZ(UIimg,x,y,z,1)
                if(UIimg(x,y,z)!=0)
        {
            unsigned int p1=UIimg(x,y,z),p2;
            p2=UIimg(x+1,y,z); if(p2!=0) if(numIdentical(regions[p1-1],regions[p2-1])>=3) { neighbors[p1-1].insert(p2-1); neighbors[p2-1].insert(p1-1); }
            p2=UIimg(x-1,y,z); if(p2!=0) if(numIdentical(regions[p1-1],regions[p2-1])>=3) { neighbors[p1-1].insert(p2-1); neighbors[p2-1].insert(p1-1); }
            p2=UIimg(x,y+1,z); if(p2!=0) if(numIdentical(regions[p1-1],regions[p2-1])>=3) { neighbors[p1-1].insert(p2-1); neighbors[p2-1].insert(p1-1); }
            p2=UIimg(x,y-1,z); if(p2!=0) if(numIdentical(regions[p1-1],regions[p2-1])>=3) { neighbors[p1-1].insert(p2-1); neighbors[p2-1].insert(p1-1); }
            p2=UIimg(x,y,z+1); if(p2!=0) if(numIdentical(regions[p1-1],regions[p2-1])>=3) { neighbors[p1-1].insert(p2-1); neighbors[p2-1].insert(p1-1); }
            p2=UIimg(x,y,z-1); if(p2!=0) if(numIdentical(regions[p1-1],regions[p2-1])>=3) { neighbors[p1-1].insert(p2-1); neighbors[p2-1].insert(p1-1); }
        }
        UIimg.clear();

        // iteratively remove vertices with one or two edges
        count = 0;
        while(!cleanGraph(neighbors,regions)) count++;

        // merge points close to each other
        count = 0;
        IDtoCoordAndUI coordsum;
        for(IDtoCoordIt p=coords.begin();p!=coords.end();++p) coordsum[p->first]=std::pair<Coord,unsigned int>(p->second,1);        // intitialize accumulator of coords
        while(!removeSmallEdges(neighbors,regions,coords,coordsum,minLength.getValue())) count++;
        for(IDtoCoordIt p=coords.begin();p!=coords.end();++p) p->second = coordsum[p->first].first/(Real)coordsum[p->first].second;        // average to get mean of merged points
        if(this->f_printLog.getValue()) std::cout<<this->name<<": removed "<<count<<" edges"<<std::endl;

        // reclean
        count = 0;
        while(!cleanGraph(neighbors,regions)) count++;

        // collect face points
        indtoInd facePts;
        for(IDtoIndIt p=regions.begin();p!=regions.end();++p)
        {
            for(indSetIt r=p->second.begin();r!=p->second.end();++r)
                for(indSetIt r2=r;r2!=p->second.end();++r2)
                    if(r2!=r)
                        if(*r2<mx || *r<mx) // discard fictitious corner face
                        {
                            indSet l;
                            l.insert(*r);
                            l.insert(*r2);
                            facePts[l].insert(p->first);
                        }
        }

        // order face points according to edges and remove singular faces
        std::vector<std::vector<unsigned int> > faces;
        count = 0;
        for(indtoIndIt f=facePts.begin();f!=facePts.end();++f)
        {
            // keep points with at least two neighbors on the face
            indSet pts;
            for(indSetIt p=f->second.begin();p!=f->second.end();++p)
                if(numIdentical(f->second,neighbors[*p])>1)
                    pts.insert(*p);

            if(pts.size()>2)
            {
                std::vector<unsigned int> face;
                if(orderFace(pts,neighbors,face)) faces.push_back(face);
                else count++;
            }
            //else count++;
        }
        if(this->f_printLog.getValue()) std::cout<<this->name<<": detected "<<count<<" singular faces"<<std::endl;

        // update neighbors
        neighbors.clear();
        for(unsigned int i=0;i<faces.size();i++)
            for(unsigned int j=0;j<faces[i].size();j++)
            {
                unsigned int p1 = faces[i][j==0?faces[i].size()-1:j-1] , p2 = faces[i][j];
                neighbors[p1].insert(p2);
                neighbors[p2].insert(p1);
            }

        // export points
        typedef std::map<unsigned int,unsigned int > UItoUIMap;
        UItoUIMap indexmap;
        waPositions pos(this->position);
        pos.clear();

        for(IDtoIndIt p=neighbors.begin();p!=neighbors.end();++p)
        {
            indexmap[p->first]=pos.size();
            pos.push_back(inT->fromImage(coords[p->first]));
        }

        // export edges
        waEdges Edges(this->edges);
        Edges.clear();
        for(IDtoIndIt p=neighbors.begin();p!=neighbors.end();++p)
            for(indSetIt n=p->second.begin();n!=p->second.end();++n)
                if(p->first<*n)
                    Edges.push_back(Edge(indexmap[p->first],indexmap[*n]));

        // triangulate by inserting a face centroid
        waTriangles tri(this->triangles);
        tri.clear();
        for(unsigned int i=0;i<faces.size();i++)
        {
            Coord p;   for(unsigned int j=0;j<faces[i].size();j++) p+=pos[indexmap[faces[i][j]]];            p/=(Real)faces[i].size();
            unsigned int index = pos.size();
            pos.push_back(p);
            for(unsigned int j=0;j<faces[i].size();j++) tri.push_back(Triangle( index, indexmap[faces[i][j==0?faces[i].size()-1:j-1]], indexmap[faces[i][j]] ));
        }

        if(this->f_printLog.getValue()) std::cout<<this->name<<": done"<<std::endl;
        cleanDirty();
    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if (simulation::AnimateEndEvent::checkEventType(event))
        {
            raImage in(this->image);
            raTransform inT(this->transform);

            // get current time modulo dimt
            const unsigned int dimt=in->getDimensions()[4];
            if(!dimt) return;
            Real t=inT->toImage(this->getContext()->getTime()) ;
            t-=(Real)((int)((int)t/dimt)*dimt);
            t=(t-floor(t)>0.5)?ceil(t):floor(t); // nearest
            if(t<0) t=0.0; else if(t>=(Real)dimt) t=(Real)dimt-1.0; // clamp

            if(this->time!=(unsigned int)t) { this->time=(unsigned int)t; update(); }
        }
    }

    virtual void draw(const core::visual::VisualParams* vparams)
    {
#ifndef SOFA_NO_OPENGL

        if (!vparams->displayFlags().getShowVisualModels()) return;
        if (!this->showMesh.getValue()) return;

        vparams->drawTool()->drawPoints(this->position.getValue(),5,defaulttype::Vec4f(0.2,1,0.2,1));

        raPositions pos(this->position);
        std::vector<defaulttype::Vector3> points;
        raEdges Edges(this->edges);
        points.resize(2*Edges.size());
        for (unsigned int i=0; i<Edges.size(); ++i)
        {
            points[2*i][0]=pos[Edges[i][0]][0];            points[2*i][1]=pos[Edges[i][0]][1];            points[2*i][2]=pos[Edges[i][0]][2];
            points[2*i+1][0]=pos[Edges[i][1]][0];          points[2*i+1][1]=pos[Edges[i][1]][1];          points[2*i+1][2]=pos[Edges[i][1]][2];
        }
        vparams->drawTool()->drawLines(points,2.0,defaulttype::Vec4f(0.7,1,0.7,1));

#endif /* SOFA_NO_OPENGL */
    }
};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_VoronoiToMeshENGINE_H
