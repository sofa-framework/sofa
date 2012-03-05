/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_IMAGE_IMAGESAMPLER_H
#define SOFA_IMAGE_IMAGESAMPLER_H

#include "initImage.h"
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateEndEvent.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/rmath.h>

#include <omp.h>

#define REGULAR 0
#define LLOYD 1


namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;
using namespace helper;
using namespace core::topology;

/**
 * This class samples an object represented by an image
 */


template <class _ImageTypes>
class ImageSampler : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageSampler,_ImageTypes),Inherited);

    typedef SReal Real;

    //@name Image data
    /**@{*/
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;
    /**@}*/

    //@name Transform data
    /**@{*/
    typedef ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;
    /**@}*/

    //@name option data
    /**@{*/
    typedef vector<double> ParamTypes;
    typedef helper::ReadAccessor<Data< ParamTypes > > raParam;

    Data<OptionsGroup> method;
    Data< bool > computeRecursive;
    Data< ParamTypes > param;
    /**@}*/

    //@name sample data
    /**@{*/
    typedef vector<Vec<3,Real> > SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    typedef helper::WriteAccessor<Data< SeqPositions > > waPositions;
    Data< SeqPositions > position;
    Data< SeqPositions > fixedPosition;

    typedef typename BaseMeshTopology::Edge Edge;
    typedef typename BaseMeshTopology::SeqEdges SeqEdges;
    typedef helper::ReadAccessor<Data< SeqEdges > > raEdges;
    typedef helper::WriteAccessor<Data< SeqEdges > > waEdges;
    Data< SeqEdges > edges;
    Data< SeqEdges > graph;

    typedef typename BaseMeshTopology::Hexa Hexa;
    typedef typename BaseMeshTopology::SeqHexahedra SeqHexahedra;
    typedef helper::ReadAccessor<Data< SeqHexahedra > > raHexa;
    typedef helper::WriteAccessor<Data< SeqHexahedra > > waHexa;
    Data< SeqHexahedra > hexahedra;
    /**@}*/

    //@name visu data
    /**@{*/
    Data< bool > showSamples;
    Data< bool > showEdges;
    Data< bool > showGraph;
    /**@}*/

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ImageSampler<ImageTypes>* = NULL) { return ImageTypes::Name();    }
    ImageSampler()    :   Inherited()
        , image(initData(&image,ImageTypes(),"image",""))
        , transform(initData(&transform,TransformType(),"transform",""))
        , method ( initData ( &method,"method","method (param)" ) )
        , computeRecursive(initData(&computeRecursive,false,"computeRecursive","if true: insert nodes recursively and build the graph"))
        , param ( initData ( &param,"param","Parameters" ) )
        , position(initData(&position,SeqPositions(),"position","output positions"))
        , fixedPosition(initData(&fixedPosition,SeqPositions(),"fixedPosition","user defined sample positions"))
        , edges(initData(&edges,SeqEdges(),"edges","edges connecting neighboring nodes"))
        , graph(initData(&graph,SeqEdges(),"graph","graph where each node is connected to its neighbors when created"))
        , hexahedra(initData(&hexahedra,SeqHexahedra(),"hexahedra","output hexahedra"))
        , showSamples(initData(&showSamples,false,"showSamples","show samples"))
        , showEdges(initData(&showEdges,false,"showEdges","show edges"))
        , showGraph(initData(&showGraph,false,"showGraph","show graph"))
        , time((unsigned int)0)
    {
        image.setReadOnly(true);
        transform.setReadOnly(true);
        f_listening.setValue(true);

        helper::OptionsGroup methodOptions(2,"0 - Regular sampling (at voxel center(0) or corners (1)) "
                ,"1 - Uniform sampling using Lloyd relaxation (nbSamples | bias distances? | nbiterations=100)"
                                          );
        methodOptions.setSelectedItem(REGULAR);
        method.setValue(methodOptions);
    }

    virtual void init()
    {
        addInput(&image);
        addInput(&transform);
        addInput(&fixedPosition);
        addOutput(&position);
        addOutput(&edges);
        addOutput(&graph);
        addOutput(&hexahedra);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:

    unsigned int time;

    virtual void update()
    {
        cleanDirty();

        raParam params(this->param);

        if(this->method.getValue().getSelectedId() == REGULAR)
        {
            // get params
            bool atcorners=false; if(params.size())   atcorners=(bool)params[0];

            // sampling
            if(!computeRecursive.getValue()) regularSampling (atcorners);
        }
        else if(this->method.getValue().getSelectedId() == LLOYD)
        {
            // get params
            unsigned int nb=0;        if(params.size())       nb=(unsigned int)params[0];
            bool bias=false;          if(params.size()>1)     bias=(bool)params[1];
            unsigned int lloydIt=100; if(params.size()>2)     lloydIt=(unsigned int)params[2];

            // sampling
            uniformSampling(nb,bias,lloydIt);
        }

        if(this->f_printLog.getValue()) if(this->position.getValue().size())    std::cout<<"ImageSampler: "<< this->position.getValue().size() <<" generated samples"<<std::endl;
        if(this->f_printLog.getValue()) if(this->edges.getValue().size())       std::cout<<"ImageSampler: "<< this->edges.getValue().size() <<" generated edges"<<std::endl;
        if(this->f_printLog.getValue()) if(this->hexahedra.getValue().size())   std::cout<<"ImageSampler: "<< this->hexahedra.getValue().size() <<" generated hexahedra"<<std::endl;
    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if ( dynamic_cast<simulation::AnimateEndEvent*>(event))
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
        if (!vparams->displayFlags().getShowVisualModels()) return;

        raPositions pos(this->position);
        raEdges e(this->edges);
        raEdges g(this->graph);

        if (this->showSamples.getValue()) vparams->drawTool()->drawPoints(this->position.getValue(),5.0,Vec4f(0.2,1,0.2,1));
        if (this->showSamples.getValue()) vparams->drawTool()->drawPoints(this->fixedPosition.getValue(),7.0,Vec4f(1,0.2,0.2,1));
        if (this->showEdges.getValue())
        {
            std::vector<Vector3> points;
            points.resize(2*e.size());
            for (unsigned int i=0; i<e.size(); ++i)
            {
                points[2*i][0]=pos[e[i][0]][0];            points[2*i][1]=pos[e[i][0]][1];            points[2*i][2]=pos[e[i][0]][2];
                points[2*i+1][0]=pos[e[i][1]][0];          points[2*i+1][1]=pos[e[i][1]][1];          points[2*i+1][2]=pos[e[i][1]][2];
            }
            vparams->drawTool()->drawLines(points,2.0,Vec4f(0.7,1,0.7,1));
        }
        if (this->showGraph.getValue())
        {
            std::vector<Vector3> points;
            points.resize(2*g.size());
            for (unsigned int i=0; i<g.size(); ++i)
            {
                points[2*i][0]=pos[g[i][0]][0];            points[2*i][1]=pos[g[i][0]][1];            points[2*i][2]=pos[g[i][0]][2];
                points[2*i+1][0]=pos[g[i][1]][0];          points[2*i+1][1]=pos[g[i][1]][1];          points[2*i+1][2]=pos[g[i][1]][2];
            }
            vparams->drawTool()->drawLines(points,2.0,Vec4f(1,1,0.5,1));
        }
    }


    /**
    * put regularly spaced samples at each non empty voxel center or corners
    * generated topology: edges + hexahedra
    * @param atcorners : put samples at voxel corners instead of centers ?
    */
    void regularSampling ( bool atcorners=false )
    {
        // get tranform and image at time t
        raImage in(this->image);
        raTransform inT(this->transform);
        const CImg<T>& inimg = in->getCImg(this->time);

        // data access
        waPositions pos(this->position);       pos.clear();
        waEdges e(this->edges);                e.clear();
        waEdges g(this->graph);                g.clear();
        waHexa h(this->hexahedra);             h.clear();

        // convert to single channel boolean image
        CImg<bool> img(inimg.width(),inimg.height(),inimg.depth(),1,false);
        if(atcorners)
        {
            CImg_2x2x2(I,bool);
            cimg_for2x2x2(inimg,x,y,z,0,I,bool) if(Iccc || Iccn || Icnc || Incc || Innc || Incn || Icnn || Innn) img(x,y,z)=true;
        }
        else cimg_forXYZC(inimg,x,y,z,c) if(inimg(x,y,z,c)) img(x,y,z)=true;

        // count non empty voxels
        unsigned int nb=0;
        cimg_foroff(img,off) if(img[off]) nb++;
        pos.resize(nb);
        // record indices of previous y line and z plane for connectivity
        CImg<unsigned int> pLine(inimg.height()),nLine(inimg.height());
        CImg<unsigned int> pPlane(inimg.width(),inimg.height()),nPlane(inimg.width(),inimg.height());
        // fill pos and edges
        nb=0;
        cimg_forZ(img,z)
        {
            cimg_forY(img,y)
            {
                cimg_forX(img,x)
                {
                    if(img(x,y,z))
                    {
                        // pos
                        if(atcorners) pos[nb]=inT->fromImage(Coord(x+0.5,y+0.5,z+0.5));
                        else pos[nb]=inT->fromImage(Coord(x,y,z));
                        // edges
                        if(x) if(img(x-1,y,z)) e.push_back(Edge(nb-1,nb));
                        if(y) if(img(x,y-1,z)) e.push_back(Edge(pLine(x),nb));
                        if(z) if(img(x,y,z-1)) e.push_back(Edge(pPlane(x,y),nb));
                        // hexa
                        if(x && y && z) if(img(x-1,y,z) && img(x,y-1,z) && img(x,y,z-1) && img(x-1,y-1,z) && img(x-1,y,z-1)  && img(x,y-1,z-1)   && img(x-1,y-1,z-1) )
                                h.push_back(Hexa(nb,pLine(x),pLine(x-1),nb-1,pPlane(x,y),pPlane(x,y-1),pPlane(x-1,y-1),pPlane(x-1,y) ));

                        nLine(x)=nb; nPlane(x,y)=nb;
                        nb++;
                    }
                }
                nLine.swap(pLine);
            }
            nPlane.swap(pPlane);
        }
    }


    /**
    * computes a uniform sample distribution (=point located at the center of their Voronoi cell) based on farthest point sampling + Lloyd (=kmeans) relaxation
    * generated topology: edges
    * @param nb : target number of samples
    * @param bias : bias distances using the input image ?
    * @param lloydIt : maximum number of Lloyd iterations.
    */

    void uniformSampling ( unsigned int nb=0,  bool bias=false, unsigned int lloydIt=100)
    {
        // get tranform and image at time t
        raImage in(this->image);
        raTransform inT(this->transform);
        const CImg<T>& inimg = in->getCImg(this->time);

        // data access
        raPositions fpos(this->fixedPosition);
        waPositions pos(this->position);       pos.clear();
        waEdges e(this->edges);                e.clear();
        waEdges g(this->graph);                g.clear();
        waHexa h(this->hexahedra);             h.clear();

        // init voronoi and distances
        CImg<unsigned int>  voronoi(inimg.width(),inimg.height(),inimg.depth(),1,0);
        CImg<Real>          distances(inimg.width(),inimg.height(),inimg.depth(),1,-1);
        cimg_forXYZC(inimg,x,y,z,c) if(inimg(x,y,z,c)) distances(x,y,z)=cimg::type<Real>::max();

        // farthest point sampling using geodesic distances
        while(pos.size()<nb)
        {
            Real dmax=0;  Coord pmax;
            cimg_forXYZ(distances,x,y,z) if(distances(x,y,z)>dmax) { dmax=distances(x,y,z); pmax =Coord(x,y,z); }
            if(dmax)
            {
                pos.push_back(inT->fromImage(pmax));
                dijkstra (distances, voronoi,  bias);
            }
            else break;
        }
        //voronoi.display();

        unsigned int it=0;
        bool converged =false;
        while(!converged)
        {
            if(Lloyd(distances,voronoi,bias))
            {
                cimg_foroff(distances,off) if(distances[off]!=-1) distances[off]=cimg::type<Real>::max();
                dijkstra (distances, voronoi,  bias);
                it++; if(it==lloydIt) converged=true;
            }
            else converged=true;
        }
        if(this->f_printLog.getValue()) std::cout<<"ImageSampler: Completed "<< it <<" Lloyd iterations"<<std::endl;

    }



    /**
    *  Move points in position data to the centroid of their voronoi region
    *  centroid are computed given a (biased) distance measure d as $p + (b)/N \sum_i d(p,p_i)*2/((b)+(b_i))*(p_i-p)/|p_i-p|$
    *  with no bias, we obtain the classical mean $1/N \sum_i p_i$
    * returns true if points have moved
    */

    bool Lloyd (CImg<Real>& distances, CImg<unsigned int>& voronoi,  bool bias=false )
    {
        waPositions pos(this->position);
        raTransform inT(this->transform);

        unsigned int nbp=pos.size();
        bool moved=false;

        // get rounded point coordinates in image (to check that points do not share the same voxels)
        std::vector<Coord> P;
        P.resize(nbp);
        for (unsigned int i=0; i<nbp; i++) { Coord p = inT->toImage(pos[i]);  for (unsigned int j=0; j<3; j++)  P[i][j]=round(p[j]); }

        #pragma omp parallel for
        for (unsigned int i=0; i<nbp; i++)
        {
            // compute centroid
            Coord c,p,u;
            unsigned int count=0;
            bool valid=true;
            cimg_forXYZ(voronoi,x,y,z) if (voronoi(x,y,z)==i+1)
            {
                p=inT->fromImage(Coord(x,y,z));
                u=p-pos[i]; u.normalize();
                c+=u*distances(x,y,z);
                count++;
            }
            if(!count) goto stop;

            c/=(Real)count;

            if (bias)
            {
                //                Real stiff=getStiffness(grid.data()[indices[i]]);
                //                if(biasFactor!=(Real)1.) stiff=(Real)pow(stiff,biasFactor);
                //                c*=stiff;
            }

            c+=pos[i];

            // check validity
            p = inT->toImage(c); for (unsigned int j=0; j<3; j++) p[j]=round(p[j]);
            if (distances(p[0],p[1],p[2])==-1) valid=false; // out of object
            else { for (unsigned int j=0; j<nbp; j++) if(i!=j) if(P[j][0]==p[0]) if(P[j][1]==p[1]) if(P[j][2]==p[2]) valid=false; } // check occupancy

            while(!valid)  // get closest unoccupied point in voronoi
            {
                Real dmin=cimg::type<Real>::max();
                cimg_forXYZ(voronoi,x,y,z) if (voronoi(x,y,z)==i+1)
                {
                    Coord pi=inT->fromImage(Coord(x,y,z));
                    Real d2=(c-pi).norm2();
                    if(dmin>d2) { dmin=d2; p=Coord(x,y,z); }
                }
                if(dmin==cimg::type<Real>::max()) goto stop;// no point found
                bool val2=true; for (unsigned int j=0; j<nbp; j++) if(i!=j) if(P[j][0]==p[0]) if(P[j][1]==p[1]) if(P[j][2]==p[2]) val2=false; // check occupancy
                if(val2) valid=true;
                else voronoi(p[0],p[1],p[2])=0;
            }

            if(P[i][0]!=p[0] || P[i][1]!=p[1] || P[i][2]!=p[2]) // set new position if different
            {
                pos[i] = inT->fromImage(p);
                for (unsigned int j=0; j<3; j++) P[i][j]=p[j];
                moved=true;
            }
stop: ;
        }

        return moved;
    }


    /**
    * Computes geodesic distances in the image from position+fixedPosition up to @param distMax, given a bias distance function F.
    * This is equivalent to solve for the eikonal equation || grad d_ijk || = F_ijk with d_ijk=0 at positions
    * using fast marching method presented in http://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-658.html
    * distances should be intialized (-1 outside the object, and >0 inside)
    * returns @param voronoi and @param distances
    */

    void fastMarching (CImg<Real>& /*distances*/, CImg<unsigned int>& /*voronoi*/,  bool /*bias=false*/, const Real /*distMax=cimg::type<Real>::max()*/)
    {
    }


    /**
    * Computes geodesic distances in the image from position and fixedPosition data up to @param distMax, given a bias distance function F.
    * This is equivalent to solve for the eikonal equation || grad d_ijk || = F_ijk with d_ijk=0 at positions
    * using dijkstra minimum path algorithm
    * distances should be intialized (-1 outside the object, and >0 inside)
    * returns @param voronoi and @param distances
    */

    void dijkstra (CImg<Real>& distances, CImg<unsigned int>& voronoi,  bool bias=false, const Real distMax=cimg::type<Real>::max())
    {
        raPositions fpos(this->fixedPosition);
        raPositions pos(this->position);
        raTransform inT(this->transform);

        unsigned int nbp=pos.size(),nbfp=fpos.size();

        // get rounded point coordinates in image
        std::vector<Vec<3,int> > P;
        P.resize(nbp+nbfp);
        for (unsigned int i=0; i<nbp; i++)  { Coord p = inT->toImage(pos[i]);  for (unsigned int j=0; j<3; j++)  P[i][j]=round(p[j]); }
        for (unsigned int i=0; i<nbfp; i++) { Coord p = inT->toImage(fpos[i]); for (unsigned int j=0; j<3; j++)  P[i+nbp][j]=round(p[j]); }

        // init
        CImg_3x3x3(D,Real); // cimg neighborhood for distances
        CImg_3x3x3(V,unsigned int); // cimg neighborhood for voronoi
        Vec<27, Vec<3,int> > offset; // image coord offsets related to neighbors
        int count=0; for (int k=-1; k<=1; k++) for (int j=-1; j<=1; j++) for (int i=-1; i<=1; i++) offset[count++]=Vec<3,int>(i,j,k);

        CImg<Real> lD(3,3,3);  // local distances
        if(!bias) // precompute local distances (supposing that the transformation is not projective)
        {
            lD(1,1,1)=0;
            lD(2,1,1) = lD(0,1,1) = inT->getScale()[0];
            lD(1,2,1) = lD(1,0,1) = inT->getScale()[1];
            lD(1,1,2) = lD(1,1,0) = inT->getScale()[2];
            lD(2,2,1) = lD(0,2,1) = lD(2,0,1) = lD(0,0,1) = sqrt(inT->getScale()[0]*inT->getScale()[0] + inT->getScale()[1]*inT->getScale()[1]);
            lD(2,1,2) = lD(0,1,2) = lD(2,1,0) = lD(0,1,0) = sqrt(inT->getScale()[0]*inT->getScale()[0] + inT->getScale()[2]*inT->getScale()[2]);
            lD(1,2,2) = lD(1,0,2) = lD(1,2,0) = lD(1,0,0) = sqrt(inT->getScale()[2]*inT->getScale()[2] + inT->getScale()[1]*inT->getScale()[1]);
            lD(2,2,2) = lD(2,0,2) = lD(2,2,0) = lD(2,0,0) = lD(0,2,2) = lD(0,0,2) = lD(0,2,0) = lD(0,0,0) = sqrt(inT->getScale()[0]*inT->getScale()[0] + inT->getScale()[1]*inT->getScale()[1] + inT->getScale()[2]*inT->getScale()[2]);
        }

        // add samples t the queue
        typedef std::pair<Real,Vec<3,int> > DistanceToPoint;
        typedef std::set<DistanceToPoint>::iterator ITER;
        std::set<DistanceToPoint> q; // priority queue
        for (unsigned int i=0; i<nbp; i++)
        {
            if(distances.containsXYZC(P[i][0],P[i][1],P[i][2]))
                if(distances(P[i][0],P[i][1],P[i][2])!=-1)
                {
                    q.insert( DistanceToPoint(0.,P[i]) );
                    distances(P[i][0],P[i][1],P[i][2])=0;
                    voronoi(P[i][0],P[i][1],P[i][2])=i+1;
                }
        }

        // dijkstra
        while( !q.empty() )
        {
            DistanceToPoint top = *q.begin();
            q.erase(q.begin());
            Vec<3,int> v = top.second;

            int x = v[0] ,y = v[1] ,z = v[2];
            const int _p1x = x?x-1:x, _p1y = y?y-1:y, _p1z = z?z-1:z, _n1x = x<distances.width()-1?x+1:x, _n1y = y<distances.height()-1?y+1:y, _n1z = z<distances.depth()-1?z+1:z;    // boundary conditions for cimg neighborood manipulation macros

            cimg_get3x3x3(distances,x,y,z,0,D,Real); // get distances in neighborhood
            cimg_get3x3x3(voronoi,x,y,z,0,V,unsigned int); // get voronoi in neighborhood

            if(bias) { }  // TO DO!!!   define lD for biased distances

            for (unsigned int i=0; i<27; i++)
            {
                Real d = Dccc + lD[i];
                if(D[i] > d )
                {
                    Vec<3,int> v2 = v + offset[i];
                    if(distances.containsXYZC(v2[0],v2[1],v2[2]))
                    {
                        if(D[i] < distMax) { ITER it=q.find(DistanceToPoint(D[i],v2)); if(it!=q.end()) q.erase(it); }
                        voronoi(v2[0],v2[1],v2[2]) = Vccc;
                        distances(v2[0],v2[1],v2[2]) = d;
                        q.insert( DistanceToPoint(d,v2) );
                    }
                }
            }
        }
    }


};

} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_IMAGESAMPLER_H
