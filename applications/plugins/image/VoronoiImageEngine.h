/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_IMAGE_VoronoiImageEngine_H
#define SOFA_IMAGE_VoronoiImageEngine_H

#include <image/config.h>
#include "ImageTypes.h"
#include "ImageAlgorithms.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/OptionsGroup.h>


#define FASTMARCHING 0
#define DIJKSTRA 1

namespace sofa
{

namespace component
{

namespace engine
{



/// Default implementation does not compile
template <class ImageType>
struct VoronoiImageEngineSpecialization
{
};

/// forward declaration
template <class ImageType> class VoronoiImageEngine;


/// Specialization for regular Image
template <class T>
struct VoronoiImageEngineSpecialization<defaulttype::Image<T>>
{
    typedef VoronoiImageEngine<defaulttype::Image<T>> VoronoiImageEngineT;

    typedef defaulttype::Image<SReal> DistTypes;
    typedef defaulttype::Image<unsigned int> VorTypes;

    static void init( VoronoiImageEngineT* /*engine*/ )
    {
    }

    static void compute( VoronoiImageEngineT* engine,const bool bias=false, const unsigned int method=DIJKSTRA )
    {
        typedef typename VoronoiImageEngineT::Real Real;
        typedef typename VoronoiImageEngineT::Coord Coord;

        // get transform and image at time t
        typename VoronoiImageEngineT::raImage in(engine->d_image);
        typename VoronoiImageEngineT::raTransform inT(engine->d_transform);
        const cimg_library::CImg<T>& inimg = in->getCImg(engine->time);
        const cimg_library::CImg<T>* biasFactor=bias?&inimg:NULL;

        // data access
        typename VoronoiImageEngineT::raPositions fpos(engine->d_position);
        typename VoronoiImageEngineT::imCoord dim = in->getDimensions();

        // init voronoi and distances
        dim[3]=dim[4]=1;
        typename VoronoiImageEngineT::waVor vorData(engine->d_voronoi);
        vorData->setDimensions(dim);
        cimg_library::CImg<unsigned int>& voronoi = vorData->getCImg(); voronoi.fill(0);
        typename VoronoiImageEngineT::waDist distData(engine->d_distances);
        distData->setDimensions(dim);
        cimg_library::CImg<Real>& dist = distData->getCImg(); dist.fill(-1);
        cimg_forXYZC(inimg,x,y,z,c) if(inimg(x,y,z,c)) dist(x,y,z)=cimg_library::cimg::type<Real>::max();

        // list of seed points
        std::set<std::pair<Real,sofa::defaulttype::Vec<3,int> > > trial;

        // add fixed points
        helper::vector<unsigned int> fpos_voronoiIndex;
        helper::vector<Coord> fpos_VoxelIndex;

        for(unsigned int i=0; i<fpos.size(); i++)
        {
            fpos_voronoiIndex.push_back(i+1);
            fpos_VoxelIndex.push_back(inT->toImage(fpos[i]));
            AddSeedPoint<Real>(trial,dist,voronoi, fpos_VoxelIndex[i],fpos_voronoiIndex[i]);
        }

        switch(method)
        {
        case FASTMARCHING : fastMarching<Real,T>(trial,dist, voronoi, engine->d_transform.getValue().getScale(),biasFactor ); break;
        case DIJKSTRA : dijkstra<Real,T>(trial,dist, voronoi, engine->d_transform.getValue().getScale(), biasFactor); break;
        default : engine->serr << "Unknown Distance Field Computation Method" << engine->sendl; break;
        };
    }

};




/**
 * This class computes a voronoi image from a set of points inside an image mask
 */


template <class _ImageTypes>
class VoronoiImageEngine : public core::DataEngine
{
    friend struct VoronoiImageEngineSpecialization<_ImageTypes>;

public:

    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(VoronoiImageEngine,_ImageTypes),Inherited);

    typedef SReal Real;

    //@name Image data
    /**@{*/
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > d_image;
    /**@}*/

    //@name Transform data
    /**@{*/
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > d_transform;
    /**@}*/

    //@name option data
    /**@{*/
    Data<bool> d_bias;
    Data<helper::OptionsGroup> d_method;
    /**@}*/

    //@name sample data (points+connectivity)
    /**@{*/
    typedef helper::vector<defaulttype::Vec<3,Real> > SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    Data< SeqPositions > d_position;
    /**@}*/

    //@name distances (may be used for shape function computation)
    /**@{*/
    typedef typename VoronoiImageEngineSpecialization<ImageTypes>::DistTypes DistTypes;
    typedef helper::WriteOnlyAccessor<Data< DistTypes > > waDist;
    Data< DistTypes > d_distances;
    Data<bool> d_clearData;
    /**@}*/

    //@name voronoi
    /**@{*/
    typedef typename VoronoiImageEngineSpecialization<ImageTypes>::VorTypes VorTypes;
    typedef helper::WriteOnlyAccessor<Data< VorTypes > > waVor;
    Data< VorTypes > d_voronoi;
    /**@}*/


    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const VoronoiImageEngine<ImageTypes>* = NULL) { return ImageTypes::Name();    }
    VoronoiImageEngine()    :   Inherited()
      , d_image(initData(&d_image,ImageTypes(),"image",""))
      , d_transform(initData(&d_transform,TransformType(),"transform",""))
      , d_bias ( initData ( &d_bias,"bias","bias distances using image intensities?" ) )
      , d_method ( initData ( &d_method,"method","method" ) )
      , d_position(initData(&d_position,SeqPositions(),"position","output positions"))
      , d_distances(initData(&d_distances,DistTypes(),"distances",""))
      , d_clearData(initData(&d_clearData,true,"clearData","clear distance image after computation"))
      , d_voronoi(initData(&d_voronoi,VorTypes(),"voronoi",""))
      , time((unsigned int)0)
    {
        d_image.setReadOnly(true);
        d_transform.setReadOnly(true);
        f_listening.setValue(true);

        helper::OptionsGroup methodOptions(2,"FastMarching",
                                           "Dijkstra"
                                           );
        methodOptions.setSelectedItem(DIJKSTRA);
        d_method.setValue(methodOptions);

        VoronoiImageEngineSpecialization<ImageTypes>::init( this );
    }

    virtual void init()
    {
        addInput(&d_image);
        addInput(&d_transform);
        addInput(&d_position);
        addInput(&d_method);
        addInput(&d_bias);
        addOutput(&d_distances);
        addOutput(&d_voronoi);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:

    unsigned int time;

    virtual void update()
    {
        updateAllInputsIfDirty(); // easy to ensure that all inputs are up-to-date

        cleanDirty();

        const bool bias=d_bias.getValue();
        const unsigned int method = this->d_method.getValue().getSelectedId();

        VoronoiImageEngineSpecialization<ImageTypes>::compute( this, bias, method );

        // clear distance image ?
        if(this->d_clearData.getValue())
        {
            waDist dist(this->d_distances); dist->clear();
        }
    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if (simulation::AnimateEndEvent::checkEventType(event))
        {
            raImage in(this->d_image);
            raTransform inT(this->d_transform);

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

    virtual void draw(const core::visual::VisualParams* /*vparams*/)
    {
    }




};



} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_VoronoiImageEngine_H
