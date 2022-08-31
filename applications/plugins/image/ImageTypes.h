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

#ifndef IMAGE_IMAGETYPES_H
#define IMAGE_IMAGETYPES_H

#include <image/config.h>

#if  defined (SOFA_HAVE_FFMPEG)  || defined (SOFA_EXTLIBS_FFMPEG)
#define cimg_use_ffmpeg
#endif
#ifdef SOFA_IMAGE_HAVE_OPENCV // should be "SOFA_HAVE_OPENCV" -> use "SOFA_IMAGE_HAVE_OPENCV" until the opencv plugin is fixed..
#define cimg_use_opencv
#endif

#include <CImgPlugin/SOFACImg.h>

#include "VectorVis.h"

//(imports + Image data structure + others) are in here
#include <image/CImgData.h>

#include <sofa/component/visual/VisualModelImpl.h>
#include <sofa/component/visual/VisualStyle.h>

namespace sofa
{

namespace defaulttype
{


//-----------------------------------------------------------------------------------------------//
// Histogram structure
//-----------------------------------------------------------------------------------------------//

template<typename _T>
struct Histogram
{
    typedef _T T;
    typedef Image<T> ImageTypes;

protected:
    const ImageTypes* img;

    unsigned int dimx;		// input number of bins
    unsigned int dimy;		// input histogram image height
    bool mergeChannels;		// histogram of norm ?

    double scaleVal;	double offsetVal;		// output histo abscisse to intensity transfer function :  intensity = x * scaleVal + offsetVal

    cimg_library::CImg<unsigned int> histogram;	// output image of size [dimx,1,1,nbChannels]
    cimg_library::CImg<bool> image;				// output image of size [dimx,dimy,1,nbChannels]

    type::Vec<2,T> clamp;					// stored clamp values (for visualization)

public:
    static const char* Name() { return "Histogram"; }

    Histogram(const unsigned int _dimx=256, const unsigned int _dimy=256, const bool _mergeChannels=false)
        :img(NULL),dimx(_dimx),dimy(_dimy),mergeChannels(_mergeChannels),scaleVal(0.0),offsetVal(0.0),
          clamp(type::Vec<2,T>(cimg_library::cimg::type<T>::min(),cimg_library::cimg::type<T>::max()))
    { }

    void setInput(const ImageTypes& _img)
    {
        img=&_img;
        update();
    }

    const cimg_library::CImg<bool>& getImage() const {return image;}
    const cimg_library::CImg<unsigned int>& getHistogram() const {return histogram;}
    const type::Vec<2,T>& getClamp() const {return clamp;}
    void setClamp(const type::Vec<2,T> _clamp)  { clamp[0] = _clamp[0]; clamp[1] = _clamp[1];	}
    const bool& getMergeChannels() const {return this->mergeChannels;}
    void setMergeChannels(const bool _mergeChannels)
    {
        if(this->mergeChannels==_mergeChannels) return;
        this->mergeChannels=_mergeChannels;
        this->setClamp(type::Vec<2,T>(cimg_library::cimg::type<T>::min(),cimg_library::cimg::type<T>::max()));
        this->update();
    }

    void update()
    {
        if(!img) return;
        if(!img->getCImgList().size()) return;

        T vmin,vmax;
        histogram = img->get_histogram(vmin,vmax,dimx,mergeChannels);
        //if(clamp[1]<vmin || clamp[1]>vmax)
        clamp[1]=vmax;
        //if(clamp[0]<vmin || clamp[0]>vmax)
        clamp[0]=vmin;

        offsetVal = (double)vmin;
        scaleVal = (double)(vmax-vmin)/(double)(dimx-1);
        image = cimg_library::CImg<bool>(dimx,dimy,1,histogram.spectrum(),0);
        bool tru=true;
        cimg_forC(histogram,c) image.get_shared_channel(c).draw_graph(histogram.get_shared_channel(c),&tru,1,3,0);

    }

    T fromHistogram(const unsigned int i) const {return (T)(scaleVal*(double)i + offsetVal); }
    unsigned int toHistogram(const T i) const {return (unsigned int)( ((double)i - offsetVal)/scaleVal ); }

    inline friend std::istream& operator >> ( std::istream& in, Histogram& h )
    {
        type::Vec<2,T> myclamp;
        in>>myclamp;
        h.setClamp(myclamp);
        return in;
    }

    friend std::ostream& operator << ( std::ostream& out, const Histogram& h )
    {
        out<<h.getClamp();
        return out;
    }

};


//-----------------------------------------------------------------------------------------------//
// Image plane selector (to be embedded into a Data, and visualized with ImagePlaneWidget)
//-----------------------------------------------------------------------------------------------//

template<typename _T>
struct ImagePlane
{
    typedef _T T;
    typedef Image<T> ImageTypes;
    typedef typename ImageTypes::imCoord imCoord;
    typedef type::Vec<3,unsigned int> pCoord;

    typedef SReal Real; // put as template param ?
    typedef ImageTransform<Real> TransformTypes;
    typedef typename TransformTypes::Coord Coord;

    typedef typename sofa::component::visual::VisualModelImpl VisualModelTypes;
    typedef std::vector<VisualModelTypes*> VecVisualModel;

    //    const VectorVis* vectorvis; //! A reference to the VectorVis data allows the plane images to switch between RGB or greyscale norms, as the user changes the options in the GUI

protected:
    const ImageTypes* img;				// input image
    VecVisualModel visualModels;		// input models to draw on images
    const TransformTypes* transform;	// input transform

    Coord point;				// Point double-clicked on slice 3D for navigation
    pCoord plane;				// input [x,y,z] coord of a selected planes. >=dimensions means no selection
    unsigned int time;			// input selected time
    type::Vec<2,T> clamp;				// input clamp values

    bool newPointClicked;		// True when a point is double-clicked on an image plane
    bool imagePlaneDirty;			// Dirty when output plane images should be updated
    bool mergeChannels;		// multichannel image or norm ?

public:
    static const char* Name() { return "ImagePlane"; }

    ImagePlane()
        :img(NULL), plane(pCoord(0,0,0)), time(0), clamp(type::Vec<2,T>(cimg_library::cimg::type<T>::min(),cimg_library::cimg::type<T>::max()))
        , newPointClicked(false), imagePlaneDirty(true), mergeChannels(false)//, point(0,0,0) // set by user or other objects
    {
    }

    void setInput(const ImageTypes& _img,const TransformTypes& _transform, const VecVisualModel& _visualModels)
    {
        transform=&_transform;
        img=&_img;
        visualModels.assign(_visualModels.begin(),_visualModels.end());
        //        this->setPlane(pCoord(this->img->getDimensions()[0]/2,this->img->getDimensions()[1]/2,this->img->getDimensions()[2]/2));
        //        this->imagePlaneDirty=true;
    }

    const pCoord& getPlane() const {return plane;}
    const unsigned int& getTime() const {return time;}
    const type::Vec<2,T>& getClamp() const {return clamp;}
    imCoord getDimensions() const {  if(!this->img)  { imCoord c; c.fill(0); return c;} else return img->getDimensions(); }
    const bool& isImagePlaneDirty() const {return this->imagePlaneDirty;}
    const bool& isnewPointClicked() const {return this->newPointClicked;}

    const bool& getMergeChannels() const {return this->mergeChannels;}
    void setMergeChannels(const bool _mergeChannels)  {   this->mergeChannels=_mergeChannels;    }

    void setNewPoint(const Coord& newPoint)
    {
        point = newPoint;
        this->newPointClicked=true;
    }

    const Coord& getNewPoint() const
    {
        return this->point;
    }

    void setPlane(const Coord& p)
    {
        bool different=false;
        for(unsigned int i=0; i<3; i++) if(plane[i]!=p[i]) { plane[i]=p[i]; different=true; }
        if(different) this->imagePlaneDirty=true;
    }

    void setTime(const Real t, bool repeat=true)
    {
        if(!this->img )  return;
        unsigned int size = this->img->getCImgList().size();
        if(!t || !this->transform) return;
        Real t2=this->transform->toImage(t) ;
        if(repeat) t2-=(Real)((int)((int)t2/size)*size);
        t2=(t2-floor(t2)>0.5)?ceil(t2):floor(t2); // nearest
        if(t2<0) t2=0.0; else if(t2>=(Real)size) t2=(Real)size-1.0; // clamp
        if(this->time!=(unsigned int)t2)
        {
            this->time=(unsigned int)t2;
            this->imagePlaneDirty=true;
        }
    }


    void setClamp(const type::Vec<2,T> _clamp)
    {
        if(clamp[0]!=_clamp[0] || clamp[1]!=_clamp[0])
        {
            clamp=_clamp;
            this->imagePlaneDirty=true;
        }
    }

    void setImagePlaneDirty(const bool val) { imagePlaneDirty=val; }

    void setNewPointClicked(const bool val) { newPointClicked = val;}
    // returns value at point (for the widget)
    cimg_library::CImg<T> get_point(const Coord& p) const
    {
        if(!this->img) return cimg_library::CImg<T>();
        if(!this->img->getCImgList().size()) return cimg_library::CImg<T>();
        if(this->time>=this->img->getDimensions()[4]) return cimg_library::CImg<T>();
        for(unsigned int i=0; i<3; i++) if(p[i]<0 || p[i]>this->img->getDimensions()[i]-1) return cimg_library::CImg<T>();
        cimg_library::CImg<T> ret(1,1,1,this->img->getDimensions()[3]);
        cimg_forC(ret,c) ret(0,0,0,c)=this->img->getCImg(this->time).atXYZC((unsigned int)helper::round(p[0]),(unsigned int)helper::round(p[1]),(unsigned int)helper::round(p[2]),c);
        return ret;
    }
    // returns slice image
    cimg_library::CImg<T> get_slice(const unsigned int index,const unsigned int axis,const type::Mat<2,3,unsigned int>& roi) const
    {
        if(!this->img) return cimg_library::CImg<T>();
        if(!this->img->getCImgList().size()) return cimg_library::CImg<T>();
        if(index>=this->img->getDimensions()[axis] || this->time>=this->img->getDimensions()[4]) return cimg_library::CImg<T>();			// discard out of volume planes
        if((this->img->getDimensions()[0]==1 && axis!=0) || (this->img->getDimensions()[1]==1 && axis!=1) || (this->img->getDimensions()[2]==1 && axis!=2)) return cimg_library::CImg<T>();  // discard unit width/height images
        return this->img->get_plane(index,axis,roi,this->time,this->mergeChannels);
    }
    cimg_library::CImg<T> get_slice(const unsigned int index,const unsigned int axis) const
    {
        type::Mat<2,3,unsigned int> roi;
        for(unsigned int i=0; i<3; i++) { roi[0][i]=0; roi[1][i]=img->getDimensions()[i]-1; }
        return get_slice(index,axis,roi);
    }

    // returns 8-bits color image cutting through visual models
    cimg_library::CImg<unsigned char> get_slicedModels(const unsigned int index,const unsigned int axis,const type::Mat<2,3,unsigned int>& roi) const
    {
        if(!this->img) return cimg_library::CImg<unsigned char>();
        if(!this->img->getCImgList().size()) return cimg_library::CImg<unsigned char>();
        if(index>=this->img->getDimensions()[axis] || this->time>=this->img->getDimensions()[4]) return cimg_library::CImg<unsigned char>();			// discard out of volume planes
        if((this->img->getDimensions()[0]==1 && axis!=0) || (this->img->getDimensions()[1]==1 && axis!=1) || (this->img->getDimensions()[2]==1 && axis!=2)) return cimg_library::CImg<unsigned char>();  // discard unit width/height images

        const unsigned int dim[3]= {roi[1][0]-roi[0][0]+1,roi[1][1]-roi[0][1]+1,roi[1][2]-roi[0][2]+1};
        cimg_library::CImg<unsigned char> ret;
        if(axis==0)  ret=cimg_library::CImg<unsigned char>(dim[2],dim[1],1,3);
        else if(axis==1)  ret=cimg_library::CImg<unsigned char>(dim[0],dim[2],1,3);
        else ret=cimg_library::CImg<unsigned char>(dim[0],dim[1],1,3);
        ret.fill(0);

        for(unsigned int m=0; m<visualModels.size(); m++)
        {
            sofa::component::visual::VisualStyle::SPtr ptr = visualModels[m]->getContext()->template get<sofa::component::visual::VisualStyle>();
            if (ptr && !ptr->displayFlags.getValue().getShowVisualModels()) continue;

            const sofa::type::vector<VisualModelTypes::Coord>& verts= visualModels[m]->getVertices();

            sofa::type::vector<Coord> tposition;
            tposition.resize(verts.size());
            for(unsigned int i=0; i<tposition.size(); i++)
            {
                tposition[i]=transform->toImage(Coord((Real)verts[i][0],(Real)verts[i][1],(Real)verts[i][2]));
            }
            helper::ReadAccessor<Data< sofa::type::Material > > mat(visualModels[m]->material);
            const unsigned char color[3]= {(unsigned char)helper::round(mat->diffuse[0]*255.),(unsigned char)helper::round(mat->diffuse[1]*255.),(unsigned char)helper::round(mat->diffuse[2]*255.)};

            cimg_library::CImg<bool> tmp = this->img->get_slicedModels(index,axis,roi,tposition,visualModels[m]->getTriangles(),visualModels[m]->getQuads());
            cimg_foroff(tmp,off)
                    if(tmp[off])
            {
                ret.get_shared_channel(0)[off]=color[0];
                ret.get_shared_channel(1)[off]=color[1];
                ret.get_shared_channel(2)[off]=color[2];
            }
        }

        return ret;

    }

    cimg_library::CImg<unsigned char> get_slicedModels(const unsigned int index,const unsigned int axis) const
    {
        type::Mat<2,3,unsigned int> roi;
        for(unsigned int i=0; i<3; i++) { roi[0][i]=0; roi[1][i]=img->getDimensions()[i]-1; }
        return get_slicedModels(index,axis,roi);
    }

    // returns the transformed parameters (for the widget)
    Coord get_transformTranslation() const { return transform->getTranslation(); }
    Coord get_transformRotation() const { return transform->getRotation(); }
    Coord get_transformScale() const { return transform->getScale(); }

    // returns the transformed point (for the widget)
    Coord get_pointCoord(const Coord& ip) const { return transform->fromImage(ip); }
    Coord get_pointImageCoord(const Coord& ip) const { return transform->toImage(ip); }
    // returns the 4 slice corners
    type::Vec<4,Coord> get_sliceCoord(const unsigned int index,const unsigned int axis,const type::Mat<2,3,unsigned int>& roi) const
    {
        type::Vec<4,Coord> ip;
        if(axis==0) // zy
        {
            ip[0] = Coord(index,roi[0][1]-0.5,roi[0][2]-0.5);
            ip[1] = Coord(index,roi[0][1]-0.5,roi[1][2]+0.5);
            ip[2] = Coord(index,roi[1][1]+0.5,roi[1][2]+0.5);
            ip[3] = Coord(index,roi[1][1]+0.5,roi[0][2]-0.5);
        }
        else if (axis==1) // xz
        {
            ip[0] = Coord(roi[0][0]-0.5,index,roi[0][2]-0.5);
            ip[1] = Coord(roi[1][0]+0.5,index,roi[0][2]-0.5);
            ip[2] = Coord(roi[1][0]+0.5,index,roi[1][2]+0.5);
            ip[3] = Coord(roi[0][0]-0.5,index,roi[1][2]+0.5);
        }
        else //xy
        {
            ip[0] = Coord(roi[0][0]-0.5,roi[0][1]-0.5,index);
            ip[1] = Coord(roi[1][0]+0.5,roi[0][1]-0.5,index);
            ip[2] = Coord(roi[1][0]+0.5,roi[1][1]+0.5,index);
            ip[3] = Coord(roi[0][0]-0.5,roi[1][1]+0.5,index);
        }

        type::Vec<4,Coord> ret;
        for(unsigned int i=0; i<4; i++) ret[i] = transform->fromImage(ip[i]);

        return ret;
    }
    type::Vec<4,Coord> get_sliceCoord(const unsigned int index,const unsigned int axis) const
    {
        type::Mat<2,3,unsigned int> roi;
        for(unsigned int i=0; i<3; i++) { roi[0][i]=0; roi[1][i]=img->getDimensions()[i]-1; }
        return get_sliceCoord(index,axis,roi);
    }





    inline friend std::istream& operator >> ( std::istream& in, ImagePlane& p )
    {
        type::Vec<3,int> _plane;
        in>>_plane;
        p.setPlane(pCoord((unsigned int)_plane[0],(unsigned int)_plane[1],(unsigned int)_plane[2]));
        return in;
    }

    friend std::ostream& operator << ( std::ostream& out, const ImagePlane& p )
    {
        out<<p.getPlane();
        return out;
    }


};

} // namespace defaulttype


} // namespace sofa


#endif
