#ifndef IMAGETOOLBOXDATA_H
#define IMAGETOOLBOXDATA_H

#include "labelimagetoolbox.h"
#include <image/ImageTypes.h>

#include <image_gui/config.h>

namespace sofa
{

namespace defaulttype
{

template<typename _T>
struct SOFA_IMAGE_GUI_API ImageToolBoxData
{
public:
    typedef _T T;
    typedef Image<T> ImageTypes;
    typedef ImagePlane<T> ImagePlaneType;
    typedef sofa::component::engine::LabelImageToolBox Label;
    typedef type::vector<Label*> VecLabel;
    
protected:
    const ImageTypes* p_img;
    ImagePlaneType *p_plane;
    VecLabel v_label;
    
     type::Vec<2,T> clamp;				// input clamp values
    
public:
    ImageToolBoxData()
    {
    
    }
    
    void setInput(const ImageTypes& img)
    {
        p_img=&img;
        update();
    }
    
    void setPlane(ImagePlaneType &p)
    {
        p_plane=&p;
    }
    
    const ImagePlaneType& getPlane()const{return *p_plane;}
    
    ImagePlaneType& plane(){return *p_plane;}
    
    void setImage(const ImageTypes& img)
    {
        p_img = &img;
    }
    
    void setLabels(VecLabel &vlabel)
    {
        v_label = vlabel;
    }
    
    void addLabel(Label &l)
    {
        v_label.push_back(&l);
    }
    
    const VecLabel& getLabels()const{return v_label;}
    
    int getLabelsSize()const {return (int)v_label.size();}
    
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
    
    }
    
    inline friend std::istream& operator >> ( std::istream& in, ImageToolBoxData& h )
    {
        type::Vec<2,T> myclamp;
        in>>myclamp;
        h.setClamp(myclamp);
        return in;
    }

    friend std::ostream& operator << ( std::ostream& out, const ImageToolBoxData& h )
    {
        out<<h.getClamp();
        return out;
    }

};

}
}

#endif // IMAGETOOLBOXDATA_H
