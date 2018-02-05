#ifndef IMAGETOOLBOX_H
#define IMAGETOOLBOX_H


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


#include <image/image_gui/config.h>
#include <image/ImageTypes.h>
#include <image/VectorVis.h>

#include <sofa/helper/io/Image.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Event.h>
#include <SofaBaseVisual/VisualModelImpl.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/CollisionModel.h>

#include "imagetoolboxdata.h"

namespace sofa
{

namespace component
{

namespace misc
{

using namespace cimg_library;
using defaulttype::Vec;
using defaulttype::Vector3;


template<class _ImageTypes>
class SOFA_IMAGE_GUI_API ImageToolBox : public sofa::core::objectmodel::BaseObject
{
public:
    typedef sofa::core::objectmodel::BaseObject Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageToolBox, _ImageTypes), Inherited);
    
    // image data
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;
    
    // @name ToolBoxData
    /**@{*/
    typedef defaulttype::ImageToolBoxData<T> ImageToolBoxDataType;
     typedef helper::WriteAccessor<Data< ImageToolBoxDataType > > waToolBox;
    typedef helper::ReadAccessor<Data< ImageToolBoxDataType > > raToolBox;
    Data< ImageToolBoxDataType > toolbox;
    /**@}*/
    
    // @name Plane selection
    /**@{*/
    typedef defaulttype::ImagePlane<T> ImagePlaneType;
    //typedef helper::ReadAccessor<Data< ImagePlaneType > > raPlane;
    //typedef helper::WriteAccessor<Data< ImagePlaneType > > waPlane;
    //Data< ImagePlaneType > plane;
    
    typedef ImagePlaneType* raPlane;
    typedef ImagePlaneType* waPlane;
    ImagePlaneType plane;
    
    
    /**@}*/
    
    //@name Transform data
    /**@{*/
    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;
    /**@}*/
    
    // @name Histogram
    /**@{*/
    typedef defaulttype::Histogram<T> HistogramType;
    typedef helper::WriteAccessor<Data< HistogramType > > waHisto;
    typedef helper::ReadAccessor<Data< HistogramType > > raHisto;
    Data< HistogramType > histo;
    /**@}*/
    
        // @name Vector visualization
    /**@{*/
    typedef helper::ReadAccessor<Data< defaulttype::VectorVis > > raVis;
    typedef helper::WriteAccessor<Data< defaulttype::VectorVis > > waVis;
    Data<defaulttype::VectorVis> vectorVisualization;
    /**@}*/
    
    typedef component::visualmodel::VisualModelImpl VisuModelType;
    
     typedef sofa::component::engine::LabelImageToolBox Label;
    typedef helper::vector<Label*> VecLabel;
        
    std::string getTemplateName() const  override {	return templateName(this);	}
    static std::string templateName(const ImageToolBox<ImageTypes>* = NULL)	{ return ImageTypes::Name(); }
    
    ImageToolBox() : Inherited()
      , image(initData(&image,ImageTypes(),"image","input image"))
      , toolbox(initData(&toolbox, "toolbox",""))
      , transform(initData(&transform, TransformType(), "transform" , ""))
      , histo(initData(&histo, HistogramType(256,256,false),"histo",""))
    {
        this->addAlias(&image, "outputImage");
        
        image.setGroup("Image");
        image.setReadOnly(true);
        
        histo.setGroup("Histogram");
        histo.setWidget("imagehistogram");
        
        transform.setGroup("Transform");
        transform.setReadOnly(true);
        
        toolbox.setGroup("toolbox");
        toolbox.setWidget("ImageToolBoxWidget");
        //toolbox.setPlane(&plane);
        
        vectorVisualization.setWidget("vectorvis");

        
        //for(unsigned int i=0;i<3;i++)	cutplane_tex[i]=NULL;
    }
    
    
    virtual ~ImageToolBox()
    {
        //for(unsigned int i=0;i<3;i++)	if(cutplane_tex[i]) delete cutplane_tex[i];
    }
    
    virtual void init() override
    {
        
        // getvisuals
        std::vector<VisuModelType*> visuals;
        sofa::core::objectmodel::BaseContext* context = this->getContext();
        context->get<VisuModelType>(&visuals,core::objectmodel::BaseContext::SearchRoot);
        
        // set histogram data
        waHisto whisto(this->histo);        whisto->setInput(image.getValue());
        
        // record user values of plane position
        typename ImagePlaneType::pCoord pc; bool usedefaultplanecoord=true;
        /*if(this->plane.isSet())*/
        
        {  raPlane rplane = &(this->plane); pc=rplane->getPlane(); usedefaultplanecoord=false; }
        
        // set plane data
        waPlane wplane = &(this->plane);
        wplane->setInput(image.getValue(),transform.getValue(),visuals);
        
        // set recorded plane pos
        if(!usedefaultplanecoord) wplane->setPlane(pc);
        
        // enable vecorvis ?
        if(wplane->getDimensions()[3]<2) vectorVisualization.setDisplayed(false);
        else         vectorVisualization.setGroup("Vectors");
                
        //for(unsigned int i=0;i<3;i++)
        //{
            //cutplane_tex[i]= new helper::gl::Texture(new helper::io::Image,false);
            //cutplane_tex[i]->getImage()->init(cutplane_res,cutplane_res,32);
        //}
 
        raVis rvis(this->vectorVisualization);

        whisto->setMergeChannels(!rvis->getRgb());
        wplane->setMergeChannels(!rvis->getRgb());
        wplane->setClamp(whisto->getClamp());
        
        waToolBox wtoolbox(this->toolbox);
        wtoolbox->setPlane(plane);
        
        //get labels
        VecLabel vec;
        context->get<Label, VecLabel >(&vec, sofa::core::objectmodel::BaseContext::SearchUp);
        wtoolbox->setLabels(vec);
        //std::cout << "vec<Label>" <<vec.size() << " "<< wtoolbox->getLabels().size() << std::endl;

        //updateTextures();
        
       // for(unsigned int i=0;i<3;i++)	cutplane_tex[i]->init();
        /*
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);*/
    }
    
    
    virtual void reinit() override
    {
        /*waHisto whisto(this->histo);
        waPlane wplane(this->plane);
        raVis rvis(this->vectorVisualization);
        
        whisto->setMergeChannels(!rvis->getRgb());
        wplane->setMergeChannels(!rvis->getRgb());
        wplane->setClamp(whisto->getClamp());*/
    }
    
    virtual void handleEvent( sofa::core::objectmodel::Event* /*event*/) override
    {
        /*typename ImagePlaneType::pCoord pc(0,0,0);

        if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
        {
            sofa::core::objectmodel::KeypressedEvent *ev = static_cast<sofa::core::objectmodel::KeypressedEvent *>(event);

            waPlane wplane(this->plane);
            unsigned int xmax = this->image.getValue().getDimensions()[0];
            unsigned int ymax = this->image.getValue().getDimensions()[1];
            unsigned int zmax = this->image.getValue().getDimensions()[2];
            switch(ev->getKey())
            {
            case '4':
                pc = wplane->getPlane();
                if (pc[0] > 0) pc[0]--;
                wplane->setPlane(pc);
                break;
            case '6':
                pc = wplane->getPlane();
                if (pc[0] < xmax) pc[0]++;
                wplane->setPlane(pc);
                break;
            case '-':
                pc = wplane->getPlane();
                if (pc[1] > 0) pc[1]--;
                wplane->setPlane(pc);
                break;
            case '+':
                pc = wplane->getPlane();
                if (pc[1] < ymax) pc[1]++;
                wplane->setPlane(pc);
                break;
            case '8':
                pc = wplane->getPlane();
                if (pc[2] > 0) pc[2]--;
                wplane->setPlane(pc);
                break;
            case '2':
                pc = wplane->getPlane();
                if (pc[2] < zmax) pc[2]++;
                wplane->setPlane(pc);
                break;
            }
        }*/
    }
    
    virtual void draw(const core::visual::VisualParams* /*vparams*/) override
    {}
    
    
protected:
    

};


} // namespace misc

} // namespace component

} // namespace sofa


#endif // IMAGETOOLBOX_H
