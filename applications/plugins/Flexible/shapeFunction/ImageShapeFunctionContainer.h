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
#ifndef FLEXIBLE_ImageShapeFunctionContainer_H
#define FLEXIBLE_ImageShapeFunctionContainer_H

#include <Flexible/config.h>
#include "../shapeFunction/BaseShapeFunction.h"
#include "../shapeFunction/BaseImageShapeFunction.h"

#include <image/ImageTypes.h>

namespace sofa
{
namespace component
{
namespace shapefunction
{


/**
Provides interface to mapping from precomputed shape functions
  */


template <class ShapeFunctionTypes_,class ImageTypes_>
class ImageShapeFunctionContainer : public BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(ImageShapeFunctionContainer, ShapeFunctionTypes_,ImageTypes_) , SOFA_TEMPLATE2(BaseImageShapeFunction, ShapeFunctionTypes_,ImageTypes_));
    typedef BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_> Inherit;


    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const ImageShapeFunctionContainer<ShapeFunctionTypes_,ImageTypes_>* = NULL) { return ShapeFunctionTypes_::Name()+std::string(",")+ImageTypes_::Name(); }


    virtual void init()    { Inherit::init(); reinit();}
    virtual void reinit()
    {
        Inherit::reinit();
        // chane nbref according to nb channels
        unsigned int nbchannels = this->f_w.getValue().getDimensions()[Inherit::DistTypes::DIMENSION_S];
        if(nbchannels!=this->f_nbRef.getValue())
        {
            if(this->f_printLog.getValue()) std::cout<<this->getName()<<" changed nbref according to nbChannels: "<<nbchannels<<std::endl;
            this->f_nbRef.setValue(nbchannels);
        }
    }

protected:
    ImageShapeFunctionContainer()
        :Inherit()
    {
        this->image.setDisplayed(false);
        this->transform.setReadOnly(true);
        this->f_w.setReadOnly(true);
        this->f_index.setReadOnly(true);
        this->f_w.setGroup("input");
        this->f_index.setGroup("input");
    }

    virtual ~ImageShapeFunctionContainer()
    {

    }

};


}
}
}


#endif
