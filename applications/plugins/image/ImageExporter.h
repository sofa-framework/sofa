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

#ifndef SOFA_IMAGE_IMAGEEXPORTER_H
#define SOFA_IMAGE_IMAGEEXPORTER_H

#include "initImage.h"
#include "ImageTypes.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/rmath.h>

namespace sofa
{

namespace component
{

namespace misc
{

using namespace defaulttype;

template <class _ImageTypes>
class ImageExporter : public virtual core::objectmodel::BaseObject
{
public:
    typedef core::objectmodel::BaseObject Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageExporter,_ImageTypes),Inherited);

    // image data
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;

    // transform data
    typedef SReal Real;
    typedef ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;;

    // output file
    sofa::core::objectmodel::DataFileName m_filename;

    Data<unsigned int> exportEveryNbSteps;
    Data<bool> exportAtBegin;
    Data<bool> exportAtEnd;


    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ImageExporter<ImageTypes>* = NULL) { return ImageTypes::Name(); }

    ImageExporter()	: Inherited()
        , image(initData(&image,ImageTypes(),"image","image"))
        , transform(initData(&transform, TransformType(), "transform" , ""))
        , m_filename( initData(&m_filename, "filename", "output file"))
        , exportEveryNbSteps( initData(&exportEveryNbSteps, (unsigned int)0, "exportEveryNumberOfSteps", "export file only at specified number of steps (0=disable)"))
        , exportAtBegin( initData(&exportAtBegin, false, "exportAtBegin", "export file at the initialization"))
        , exportAtEnd( initData(&exportAtEnd, false, "exportAtEnd", "export file when the simulation is finished"))
        , stepCounter(0)
        , time(0)
    {
        this->addAlias(&image, "outputImage");
        this->addAlias(&transform, "outputTransform");
        image.setReadOnly(true);
        transform.setReadOnly(true);
        f_listening.setValue(true);
    }

    virtual ~ImageExporter() {}

    virtual	void cleanup() 	{ if (exportAtEnd.getValue()) write();	}

    virtual void bwdInit()	{ if (exportAtBegin.getValue())	write(); }

protected:

    bool write()
    {
        if (!this->m_filename.isSet()) { serr << "ImageExporter: file not set"<<name<<sendl; return false; }
        std::string fname(this->m_filename.getFullPath());

        raImage rimage(this->image);
        raTransform rtransform(this->transform);
        if (!rimage->getCImgList()) { serr << "ImageExporter: no image "<<name<<sendl; return false; }

        if(fname.find(".mhd")!=std::string::npos || fname.find(".MHD")!=std::string::npos || fname.find(".Mhd")!=std::string::npos
           || fname.find(".raw")!=std::string::npos || fname.find(".RAW")!=std::string::npos || fname.find(".Raw")!=std::string::npos)
        {
            if(fname.find(".raw")!=std::string::npos || fname.find(".RAW")!=std::string::npos || fname.find(".Raw")!=std::string::npos)      fname.replace(fname.find_last_of('.')+1,fname.size(),"mhd");

            double scale[3]; for(unsigned int i=0; i<3; i++) scale[i]=(double)rtransform->getScale()[i];
            double translation[3]; for(unsigned int i=0; i<3; i++) translation[i]=(double)rtransform->getTranslation()[i];
            Vec<3,Real> rotation = rtransform->getRotation() * (Real)M_PI / (Real)180.0;
            helper::Quater< Real > q = helper::Quater< Real >::createQuaterFromEuler(rotation);
            Mat<3,3,Real> R;  q.toMatrix(R);
            double affine[9]; for(unsigned int i=0; i<3; i++) for(unsigned int j=0; j<3; j++) affine[3*i+j]=(double)R[i][j];
            double offsetT=(double)rtransform->getOffsetT();
            double scaleT=(double)rtransform->getScaleT();
            bool isPerspective=rtransform->isPerspective();
            save_metaimage<T,double>(rimage->getCImgList(),fname.c_str(),scale,translation,affine,&offsetT,&scaleT,&isPerspective);
        }
        else if(fname.find(".nfo")!=std::string::npos || fname.find(".NFO")!=std::string::npos || fname.find(".Nfo")!=std::string::npos)
        {
            // nfo files are used for compatibility with gridmaterial of frame and voxelizer plugins
            std::ofstream fileStream (fname.c_str(), std::ofstream::out);
            if (!fileStream.is_open()) { serr << "GridMaterial, Can not open " << fname << sendl; return false; }
            fileStream << "voxelType: " << CImg<T>::pixel_type() << std::endl;
            fileStream << "dimensions: " << rimage->getDimensions()[0] << " " << rimage->getDimensions()[1]<< " " << rimage->getDimensions()[2]  << std::endl;
            fileStream << "origin: " << rtransform->getTranslation()[0] << " " << rtransform->getTranslation()[1]<< " " << rtransform->getTranslation()[2]<< std::endl;
            fileStream << "voxelSize: " << rtransform->getScale()[0] << " " << rtransform->getScale()[1]<< " " << rtransform->getScale()[2]<< std::endl;
            fileStream.close();
            std::string imgName (fname);  imgName.replace(imgName.find_last_of('.')+1,imgName.size(),"raw");
            CImg<unsigned char> ucimg = rimage->getCImg(this->time);
            ucimg.save_raw(imgName.c_str());
        }
        else if	(fname.find(".cimg")!=std::string::npos || fname.find(".CIMG")!=std::string::npos || fname.find(".Cimg")!=std::string::npos || fname.find(".CImg")!=std::string::npos)
            rimage->getCImgList().save_cimg(fname.c_str());
        else if(fname.find(".avi")!=std::string::npos || fname.find(".mov")!=std::string::npos || fname.find(".asf")!=std::string::npos || fname.find(".divx")!=std::string::npos || fname.find(".flv")!=std::string::npos || fname.find(".mpg")!=std::string::npos || fname.find(".m1v")!=std::string::npos || fname.find(".m2v")!=std::string::npos || fname.find(".m4v")!=std::string::npos || fname.find(".mjp")!=std::string::npos || fname.find(".mkv")!=std::string::npos || fname.find(".mpe")!=std::string::npos || fname.find(".movie")!=std::string::npos || fname.find(".ogm")!=std::string::npos || fname.find(".ogg")!=std::string::npos || fname.find(".qt")!=std::string::npos || fname.find(".rm")!=std::string::npos || fname.find(".vob")!=std::string::npos || fname.find(".wmv")!=std::string::npos || fname.find(".xvid")!=std::string::npos || fname.find(".mpeg")!=std::string::npos )
            rimage->getCImgList().save_ffmpeg(fname.c_str());
        else if (fname.find(".hdr")!=std::string::npos || fname.find(".nii")!=std::string::npos)
        {
            float voxsize[3];
            for(unsigned int i=0; i<3; i++) voxsize[i]=(float)rtransform->getScale()[i];
            rimage->getCImg(this->time).save_analyze(fname.c_str(),voxsize);
        }
        else if (fname.find(".inr")!=std::string::npos)
        {
            float voxsize[3];
            for(unsigned int i=0; i<3; i++) voxsize[i]=(float)rtransform->getScale()[i];
            rimage->getCImg(this->time).save_inr(fname.c_str(),voxsize);
        }
        else rimage->getCImg(this->time).save(fname.c_str());

        sout << "Saved image " << fname <<" ("<< rimage->getCImg(this->time).pixel_type() <<")"  << sendl;

        return true;
    }



    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
        {
            //std::cout << "key pressed " << std::endl;
            switch(ev->getKey())
            {

            case 'E':
            case 'e':
                write();
                break;
            }
        }
        else if ( /*simulation::AnimateEndEvent* ev =*/  dynamic_cast<simulation::AnimateEndEvent*>(event))
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
            this->time=(unsigned int)t;

            unsigned int maxStep = exportEveryNbSteps.getValue();
            if (maxStep == 0) return;

            stepCounter++;
            if(stepCounter >= maxStep)
            {
                stepCounter = 0;
                write();
            }
        }
    }

    unsigned int stepCounter;
    unsigned int time;


};

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_IMAGEEXPORTER_H
