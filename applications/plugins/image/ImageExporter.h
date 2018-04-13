/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef SOFA_IMAGE_IMAGEEXPORTER_H
#define SOFA_IMAGE_IMAGEEXPORTER_H

#include <image/config.h>
#include "ImageTypes.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/GUIEvent.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/rmath.h>

namespace sofa
{

namespace component
{

namespace misc
{





/// Default implementation does not compile
template <class ImageType>
struct ImageExporterSpecialization
{
};

/// forward declaration
template <class ImageType> class ImageExporter;


/// Specialization for regular Image
template <class T>
struct ImageExporterSpecialization<defaulttype::Image<T>>
{
    typedef ImageExporter<defaulttype::Image<T>> ImageExporterT;


    static void init( ImageExporterT& /*exporter*/ )
    {
    }

    static bool write( ImageExporterT& exporter )
    {
        typedef typename ImageExporterT::Real Real;

        if (!exporter.m_filename.isSet()) { exporter.serr << "ImageExporter: file not set"<<exporter.name<<exporter.sendl; return false; }
        std::string fname(exporter.m_filename.getFullPath());

        typename ImageExporterT::raImage rimage(exporter.image);
        typename ImageExporterT::raTransform rtransform(exporter.transform);
        if (rimage->isEmpty()) { exporter.serr << "ImageExporter: no image "<<exporter.name<<exporter.sendl; return false; }

        if(fname.find(".mhd")!=std::string::npos || fname.find(".MHD")!=std::string::npos || fname.find(".Mhd")!=std::string::npos
           || fname.find(".raw")!=std::string::npos || fname.find(".RAW")!=std::string::npos || fname.find(".Raw")!=std::string::npos)
        {
            if(fname.find(".raw")!=std::string::npos || fname.find(".RAW")!=std::string::npos || fname.find(".Raw")!=std::string::npos)      fname.replace(fname.find_last_of('.')+1,fname.size(),"mhd");

            double scale[3]; for(unsigned int i=0; i<3; i++) scale[i]=(double)rtransform->getScale()[i];
            double translation[3]; for(unsigned int i=0; i<3; i++) translation[i]=(double)rtransform->getTranslation()[i];
            defaulttype::Vec<3,Real> rotation = rtransform->getRotation() * (Real)M_PI / (Real)180.0;
            helper::Quater< Real > q = helper::Quater< Real >::createQuaterFromEuler(rotation);
            defaulttype::Mat<3,3,Real> R;  q.toMatrix(R);
            double affine[9]; for(unsigned int i=0; i<3; i++) for(unsigned int j=0; j<3; j++) affine[3*i+j]=(double)R[i][j];
            double offsetT=(double)rtransform->getOffsetT();
            double scaleT=(double)rtransform->getScaleT();
            int isPerspective=rtransform->isPerspective();
            cimg_library::save_metaimage<T,double>(rimage->getCImgList(),fname.c_str(),scale,translation,affine,offsetT,scaleT,isPerspective);
        }
        else if(fname.find(".nfo")!=std::string::npos || fname.find(".NFO")!=std::string::npos || fname.find(".Nfo")!=std::string::npos)
        {
            // nfo files are used for compatibility with gridmaterial of frame and voxelizer plugins
            std::ofstream fileStream (fname.c_str(), std::ofstream::out);
            if (!fileStream.is_open()) { exporter.serr << "GridMaterial, Can not open " << fname << exporter.sendl; return false; }
            fileStream << "voxelType: " << cimg_library::CImg<T>::pixel_type() << std::endl;
            fileStream << "dimensions: " << rimage->getDimensions()[0] << " " << rimage->getDimensions()[1]<< " " << rimage->getDimensions()[2]  << std::endl;
            fileStream << "origin: " << rtransform->getTranslation()[0] << " " << rtransform->getTranslation()[1]<< " " << rtransform->getTranslation()[2]<< std::endl;
            fileStream << "voxelSize: " << rtransform->getScale()[0] << " " << rtransform->getScale()[1]<< " " << rtransform->getScale()[2]<< std::endl;
            fileStream.close();
            std::string imgName (fname);  imgName.replace(imgName.find_last_of('.')+1,imgName.size(),"raw");
            cimg_library::CImg<unsigned char> ucimg = rimage->getCImg(exporter.time);
            ucimg.save_raw(imgName.c_str());
        }
        else if	(fname.find(".cimg")!=std::string::npos || fname.find(".CIMG")!=std::string::npos || fname.find(".Cimg")!=std::string::npos || fname.find(".CImg")!=std::string::npos)
            rimage->getCImgList().save_cimg(fname.c_str());
        else if(fname.find(".avi")!=std::string::npos || fname.find(".mov")!=std::string::npos || fname.find(".asf")!=std::string::npos || fname.find(".divx")!=std::string::npos || fname.find(".flv")!=std::string::npos || fname.find(".mpg")!=std::string::npos || fname.find(".m1v")!=std::string::npos || fname.find(".m2v")!=std::string::npos || fname.find(".m4v")!=std::string::npos || fname.find(".mjp")!=std::string::npos || fname.find(".mkv")!=std::string::npos || fname.find(".mpe")!=std::string::npos || fname.find(".movie")!=std::string::npos || fname.find(".ogm")!=std::string::npos || fname.find(".ogg")!=std::string::npos || fname.find(".qt")!=std::string::npos || fname.find(".rm")!=std::string::npos || fname.find(".vob")!=std::string::npos || fname.find(".wmv")!=std::string::npos || fname.find(".xvid")!=std::string::npos || fname.find(".mpeg")!=std::string::npos )
            rimage->getCImgList().save_ffmpeg_external(fname.c_str());
        else if (fname.find(".hdr")!=std::string::npos || fname.find(".nii")!=std::string::npos)
        {
            float voxsize[3];
            for(unsigned int i=0; i<3; i++) voxsize[i]=(float)rtransform->getScale()[i];
            rimage->getCImg(exporter.time).save_analyze(fname.c_str(),voxsize);

            //once CImg wrote the data, we complete them with a header containing spatial transformation
            typedef struct
            {
                float nifti_voxOffset; //Offset into .nii file :: This value is not set in CImg

                //////////////////////////////////////////////
                char unchanged_0[140];
                //////////////////////////////////////////////

                short nifti_QForm; // NIFTI_XFORM_* code.
                short nifti_SForm; // NIFTI_XFORM_* code.

                //////////////////////////////////////////////

                float nifti_quaternion[6]; // Quaternion parameters
                float nifti_affine[12]; // affine transform

                //////////////////////////////////////////////
                char unchanged_1[16];
                //////////////////////////////////////////////

                char nifti_magic[4];// This value is not set in CImg

            } NiftiHeader;

            NiftiHeader header;

            FILE* file = fopen(fname.c_str(), "rb+");
            fseek(file, 108, SEEK_SET);
            if (fread(&header, 1, sizeof(NiftiHeader), file) != sizeof(NiftiHeader))
                std::cerr << "Error reading the header in " << fname << std::endl;

            header.nifti_voxOffset = 352;

            header.nifti_QForm = 1; //method 2
            header.nifti_SForm = 0;

            defaulttype::Vec<3,Real> rotation = rtransform->getRotation() * (Real)M_PI / (Real)180.0;
            helper::Quater<Real> q = helper::Quater<Real>::createQuaterFromEuler(rotation);

            for (unsigned int i = 0; i< 3; i++)
            {
                header.nifti_quaternion[i]   = (float)q[i+1];
                header.nifti_quaternion[3+i] = (float)rtransform->getTranslation()[i];
            }

            defaulttype::Matrix3 mat;
            q.toMatrix(mat);

            header.nifti_affine[0] = (float)(mat(0,0) * voxsize[0]); header.nifti_affine[1] =(float) mat(0,1);              header.nifti_affine[2] = (float)mat(0,2);               header.nifti_affine[3] = (float)rtransform->getTranslation()[0];
            header.nifti_affine[4] = (float)mat(1,0);              header.nifti_affine[5] = (float)(mat(1,1) * voxsize[1]); header.nifti_affine[6] = (float)mat(1,2);               header.nifti_affine[7] = (float)rtransform->getTranslation()[1];
            header.nifti_affine[8] = (float)mat(2,0);              header.nifti_affine[9] = (float)mat(2,1);              header.nifti_affine[10] = (float)(mat(2,2) * voxsize[2]); header.nifti_affine[11] = (float)rtransform->getTranslation()[2];

            header.nifti_magic[0] = 'n';
            if (fname.find(".hdr")!=std::string::npos)
                header.nifti_magic[1] = 'i';
            else if (fname.find(".nii")!=std::string::npos)
                header.nifti_magic[1] = '+';
            header.nifti_magic[2] = '1';

            fseek(file, 108, SEEK_SET);
            if (fwrite (&header , sizeof(NiftiHeader), 1, file) != 1)
                std::cerr << "Error writing the header in " << fname << std::endl;
            fclose(file);
        }
        else if (fname.find(".inr")!=std::string::npos)
        {
            float voxsize[3];
            float translation[3];
            for(unsigned int i=0; i<3; i++) voxsize[i]=(float)rtransform->getScale()[i];
            for(unsigned int i=0; i<3; i++) translation[i]=(float)rtransform->getTranslation()[i];
            save_inr(rimage->getCImg(exporter.time),NULL,fname.c_str(),voxsize,translation);
        }
        else rimage->getCImg(exporter.time).save(fname.c_str());

        exporter.sout << "Saved image " << fname <<" ("<< rimage->getCImg(exporter.time).pixel_type() <<")"  << exporter.sendl;

        return true;
    }

};




template <class _ImageTypes>
class ImageExporter : public core::objectmodel::BaseObject
{
    friend struct ImageExporterSpecialization<_ImageTypes>;

public:
    typedef core::objectmodel::BaseObject Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageExporter,_ImageTypes),Inherited);

    // image data
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image; ///< image

    // transform data
    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    // output file
    sofa::core::objectmodel::DataFileName m_filename;

    Data<unsigned int> exportEveryNbSteps; ///< export file only at specified number of steps (0=disable)
    Data<bool> exportAtBegin; ///< export file at the initialization
    Data<bool> exportAtEnd; ///< export file when the simulation is finished


    virtual std::string getTemplateName() const    override { return templateName(this);    }
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

        ImageExporterSpecialization<ImageTypes>::init( *this );
    }

    virtual ~ImageExporter() {}

    virtual	void cleanup() override { if (exportAtEnd.getValue()) write();	}

    virtual void bwdInit() override { if (exportAtBegin.getValue())	write(); }

protected:


    bool write()
    {
        return ImageExporterSpecialization<ImageTypes>::write( *this );
    }


    void handleEvent(sofa::core::objectmodel::Event *event) override
    {
        if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
        {
            sofa::core::objectmodel::KeypressedEvent *ev = static_cast<sofa::core::objectmodel::KeypressedEvent *>(event);

            //std::cout << "key pressed " << std::endl;
            switch(ev->getKey())
            {

            case 'E':
            case 'e':
                write();
                break;
            }
        }
        else if ( /*simulation::AnimateEndEvent* ev =*/ simulation::AnimateEndEvent::checkEventType(event))
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
        else if (sofa::core::objectmodel::GUIEvent::checkEventType(event))
        {
            sofa::core::objectmodel::GUIEvent *guiEvent = static_cast<sofa::core::objectmodel::GUIEvent *>(event);

            if (guiEvent->getValueName().compare("ImageExport") == 0)
                write();
        }
    }

    unsigned int stepCounter;
    unsigned int time;


};

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_IMAGEEXPORTER_H
