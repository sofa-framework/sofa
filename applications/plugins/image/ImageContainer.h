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
#ifndef IMAGE_IMAGECONTAINER_H
#define IMAGE_IMAGECONTAINER_H

#include <image/config.h>
#include "ImageTypes.h"
#include <climits>
#include <sofa/type/Vec.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/BoundingBox.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Quat.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/system/FileRepository.h>

#if IMAGE_HAVE_ZLIB
#include <zlib.h>
#endif


namespace sofa 
{

namespace component
{

namespace container
{


/// Default implementation does not compile
template <class ImageType>
struct ImageContainerSpecialization
{
};

/// forward declaration
template<class ImageTypes> class ImageContainer;

template <class T>
struct ImageContainerSpecialization< defaulttype::Image<T> >
{
    typedef ImageContainer<defaulttype::Image<T>> ImageContainerT;

    static void constructor( ImageContainerT* container )
    {
        container->f_listening.setValue(true);  // to update camera during animate
    }

    static void parse( ImageContainerT* container, sofa::core::objectmodel::BaseObjectDescription* /* arg */ = nullptr )
    {
        if( container->image.isSet() ) return; // image is set from data link

        // otherwise try to load it from a file
        typename ImageContainerT::waImage wimage(container->image);
        if( wimage->isEmpty() )
            if( !container->load() )
                container->loadCamera();
    }

    static void init( ImageContainerT* container )
    {
        // if the image is not set from data link
        // and was not loaded from a file during parsing
        // try to load it now (maybe the loading was data-dependant, like the filename)

        typename ImageContainerT::waImage wimage(container->image);
        if( wimage->isEmpty() )
            if( !container->load() )
                if( !container->loadCamera() )
                {
                    wimage->getCImgList().push_back(cimg_library::CImg<T>());
                    msg_error(container) << "no input image" ;
                }
    }

    static bool load( ImageContainerT* container, std::string fname )
    {
        typedef typename ImageContainerT::Real Real;

        typename ImageContainerT::waImage wimage(container->image);
        typename ImageContainerT::waTransform wtransform(container->transform);

        // read image
#if IMAGE_HAVE_ZLIB
        //Load .inr.gz using ZLib
        if(fname.size() >= 3 && (fname.substr(fname.size()-7)==".inr.gz" || fname.substr(fname.size()-4)==".inr") )
        {
            float voxsize[3];
            float translation[3]={0.,0.,0.}, rotation[3]={0.,0.,0.};
            cimg_library::CImg<T> img = cimg_library::_load_gz_inr<T>(NULL, fname.c_str(), voxsize, translation, rotation);
            wimage->getCImgList().push_back(img);

            if (!container->transformIsSet)
            {

                for(unsigned int i=0;i<3;i++) wtransform->getScale()[i]=(Real)voxsize[i];
                for(unsigned int i=0;i<3;i++) wtransform->getTranslation()[i]= (Real)translation[i];

                type::Mat<3,3,Real> R;
                R = container->RotVec3DToRotMat3D(rotation);
                type::Quat< float > q; q.fromMatrix(R);
                wtransform->getRotation()=q.toEulerVector() * (Real)180.0 / (Real)M_PI ;
            }
            //			Real t0 = wtransform->getRotation()[0];
            //			Real t1 = wtransform->getRotation()[1];
            //			Real t2 = wtransform->getRotation()[2];

        }
        else
        {
#endif // IMAGE_HAVE_ZLIB
            if(fname.find(".mhd")!=std::string::npos || fname.find(".MHD")!=std::string::npos || fname.find(".Mhd")!=std::string::npos
                    || fname.find(".raw")!=std::string::npos || fname.find(".RAW")!=std::string::npos || fname.find(".Raw")!=std::string::npos)
            {
                if(fname.find(".raw")!=std::string::npos || fname.find(".RAW")!=std::string::npos || fname.find(".Raw")!=std::string::npos)      fname.replace(fname.find_last_of('.')+1,fname.size(),"mhd");

                double scale[3]={1.,1.,1.},translation[3]={0.,0.,0.},affine[9]={1.,0.,0.,0.,1.,0.,0.,0.,1.},offsetT=0.,scaleT=1.;
                int isPerspective=0;
                wimage->getCImgList().assign(cimg_library::load_metaimage<T,double>(fname.c_str(),scale,translation,affine,&offsetT,&scaleT,&isPerspective));
                if (!container->transformIsSet)
                {
                    for(unsigned int i=0;i<3;i++) wtransform->getScale()[i]=(Real)scale[i];
                    for(unsigned int i=0;i<3;i++) wtransform->getTranslation()[i]=(Real)translation[i];
                    type::Mat<3,3,Real> R; for(unsigned int i=0;i<3;i++) for(unsigned int j=0;j<3;j++) R[i][j]=(Real)affine[3*i+j];
                    type::Quat< Real > q; q.fromMatrix(R);
                    wtransform->getRotation()=q.toEulerVector() * (Real)180.0 / (Real)M_PI ;
                    wtransform->getOffsetT()=(Real)offsetT;
                    wtransform->getScaleT()=(Real)scaleT;
                    wtransform->isPerspective()=isPerspective;
                }
            }
            else if(fname.find(".nfo")!=std::string::npos || fname.find(".NFO")!=std::string::npos || fname.find(".Nfo")!=std::string::npos)
            {
                // nfo files are used for compatibility with gridmaterial of frame and voxelize rplugins

                std::ifstream fileStream (fname.c_str(), std::ifstream::in);
                if (!fileStream.is_open()) { msg_error(container) << "Cannot open " << fname ; return false; }
                std::string str;
                fileStream >> str;	char vtype[32]; fileStream.getline(vtype,32);
                type::Vec<3,unsigned int> dim;  fileStream >> str; fileStream >> dim;
                if (!container->transformIsSet)
                {
                    type::Vec<3,double> translation; fileStream >> str; fileStream >> translation;        for(unsigned int i=0;i<3;i++) wtransform->getTranslation()[i]=(Real)translation[i];
                    type::Vec<3,double> scale; fileStream >> str; fileStream >> scale;     for(unsigned int i=0;i<3;i++) wtransform->getScale()[i]=(Real)scale[i];
                }
                fileStream.close();

                std::string imgName (fname);  imgName.replace(imgName.find_last_of('.')+1,imgName.size(),"raw");
                wimage->getCImgList().push_back(cimg_library::CImg<T>().load_raw(imgName.c_str(),dim[0],dim[1],dim[2]));
            }
            else if(fname.find(".cimg")!=std::string::npos || fname.find(".CIMG")!=std::string::npos || fname.find(".Cimg")!=std::string::npos || fname.find(".CImg")!=std::string::npos)
            {
                wimage->getCImgList().load_cimg(fname.c_str());
            }
            else if(fname.find(".par")!=std::string::npos || fname.find(".rec")!=std::string::npos)
            {
                wimage->getCImgList().load_parrec(fname.c_str());
            }
            else if(fname.find(".avi")!=std::string::npos || fname.find(".mov")!=std::string::npos || fname.find(".asf")!=std::string::npos || fname.find(".divx")!=std::string::npos || fname.find(".flv")!=std::string::npos || fname.find(".mpg")!=std::string::npos || fname.find(".m1v")!=std::string::npos || fname.find(".m2v")!=std::string::npos || fname.find(".m4v")!=std::string::npos || fname.find(".mjp")!=std::string::npos || fname.find(".mkv")!=std::string::npos || fname.find(".mpe")!=std::string::npos || fname.find(".movie")!=std::string::npos || fname.find(".ogm")!=std::string::npos || fname.find(".ogg")!=std::string::npos || fname.find(".qt")!=std::string::npos || fname.find(".rm")!=std::string::npos || fname.find(".vob")!=std::string::npos || fname.find(".wmv")!=std::string::npos || fname.find(".xvid")!=std::string::npos || fname.find(".mpeg")!=std::string::npos )
            {
                wimage->getCImgList().load_ffmpeg_external(fname.c_str());
            }
            else if (fname.find(".hdr")!=std::string::npos || fname.find(".nii")!=std::string::npos)
            {
                float voxsize[3];
                wimage->getCImgList().push_back(cimg_library::CImg<T>().load_analyze(fname.c_str(),voxsize));
                if (!container->transformIsSet)
                    for(unsigned int i=0;i<3;i++) wtransform->getScale()[i]=(Real)voxsize[i];
                readNiftiHeader(container, fname);
            }
            else if (fname.find(".inr")!=std::string::npos)
            {
                float voxsize[3];
                wimage->getCImgList().push_back(cimg_library::CImg<T>().load_inr(fname.c_str(),voxsize));
                if (!container->transformIsSet)
                    for(unsigned int i=0;i<3;i++) wtransform->getScale()[i]=(Real)voxsize[i];
            }
            else
            {
                wimage->getCImgList().push_back(cimg_library::CImg<T>().load(fname.c_str()));
            }
#if IMAGE_HAVE_ZLIB
        }
#endif // IMAGE_HAVE_ZLIB

        const bool isImageEmpty = wimage->isEmpty();
        msg_info_when(!isImageEmpty, container) << "Loaded image " << fname <<" ("<< wimage->getCImg().pixel_type() <<")";
        return !isImageEmpty;
    }

    static bool loadCamera( ImageContainerT* container )
    {
        if( container->m_filename.isSet() ) return false;
        if( container->name.getValue().find("CAMERA") == std::string::npos ) return false;

#ifdef cimg_use_opencv
        typename ImageContainerT::waImage wimage(container->image);
        if(wimage->isEmpty() wimage->getCImgList().push_back(CImg<T>().load_camera());
                else wimage->getCImgList()[0].load_camera();
                if(!wimage->isEmpty())  return true;  else return false;
#else
        return false;
#endif
    }

    /* Read the header of a nifti file.
     * CImg only allows to get the voxel size of a nifti image, whereas this function
     * gives access to the whole structure of the header to get rotation and translation.
     */
    static void readNiftiHeader(ImageContainerT* container, std::string fname)
    {
        typedef typename ImageContainerT::Real Real;
        typename ImageContainerT::waTransform wtransform(container->transform);

        struct transformData{
            float b;
            float c;
            float d;

            float x;
            float y;
            float z;
        };

        transformData data;

        FILE* file = fopen(fname.c_str(), "rb");
        fseek(file, 4*sizeof(int) + 18*sizeof(float) + 136*sizeof(char) + 16*sizeof(short), SEEK_SET);

        size_t result = fread(&data, 1, sizeof(transformData), file);
        if (result!=sizeof(transformData))
            std::cerr << "Error reading header of " << fname << std::endl;
        else
        {
            wtransform->getTranslation()[0] = (Real) data.x;
            wtransform->getTranslation()[1] = (Real) data.y;
            wtransform->getTranslation()[2] = (Real) data.z;

            Real b = (Real)data.b;
            Real c = (Real)data.c;
            Real d = (Real)data.d;
            Real a = sqrt(1.0 - (b*b+c*c+d*d));
            type::Quat<Real> q(a,b,c,d);

            if (!container->transformIsSet)
            {
                wtransform->getRotation()=q.toEulerVector() * (Real)180.0 / (Real)M_PI ;
            }
        }

        float pixdim[8];
        fseek(file, 2*sizeof(int) + 3*sizeof(float) + 30*sizeof(char) + 13*sizeof(short), SEEK_SET);
        result = fread(&pixdim, 1, sizeof(pixdim), file);

        if (result!=sizeof(pixdim))
            std::cerr << "Error reading header of " << fname << std::endl;
        else
            for(unsigned int i=0;i<3;i++) wtransform->getScale()[i]=(Real)pixdim[i+1];

        fclose(file);
    }
};






/**
   * \brief This component is responsible for loading images
   *
   *  ImageContainer scene options:
   *
   *  <b>template</b>
   *
   *  <b>filename</> - the name of the image file to be loaded. Currently supported filetypes:
   *
   */
template<class _ImageTypes>
class ImageContainer : public core::objectmodel::BaseObject
{

    friend struct ImageContainerSpecialization<_ImageTypes>;

public:
    typedef core::objectmodel::BaseObject Inherited;
    SOFA_CLASS( SOFA_TEMPLATE(ImageContainer, _ImageTypes),Inherited);

    // image data
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image; ///< image

    // transform data
    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef helper::WriteAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform; ///< 12-param vector for trans, rot, scale, ...

    // input file
    sofa::core::objectmodel::DataFileName m_filename;

    Data<bool> drawBB; ///< draw bounding box

    /**
    * If true, the container will attempt to load a sequence of images starting from the file given by filename
    */
    Data<bool> sequence; ///< load a sequence of images
    /**
    * The number of frames of the sequence to be loaded.
    */
    Data<unsigned int> nFrames; ///< The number of frames of the sequence to be loaded. Default is the entire sequence.

    ImageContainer() : Inherited()
      , image(initData(&image,ImageTypes(),"image","image"))
      , transform(initData(&transform, "transform" , "12-param vector for trans, rot, scale, ..."))
      , m_filename(initData(&m_filename,"filename","Image file"))
      , drawBB(initData(&drawBB,false,"drawBB","draw bounding box"))
      , sequence(initData(&sequence, false, "sequence", "load a sequence of images"))
      , nFrames (initData(&nFrames, "numberOfFrames", "The number of frames of the sequence to be loaded. Default is the entire sequence."))
      , transformIsSet (false)
    {
        this->addAlias(&image, "inputImage");
        this->addAlias(&transform, "inputTransform");
        this->addAlias(&nFrames, "nFrames");
        this->transform.setGroup("Transform");
        this->transform.unset();

        ImageContainerSpecialization<ImageTypes>::constructor( this );
    }


    virtual void clear()
    {
        waImage wimage(this->image);
        wimage->clear();
    }

    ~ImageContainer() override {clear();}

    bool transformIsSet;

    void parse(sofa::core::objectmodel::BaseObjectDescription *arg) override
    {
        Inherited::parse(arg);

        this->transformIsSet = false;
        if (this->transform.isSet())
        {
            this->transformIsSet = true;
        }
        if (!this->transformIsSet)
        {
            this->transform.unset();
        }

        if (this->transformIsSet)
        {
            msg_info() << "Transform is set";
        }
        else
        {
            msg_info() << "Transform is NOT set";
        }

        ImageContainerSpecialization<ImageTypes>::parse( this, arg );
    }

    void init() override
    {
        ImageContainerSpecialization<ImageTypes>::init( this );

        raImage wimage(this->image);
        waTransform wtransform(this->transform);
        wtransform->setCamPos((Real)(wimage->getDimensions()[0]-1)/2.0,(Real)(wimage->getDimensions()[1]-1)/2.0); // for perspective transforms
        wtransform->update(); // update of internal data
    }




protected:


    type::Mat<3,3,Real> RotVec3DToRotMat3D(float *rotVec)
    {
        type::Mat<3,3,Real> rotMatrix;
        float c, s, k1, k2;
        float TH_TINY = 0.00001f;

        float theta2 =  rotVec[0]*rotVec[0] + rotVec[1]*rotVec[1] + rotVec[2]*rotVec[2];
        float theta = sqrt( theta2 );
        if (theta > TH_TINY){
            c = cos(theta);
            s = sin(theta);
            k1 = s / theta;
            k2 = (1 - c) / theta2;
        }
        else {  // Taylor expension around theta = 0
            k2 = 1.0f/2.0f - theta2/24.0f;
            c = 1.0f - theta2*k2;
            k1 = 1.0f - theta2/6.0f;
        }

        /* I + M*Mt */
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j <= i; j++){
                rotMatrix(i,j) = k2 * rotVec[i] * rotVec[j] ;
                if (i != j)
                    rotMatrix(j,i) = rotMatrix(i,j);
                else
                    rotMatrix(i,i) = rotMatrix(i,i) + c ;
            }
        }
        double aux = k1 * rotVec[2];
        rotMatrix(0,1) = rotMatrix(0,1) - aux;
        rotMatrix(1,0) = rotMatrix(1,0) + aux;
        aux = k1 * rotVec[1];
        rotMatrix(0,2) = rotMatrix(0,2) + aux;
        rotMatrix(2,0) = rotMatrix(2,0) - aux;
        aux = k1 * rotVec[0];
        rotMatrix(1,2) = rotMatrix(1,2) - aux;
        rotMatrix(2,1) = rotMatrix(2,1) + aux;

        return rotMatrix;
    }

    bool load()
    {
        if (!this->m_filename.isSet()) return false;

        std::string fname(this->m_filename.getFullPath());
        if (!sofa::helper::system::DataRepository.findFile(fname))
        {
            msg_error() << "ImageContainer: cannot find "<<fname;
            return false;
        }
        fname=sofa::helper::system::DataRepository.getFile(fname);

        if(sequence.getValue())
            return loadSequence(fname);
        else
            return load(fname);
    }

    bool load(std::string fname)
    {
        return ImageContainerSpecialization<ImageTypes>::load( this, fname );
    }

    //    bool load(std::FILE* const file, std::string fname)
    //    {
    //       return ImageContainerSpecialization<ImageTypes>::load( this, file, fname );
    //    }

    bool loadCamera()
    {
        return ImageContainerSpecialization<ImageTypes>::loadCamera( this );
    }

    void handleEvent(sofa::core::objectmodel::Event *event) override
    {
        if (simulation::AnimateEndEvent::checkEventType(event))
            loadCamera();
    }


    void getCorners(type::Vec<8,type::Vec3> &c) // get image corners
    {
        raImage rimage(this->image);
        const imCoord dim= rimage->getDimensions();

        type::Vec<8,type::Vec3> p;
        p[0]=type::Vec3(-0.5,-0.5,-0.5);
        p[1]=type::Vec3(dim[0]-0.5,-0.5,-0.5);
        p[2]=type::Vec3(-0.5,dim[1]-0.5,-0.5);
        p[3]=type::Vec3(dim[0]-0.5,dim[1]-0.5,-0.5);
        p[4]=type::Vec3(-0.5,-0.5,dim[2]-0.5);
        p[5]=type::Vec3(dim[0]-0.5,-0.5,dim[2]-0.5);
        p[6]=type::Vec3(-0.5,dim[1]-0.5,dim[2]-0.5);
        p[7]=type::Vec3(dim[0]-0.5,dim[1]-0.5,dim[2]-0.5);

        raTransform rtransform(this->transform);
        for(unsigned int i=0;i<p.size();i++) c[i]=rtransform->fromImage(p[i]);
    }

    void computeBBox(const core::ExecParams*  params, bool onlyVisible=false ) override
    {
        SOFA_UNUSED(params);

        if( onlyVisible && !drawBB.getValue()) return;

        type::Vec<8,type::Vec3> c;
        getCorners(c);

        Real bbmin[3]  = {c[0][0],c[0][1],c[0][2]} , bbmax[3]  = {c[0][0],c[0][1],c[0][2]};
        for(unsigned int i=1;i<c.size();i++)
            for(unsigned int j=0;j<3;j++)
            {
                if(bbmin[j]>c[i][j]) bbmin[j]=c[i][j];
                if(bbmax[j]<c[i][j]) bbmax[j]=c[i][j];
            }
        this->f_bbox.setValue(sofa::type::TBoundingBox<Real>(bbmin,bbmax));
    }

    void draw(const core::visual::VisualParams* vparams) override
    {
        // draw bounding box
        if (!vparams->displayFlags().getShowVisualModels()) return;
        if (!drawBB.getValue()) return;

        vparams->drawTool()->saveLastState();

        const sofa::type::RGBAColor color(1.,0.5,0.5,0.5);
        vparams->drawTool()->setMaterial(color);

        std::vector<type::Vec3> corners;
        type::Vec<8,type::Vec3> c;
        corners.resize(8);
        getCorners(c);
        for(unsigned int i=0;i<8;i++)
            corners[i]=c[i];

        vparams->drawTool()->drawLineLoop(corners,2.0,color);

        vparams->drawTool()->restoreLastState();
    }

    /*
    * Load a sequence of image files. The filename specified by the user should be the first in a sequence with the naming convention:
    *  name_N.extension, where name is consistent among all the files, and N is an integer that increases by 1 with each image in the sequence,
    *  and extension is the extension of a supported filetype.
    *  N can be in the form 1, 2, 3... or can have prefixed zeros (01, 02, 03...). In the case of prefixed zeros, all the values of N in the sequence
    *  must have the same number of digits. Examples: 01, 02, ... , 10, 11.   or   001, 002, ... , 010, 011, ... , 100, 101.
    */
    bool loadSequence(std::string fname)
    {
        std::string nextFname(fname);

        if (!sofa::helper::system::DataRepository.findFile(nextFname))
        {
            msg_error() << "ImageContainer: cannot find "<<fname;
            return false;
        }

        unsigned int nFramesLoaded = 0;
        unsigned int maxFrames = UINT_MAX;
        if(nFrames.isSet())
        {
            maxFrames = nFrames.getValue();
        }

        while(sofa::helper::system::DataRepository.findFile(nextFname) && nFramesLoaded < maxFrames)
        {
            load(nextFname);
            nextFname = getNextFname(nextFname);
            nFramesLoaded++;
        }
        return true;
    }

    /**
    * When loading a sequence of images, determines the filename of the next image in the sequence based on the current image's filename.
    */
    std::string getNextFname(std::string currentFname)
    {

        std::string filenameError = "ImageContainer: Invalid Filename ";
        std::string filenameDescription = "Filename of an image in a sequence must follow the convention \"name_N.extension\", where N is an integer and extension is a supported file type";
        std::size_t lastUnderscorePosition = currentFname.rfind("_");

        if(lastUnderscorePosition == std::string::npos)
        {
            msg_error() << filenameError << currentFname << msgendl
                        << filenameDescription;
            return "";
        }

        std::string fnameRoot = currentFname.substr(0, lastUnderscorePosition);

        std::size_t nextDotPosition = currentFname.find(".", lastUnderscorePosition);

        if(nextDotPosition == std::string::npos)
        {
            msg_error() << filenameError << currentFname << msgendl
                        << filenameDescription;
            return "";
        }

        std::string seqNStr = currentFname.substr(lastUnderscorePosition+1, nextDotPosition-(lastUnderscorePosition+1));

        std::string extension = currentFname.substr(nextDotPosition);


        int seqN = atoi(seqNStr.c_str());
        int nextSeqN = seqN + 1;

        std::ostringstream nextSeqNstream;
        nextSeqNstream << nextSeqN;
        std::string nextSeqNStr = nextSeqNstream.str();

        std::string prefix("");

        if(seqNStr.length() > nextSeqNStr.length())
        {
            int difference = seqNStr.length() - nextSeqNStr.length();
            for(int i=0; i<difference; i++)
            {
                prefix.append("0");
            }
        }

        std::ostringstream nextFname;
        nextFname << fnameRoot << "_" << prefix << nextSeqNStr << extension;

        return nextFname.str();
    }
};



}

}

}


#endif /*IMAGE_IMAGECONTAINER_H*/
