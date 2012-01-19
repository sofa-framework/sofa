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
#ifndef IMAGE_IMAGECONTAINER_H
#define IMAGE_IMAGECONTAINER_H

#include "initImage.h"
#include "ImageTypes.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/rmath.h>


namespace sofa
{

namespace component
{

namespace container
{

using namespace defaulttype;

template<class _ImageTypes>
class ImageContainer : public virtual core::objectmodel::BaseObject
{
public:
    typedef core::objectmodel::BaseObject Inherited;
    SOFA_CLASS( SOFA_TEMPLATE(ImageContainer, _ImageTypes),Inherited);

    // image data
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;

    // transform data
    typedef SReal Real;
    typedef ImageLPTransform<Real> TransformType;
    typedef helper::WriteAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    // input file
    sofa::core::objectmodel::DataFileName m_filename;

    Data<bool> drawBB;

    virtual std::string getTemplateName() const	{ return templateName(this); }
    static std::string templateName(const ImageContainer<ImageTypes>* = NULL) {	return ImageTypes::Name(); }

    ImageContainer() : Inherited()
        , image(initData(&image,ImageTypes(),"image","image"))
        , transform(initData(&transform, TransformType(), "transform" , ""))
        , m_filename(initData(&m_filename,"filename","Image file"))
        , drawBB(initData(&drawBB,true,"drawBB","draw bounding box"))
    {
        this->addAlias(&image, "inputImage");
        this->addAlias(&transform, "inputTransform");
        transform.setGroup("Transform");
        f_listening.setValue(true);  // to update camera during animate
    }


    virtual void clear()
    {
        waImage wimage(this->image);
        wimage->clear();
    }

    virtual ~ImageContainer() {clear();}

    virtual void init()
    {
        waImage wimage(this->image);
        waTransform wtransform(this->transform);

        if(!wimage->getCImg())
            if(!load())
                if(!loadCamera())
                {
                    wimage->getCImgList().push_back(CImg<T>());
                    serr << "ImageContainer: no input image "<<sendl;
                }

        wtransform->setCamPos((Real)wimage->getDimensions()[0]/2.0,(Real)wimage->getDimensions()[1]/2.0); // for perspective transforms
        wtransform->update(); // update of internal data
    }




protected:

    bool load()
    {
        if (!this->m_filename.isSet()) return false;

        std::string fname(this->m_filename.getFullPath());
        if (!sofa::helper::system::DataRepository.findFile(fname))
        {
            serr << "ImageContainer: cannot find "<<fname<<sendl;
            return false;
        }
        fname=sofa::helper::system::DataRepository.getFile(fname);

        waImage wimage(this->image);
        waTransform wtransform(this->transform);


        // read image
        if(fname.find(".mhd")!=std::string::npos || fname.find(".MHD")!=std::string::npos || fname.find(".Mhd")!=std::string::npos
           || fname.find(".raw")!=std::string::npos || fname.find(".RAW")!=std::string::npos || fname.find(".Raw")!=std::string::npos)
        {
            if(fname.find(".raw")!=std::string::npos || fname.find(".RAW")!=std::string::npos || fname.find(".Raw")!=std::string::npos)      fname.replace(fname.find_last_of('.')+1,fname.size(),"mhd");

            double scale[3]= {1.,1.,1.},translation[3]= {0.,0.,0.},affine[9]= {1.,0.,0.,0.,1.,0.,0.,0.,1.},offsetT=0.,scaleT=1.;
            bool isPerspective=false;
            wimage->getCImgList().assign(load_metaimage<T,double>(fname.c_str(),scale,translation,affine,&offsetT,&scaleT,&isPerspective));
            for(unsigned int i=0; i<3; i++) wtransform->getScale()[i]=(Real)scale[i];
            for(unsigned int i=0; i<3; i++) wtransform->getTranslation()[i]=(Real)translation[i];
            Mat<3,3,Real> R; for(unsigned int i=0; i<3; i++) for(unsigned int j=0; j<3; j++) R[i][j]=(Real)affine[3*i+j];
            helper::Quater< Real > q; q.fromMatrix(R);
            wtransform->getRotation()=q.toEulerVector() * (Real)180.0 / (Real)M_PI ;
            wtransform->getOffsetT()=(Real)offsetT;
            wtransform->getScaleT()=(Real)scaleT;
            wtransform->isPerspective()=isPerspective;
        }
        else if(fname.find(".cimg")!=std::string::npos || fname.find(".CIMG")!=std::string::npos || fname.find(".Cimg")!=std::string::npos || fname.find(".CImg")!=std::string::npos)
            wimage->getCImgList().load_cimg(fname.c_str());
        else if(fname.find(".par")!=std::string::npos || fname.find(".rec")!=std::string::npos)
            wimage->getCImgList().load_parrec(fname.c_str());
        else if(fname.find(".avi")!=std::string::npos || fname.find(".mov")!=std::string::npos || fname.find(".asf")!=std::string::npos || fname.find(".divx")!=std::string::npos || fname.find(".flv")!=std::string::npos || fname.find(".mpg")!=std::string::npos || fname.find(".m1v")!=std::string::npos || fname.find(".m2v")!=std::string::npos || fname.find(".m4v")!=std::string::npos || fname.find(".mjp")!=std::string::npos || fname.find(".mkv")!=std::string::npos || fname.find(".mpe")!=std::string::npos || fname.find(".movie")!=std::string::npos || fname.find(".ogm")!=std::string::npos || fname.find(".ogg")!=std::string::npos || fname.find(".qt")!=std::string::npos || fname.find(".rm")!=std::string::npos || fname.find(".vob")!=std::string::npos || fname.find(".wmv")!=std::string::npos || fname.find(".xvid")!=std::string::npos || fname.find(".mpeg")!=std::string::npos )
            wimage->getCImgList().load_ffmpeg(fname.c_str());
        else if (fname.find(".hdr")!=std::string::npos || fname.find(".nii")!=std::string::npos)
        {
            float voxsize[3];
            wimage->getCImgList().push_back(CImg<T>().load_analyze(fname.c_str(),voxsize));
            for(unsigned int i=0; i<3; i++) wtransform->getScale()[i]=(Real)voxsize[i];
        }
        else if (fname.find(".inr")!=std::string::npos)
        {
            float voxsize[3];
            wimage->getCImgList().push_back(CImg<T>().load_inr(fname.c_str(),voxsize));
            for(unsigned int i=0; i<3; i++) wtransform->getScale()[i]=(Real)voxsize[i];
        }
        else wimage->getCImgList().push_back(CImg<T>().load(fname.c_str()));

        if(wimage->getCImg()) sout << "Loaded image " << fname <<" ("<< wimage->getCImg().pixel_type() <<")"  << sendl;
        else return false;

        return true;
    }


    bool loadCamera()
    {
        if (this->m_filename.isSet()) return false;
        if(this->name.getValue().find("CAMERA")==std::string::npos) return false;

#ifdef cimg_use_opencv
        waImage wimage(this->image);
        if(!wimage->getCImgList().size()) wimage->getCImgList().push_back(CImg<T>().load_camera());
        else wimage->getCImgList()[0].load_camera();
        if(wimage->getCImg())  return true;  else return false;
#else
        return false;
#endif
    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if (dynamic_cast<simulation::AnimateEndEvent*>(event)) loadCamera();
    }


    void getCorners(Vec<8,Vector3> &c) // get image corners
    {
        raImage rimage(this->image);
        const imCoord dim= rimage->getDimensions();

        Vec<8,Vector3> p;
        p[0]=Vector3(-0.5,-0.5,-0.5);
        p[1]=Vector3(dim[0]-0.5,-0.5,-0.5);
        p[2]=Vector3(-0.5,dim[1]-0.5,-0.5);
        p[3]=Vector3(dim[0]-0.5,dim[1]-0.5,-0.5);
        p[4]=Vector3(-0.5,-0.5,dim[2]-0.5);
        p[5]=Vector3(dim[0]-0.5,-0.5,dim[2]-0.5);
        p[6]=Vector3(-0.5,dim[1]-0.5,dim[2]-0.5);
        p[7]=Vector3(dim[0]-0.5,dim[1]-0.5,dim[2]-0.5);

        raTransform rtransform(this->transform);
        for(unsigned int i=0; i<p.size(); i++) c[i]=rtransform->fromImage(p[i]);
    }

    virtual void computeBBox(const core::ExecParams*  params )
    {
        if (!drawBB.getValue()) return;
        Vec<8,Vector3> c;
        getCorners(c);

        Real bbmin[3]  = {c[0][0],c[0][1],c[0][2]} , bbmax[3]  = {c[0][0],c[0][1],c[0][2]};
        for(unsigned int i=1; i<c.size(); i++)
            for(unsigned int j=0; j<3; j++)
            {
                if(bbmin[j]>c[i][j]) bbmin[j]=c[i][j];
                if(bbmax[j]<c[i][j]) bbmax[j]=c[i][j];
            }
        this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(bbmin,bbmax));
    }

    void draw(const core::visual::VisualParams* vparams)
    {
        // draw bounding box

        if (!vparams->displayFlags().getShowVisualModels()) return;
        if (!drawBB.getValue()) return;

        glPushAttrib( GL_LIGHTING_BIT || GL_ENABLE_BIT || GL_LINE_BIT );
        glPushMatrix();

        const float color[]= {1.,0.5,0.5,0.}, specular[]= {0.,0.,0.,0.};
        glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);
        glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,specular);
        glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,0.0);
        glColor4fv(color);
        glLineWidth(2.0);

        Vec<8,Vector3> c;
        getCorners(c);

        glBegin(GL_LINE_LOOP);	glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[2][0],c[2][1],c[2][2]);	glEnd ();
        glBegin(GL_LINE_LOOP);  glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[4][0],c[4][1],c[4][2]); glVertex3d(c[6][0],c[6][1],c[6][2]); glVertex3d(c[2][0],c[2][1],c[2][2]);	glEnd ();
        glBegin(GL_LINE_LOOP);	glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[5][0],c[5][1],c[5][2]); glVertex3d(c[4][0],c[4][1],c[4][2]);	glEnd ();
        glBegin(GL_LINE_LOOP);	glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[5][0],c[5][1],c[5][2]);	glEnd ();
        glBegin(GL_LINE_LOOP);	glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[5][0],c[5][1],c[5][2]); glVertex3d(c[4][0],c[4][1],c[4][2]); glVertex3d(c[6][0],c[6][1],c[6][2]);	glEnd ();
        glBegin(GL_LINE_LOOP);	glVertex3d(c[2][0],c[2][1],c[2][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[6][0],c[6][1],c[6][2]);	glEnd ();


        glPopMatrix ();
        glPopAttrib();
    }
};






}

}

}


#endif /*IMAGE_IMAGECONTAINER_H*/
