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
#ifndef SOFA_IMAGE_IMAGEVIEWER_H
#define SOFA_IMAGE_IMAGEVIEWER_H


#include "initImage.h"
#include "ImageTypes.h"

#include <sofa/component/component.h>
#include <sofa/helper/io/Image.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/component/visualmodel/VisualModelImpl.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/CollisionModel.h>

namespace sofa
{

namespace component
{

namespace misc
{

using namespace defaulttype;

template<class _ImageTypes>
class SOFA_IMAGE_API ImageViewer : public sofa::core::objectmodel::BaseObject
{
public:
    typedef sofa::core::objectmodel::BaseObject Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageViewer, _ImageTypes), Inherited);

    // image data
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;

    // histogram
    typedef Histogram<T> HistogramType;
    typedef helper::WriteAccessor<Data< HistogramType > > waHisto;
    typedef helper::ReadAccessor<Data< HistogramType > > raHisto;
    Data< HistogramType > histo;

    // transform data
    typedef SReal Real;
    typedef ImageLPTransform<Real> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    // plane selection
    typedef ImagePlane<T> ImagePlaneType;
    typedef helper::ReadAccessor<Data< ImagePlaneType > > raPlane;
    typedef helper::WriteAccessor<Data< ImagePlaneType > > waPlane;
    Data< ImagePlaneType > plane;

    typedef component::visualmodel::VisualModelImpl VisuModelType;

    std::string getTemplateName() const  {	return templateName(this);	}
    static std::string templateName(const ImageViewer<ImageTypes>* = NULL)	{ return ImageTypes::Name(); }

    ImageViewer() : Inherited()
        , image(initData(&image,ImageTypes(),"image","input image"))
        , histo(initData(&histo, HistogramType(256,256,false),"histo",""))
        , transform(initData(&transform, TransformType(), "transform" , ""))
        , plane ( initData ( &plane, ImagePlaneType(), "plane" , "" ) )
    {
        this->addAlias(&image, "outputImage");
        this->addAlias(&transform, "outputTransform");

        image.setGroup("Image");
        image.setReadOnly(true);

        histo.setGroup("Histogram");
        histo.setWidget("imagehistogram");

        transform.setGroup("Transform");
        transform.setReadOnly(true);

        plane.setGroup("Image");
        plane.setWidget("imageplane");

        for(unsigned int i=0; i<3; i++)	cutplane_tex[i]=NULL;
    }


    virtual ~ImageViewer()
    {
        for(unsigned int i=0; i<3; i++)	if(cutplane_tex[i]) delete cutplane_tex[i];
    }

    virtual void init()
    {
        std::vector<VisuModelType*> visuals;
        sofa::core::objectmodel::BaseContext* context = this->getContext();
        context->get<VisuModelType>(&visuals,core::objectmodel::BaseContext::SearchRoot);

        waHisto whisto(this->histo);	whisto->setInput(image.getValue());
        waPlane wplane(this->plane);	wplane->setInput(image.getValue(),transform.getValue(),visuals);
        for(unsigned int i=0; i<3; i++)
        {
            cutplane_tex[i]= new helper::gl::Texture(new helper::io::Image,false);
            cutplane_tex[i]->getImage()->init(cutplane_res,cutplane_res,32);
        }
        updateTextures();
        for(unsigned int i=0; i<3; i++)	cutplane_tex[i]->init();
    }

    virtual void reinit()
    {
        raHisto rhisto(this->histo);
        waPlane wplane(this->plane);
        wplane->setClamp(rhisto->getClamp());
    }

    virtual void handleEvent( core::objectmodel::Event* )
    {
    }

    virtual void draw(const core::visual::VisualParams* vparams)
    {
        if (!vparams->displayFlags().getShowVisualModels()) return;

        waPlane wplane(this->plane);
        wplane->setTime( this->getContext()->getTime() );

        bool imagedirty=image.isDirty();
        if(imagedirty)
        {
            raImage rimage(this->image);			// used here to propagate changes across linked data
            waHisto whisto(this->histo);
            whisto->update();
        }

        if(wplane->isImagePlaneDirty() || transform.isDirty() || imagedirty)
        {
            raTransform rtransform(this->transform); // used here to propagate changes across linked data
            raImage rimage(this->image);			// used here to propagate changes across linked data
            wplane->setImagePlaneDirty(false);
            updateTextures();
        }

        glPushAttrib( GL_LIGHTING_BIT || GL_ENABLE_BIT || GL_LINE_BIT || GL_CURRENT_BIT);
        drawCutplanes();
        glPopAttrib();
    }


protected:

    static const unsigned cutplane_res=256;
    helper::gl::Texture* cutplane_tex[3];

    void drawCutplanes()
    {
        raPlane rplane(this->plane);
        if (!rplane->getDimensions()[0]) return;

        raTransform rtransform(this->transform); // used here to propagate changes in linked data..

        float color[]= {1.,1.,1.,0.}, specular[]= {0.,0.,0.,0.};
        glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);
        glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,specular);
        glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,0.0);
        glColor4fv(color);

        for (unsigned int i=0; i<3; i++)
            if(rplane->getPlane()[i]<rplane->getDimensions()[i])
                if(i==0 || rplane->getDimensions()[0]>1)
                    if(i==1 || rplane->getDimensions()[1]>1)
                        if(i==2 || rplane->getDimensions()[2]>1)
                        {
                            Vec<4,Vector3> pts = rplane->get_sliceCoord(rplane->getPlane()[i],i);
                            Vector3 n=cross(pts[1]-pts[0],pts[2]-pts[0]); n.normalize();

                            glEnable( GL_TEXTURE_2D );
                            glDisable( GL_LIGHTING);
                            cutplane_tex[i]->bind();
                            glBegin(GL_QUADS);
                            glNormal3d(n[0],n[1],n[2]);
                            glTexCoord2d(0,0); glVertex3d(pts[0][0],pts[0][1],pts[0][2]);
                            glTexCoord2d(1,0); glVertex3d(pts[1][0],pts[1][1],pts[1][2]);
                            glTexCoord2d(1,1); glVertex3d(pts[2][0],pts[2][1],pts[2][2]);
                            glTexCoord2d(0,1); glVertex3d(pts[3][0],pts[3][1],pts[3][2]);
                            glEnd ();
                            cutplane_tex[i]->unbind();
                            glDisable( GL_TEXTURE_2D );
                            glEnable( GL_LIGHTING);

                            glLineWidth(2.0);
                            glBegin(GL_LINE_LOOP);
                            glNormal3d(n[0],n[1],n[2]);
                            glVertex3d(pts[0][0],pts[0][1],pts[0][2]);
                            glVertex3d(pts[1][0],pts[1][1],pts[1][2]);
                            glVertex3d(pts[2][0],pts[2][1],pts[2][2]);
                            glVertex3d(pts[3][0],pts[3][1],pts[3][2]);
                            glEnd ();

                        }
    }

    void updateTextures()
    {
        raPlane rplane(this->plane);
        if (!rplane->getDimensions()[0]) return;
        raHisto rhisto(this->histo);

        for (unsigned int i=0; i<3; i++)
        {
            CImg<unsigned char> plane = convertToUC( rplane->get_slice(rplane->getPlane()[i],i).resize(cutplane_res,cutplane_res,1,-100,1).cut(rhisto->getClamp()[0],rhisto->getClamp()[1]) );

            if(plane)
            {
                cimg_forXY(plane,x,y)
                {
                    unsigned char *b=cutplane_tex[i]->getImage()->getPixels()+4*(y*cutplane_res+x);
                    for(unsigned int c=0; c<3 && c<(unsigned int)plane.spectrum() ; c++) b[c]=plane(x,y,0,c);
                    for(unsigned int c=plane.spectrum(); c<3; c++) b[c]=b[0];
                    b[3]=(unsigned char)(-1);
                }
                cutplane_tex[i]->update();
            }
        }
    }

};


} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_IMAGEVIEWER_H
