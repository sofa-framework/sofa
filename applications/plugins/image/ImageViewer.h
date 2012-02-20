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
#include "VectorVis.h"

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

/**
* \brief This component is responsible for displaying images in SOFA
*
*  ImageViewer scene options:
*
*  <b>image</b> - a link to the imge in the ImageContainer component
*
*  <b>transform</b> - a link to the transformation in the ImageContainer component
*
*  <b>subsample</b> - two integers representing the default subsample values for the XY plane and the Z plane respectively. Default values are (12 12).
*
*  <b>shape</b> - If true, an image that contains vector information will display the vectors using a shape (arrows or ellipsoids). Default value is false.
*
*  <b>scale</b> - A float value of the scale (size) of the shape. Default value is 11.75.
*
*  <b>histogramValues</b> - Alias include: <b>defaultHistogram</b>, <b>defaultHisto</b>, <b>histoValues</b>.
*    Two floats representing the minimum and maximum windowing (or clamping) values. Default gives no windowing.
*
*  <b>defaultSlices</b> - Three integers describing the x, y and z slices to be displayed initially. Default displays the middle slice in each plane.
*
*  <b>defaultRgb</b> - If true, an image that contains vector information will be displayed as an RGB image. Default value is false.
*
*/
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

    // @name Histogram
    /**@{*/
    typedef Histogram<T> HistogramType;
    typedef helper::WriteAccessor<Data< HistogramType > > waHisto;
    typedef helper::ReadAccessor<Data< HistogramType > > raHisto;
    Data< HistogramType > histo;
    /**@}*/

    //@name Transform data
    /**@{*/
    typedef SReal Real;
    typedef ImageLPTransform<Real> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;
    /**@}*/

    // @name Plane selection
    /**@{*/
    typedef ImagePlane<T> ImagePlaneType;
    typedef helper::ReadAccessor<Data< ImagePlaneType > > raPlane;
    typedef helper::WriteAccessor<Data< ImagePlaneType > > waPlane;
    Data< ImagePlaneType > plane;
    /**@}*/

    // @name Vector visualization
    /**@{*/
    typedef helper::ReadAccessor<Data< VectorVis > > raVis;
    typedef helper::WriteAccessor<Data< VectorVis > > waVis;
    Data<VectorVis> vectorVisualization;
    /**@}*/

    typedef component::visualmodel::VisualModelImpl VisuModelType;

    //@name Options
    /**@{*/
    Data<Vec<2,int> > subsample;
    Data<bool> shapeDefault;
    Data<int> scale;
    Data<bool> defaultRgb;
    Data< sofa::defaulttype::Vec<2,SReal>  > defaultHistogram;
    Data< sofa::defaulttype::Vec<3,unsigned int>  > defaultSlices;
    /**@}*/

    std::string getTemplateName() const  {	return templateName(this);	}
    static std::string templateName(const ImageViewer<ImageTypes>* = NULL)	{ return ImageTypes::Name(); }

    ImageViewer() : Inherited()
        , image(initData(&image,ImageTypes(),"image","input image"))
        , histo(initData(&histo, HistogramType(256,256,false),"histo",""))
        , transform(initData(&transform, TransformType(), "transform" , ""))
        , plane ( initData ( &plane, ImagePlaneType(), "plane" , "" ) )
        , vectorVisualization ( initData (&vectorVisualization, VectorVis(), "vectorvis", ""))
        , subsample ( initData ( &subsample, "subsample", "Two integer values of the default subsampling value for the XY and Z directions for vector visualization." ))
        , shapeDefault ( initData ( &shapeDefault, false, "shape", "True if vectors are to be visualized as a shape. Default value is false." ))
        , scale ( initData ( &scale, 11, "scale", "The scale of the vectors being visualized. Default value is 11." ))
        , defaultRgb ( initData (&defaultRgb, true, "rgb", "True if vectors are to be visualized as RGB values, false if vectors are to be visualized as norms. Default is true."))
        , defaultHistogram ( initData (&defaultHistogram, "histogramValues", "The default min and max values for the histogram"))
        , defaultSlices ( initData (&defaultSlices, "defaultSlices", "The default slices to be displayed"))
    {
        this->addAlias(&image, "outputImage");
        this->addAlias(&transform, "outputTransform");
        this->addAlias(&defaultHistogram, "defaultHistogram");
        this->addAlias(&defaultHistogram, "histoValues");
        this->addAlias(&defaultHistogram, "defaultHisto");

        image.setGroup("Image");
        image.setReadOnly(true);

        histo.setGroup("Histogram");
        histo.setWidget("imagehistogram");

        transform.setGroup("Transform");
        transform.setReadOnly(true);

        plane.setGroup("Image");
        plane.setWidget("imageplane");

        vectorVisualization.setGroup("Vectors");
        vectorVisualization.setWidget("vectorvis");

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

        waHisto whisto(this->histo);
        whisto->setInput(image.getValue());

        waPlane wplane(this->plane);
        wplane->setInput(image.getValue(),transform.getValue(),visuals);

        waVis wVis(this->vectorVisualization);
        wVis->setShape(shapeDefault.getValue());

        if(subsample.isSet())
        {
            wVis->setSubsampleXY(subsample.getValue()[0]);
            wVis->setSubsampleZ(subsample.getValue()[1]);
        }
        if(scale.isSet())
            wVis->setShapeScale(scale.getValue());

        if(defaultHistogram.isSet())
        {
            whisto->setClamp(defaultHistogram.getValue());
        }

        if(defaultSlices.isSet())
            wplane->setPlane(defaultSlices.getValue());
        else
            setToMiddleSlices();

        if(defaultRgb.isSet())
        {
            wVis->setRgb(defaultRgb.getValue());
        }

        whisto->setVectorVis( &(vectorVisualization.getValue()));
        whisto->update();

        wplane->setVectorVis( &(vectorVisualization.getValue()));

        for(unsigned int i=0; i<3; i++)
        {
            cutplane_tex[i]= new helper::gl::Texture(new helper::io::Image,false);
            cutplane_tex[i]->getImage()->init(cutplane_res,cutplane_res,32);
        }

        updateTextures();

        for(unsigned int i=0; i<3; i++)	cutplane_tex[i]->init();


        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    }

    //Set the displayed slices the the middle slice of each plane
    void setToMiddleSlices()
    {
        raImage rimage(this->image);
        waPlane wplane(this->plane);
        imCoord dimensions = rimage->getDimensions();
        int midX = static_cast<int>(dimensions[0] / 2);
        int midY = static_cast<int>(dimensions[1] / 2);
        int midZ = static_cast<int>(dimensions[2] / 2);

        Vec<3,unsigned int> newPlanes(midX, midY, midZ);
        wplane->setPlane(newPlanes);

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

        if(vectorVisualization.getValue().getShape())
        {
            raImage rimage(this->image);
            if(rimage->getCImg().spectrum() == 3)
            {
                drawArrowsZ();
                drawArrowsY();
                drawArrowsX();
            }
        }

        glPushAttrib( GL_LIGHTING_BIT || GL_ENABLE_BIT || GL_LINE_BIT || GL_CURRENT_BIT);
        drawCutplanes();
        glPopAttrib();

    }



protected:

    static const unsigned cutplane_res=256;
    helper::gl::Texture* cutplane_tex[3];


    /**
    * Draws an arrow
    * @param from the starting point of the arrow
    * @param arrow the tip of the arrow relative to the origin
    */
    void drawArrow(Vec<3,double>& from, Vec<3,double>& arrow, double lineWidth = 2.0)
    {
        const double epsilon = 1e-15;

        if(fabs(arrow[0]) > epsilon|| arrow[1] > epsilon ||
           arrow[2] > epsilon)
        {
            Vec<3,double> org(from);
            Vec<3,double> head(from);
            head += arrow;

            double d = arrow.norm();

            Vec<3,double> vz = arrow / d;
            Vec<3,double> vx,vy;
            if(fabs(vz[2]) >= epsilon)
            {
                vx[0] = 1.0;
                vx[1] = 1.0;
                vx[2] = (fabs(vz[0]) < epsilon && fabs(vz[1]) < epsilon)
                        ? 1.0
                        : -(vz[0] * vx[0] + vz[1] * vx[1]) / vz[2];
            }
            else
            {
                if(fabs(vz[0]) < epsilon)
                    vx = Vec<3,double>(1.0, 0.0, 1.0);
                else
                {
                    vx[1] = 1.0;
                    vx[2] = 1.0;
                    vx[0] = -(vz[1] * vx[1] - vz[2] * vx[2]) / vz[0];
                }
            }
            vx /= vx.norm();
            vy = cross<double>(vz, vx);


            d *= 0.15;

            Vec<3, double> temp;
            temp[0] = arrow[0] * 0.67;
            temp[1] = arrow[1] * 0.67;
            temp[2] = arrow[2] * 0.67;

            Vec<3, double> A;
            A[0] = org[0] + arrow[0];
            A[1] = org[1] + arrow[1];
            A[2] = org[2] + arrow[2];

            Vec<3,double> b1 = vx*d;
            Vec<3,double> b2 = vy*d;

            Vec<3,double> B(A);
            B -= vz*d;
            B += b1;

            Vec<3,double> C(A);
            C -= vz*d;
            C += b2;

            Vec<3,double> D(A);
            D -= vz*d;
            D -= b1;

            Vec<3,double> E(A);
            E -= vz*d;
            E -= b2;

            glLineWidth( (GLdouble)lineWidth);
            glBegin(GL_LINES);

            glVertex3d(org[0], org[1], org[2]);
            glVertex3d(head[0], head[1], head[2]);
            glVertex3d(B[0], B[1], B[2]);
            glVertex3d(C[0], C[1], C[2]);
            glVertex3d(C[0], C[1], C[2]);
            glVertex3d(D[0], D[1], D[2]);
            glVertex3d(D[0], D[1], D[2]);
            glVertex3d(E[0], E[1], E[2]);
            glVertex3d(E[0], E[1], E[2]);
            glVertex3d(B[0], B[1], B[2]);
            glVertex3d(B[0], B[1], B[2]);
            glVertex3d(head[0], head[1], head[2]);
            glVertex3d(C[0], C[1], C[2]);
            glVertex3d(head[0], head[1], head[2]);
            glVertex3d(D[0], D[1], D[2]);
            glVertex3d(head[0], head[1], head[2]);
            glVertex3d(E[0], E[1], E[2]);
            glVertex3d(head[0], head[1], head[2]);

            glEnd();

        }


    }

    //Draw arrows in the Z plane
    void drawArrowsZ()
    {
        int x=0;
        int y=1;
        int z=2;

        waVis wVis(this->vectorVisualization);
        if(!wVis->getRgb())
        {
            std::vector<VisuModelType*> visuals;
            sofa::core::objectmodel::BaseContext* context = this->getContext();
            context->get<VisuModelType>(&visuals,core::objectmodel::BaseContext::SearchRoot);

            waPlane wplane(this->plane);
            wplane->setInput(image.getValue(),transform.getValue(),visuals);
        }

        raImage rimage(this->image);
        raPlane rplane(this->plane);
        raTransform rtransform(this->transform);
        raVis rVis(this->vectorVisualization);

        int sliceN = rplane->getPlane()[z];
        double voxelSizeX = rtransform->getScale()[x];
        double voxelSizeY = rtransform->getScale()[y];
        double voxelSizeZ = rtransform->getScale()[z];

        double size = rVis->getShapeScale();
        int sampleXY = rVis->getSubsampleXY();
        int width = rimage->getCImg().width();
        int height = rimage->getCImg().height();

        for(int xOffset=0; xOffset < width; xOffset += sampleXY)
        {
            for(int yOffset=0; yOffset < height; yOffset += sampleXY)
            {
                const T* voxel = rimage->getCImg(rplane->getTime()).data(xOffset, yOffset, sliceN);

                Vec<3,double> base = rtransform->fromImage(Vec<3,Real>(xOffset, yOffset, sliceN));
                Vec<3, double> relativeVec((double)voxel[x], (double)voxel[y], (double)voxel[z]);
                relativeVec = relativeVec * size;

                drawArrow(base, relativeVec);
            }
        }

    }

    //Draw arrows in the Y plane
    void drawArrowsY()
    {
        int x=0;
        int y=1;
        int z=2;

        raImage rimage(this->image);
        raPlane rplane(this->plane);
        raTransform rtransform(this->transform);
        raVis rVis(this->vectorVisualization);

        int sliceN = rplane->getPlane()[y];
        double voxelSizeX = rtransform->getScale()[x];
        double voxelSizeY = rtransform->getScale()[y];
        double voxelSizeZ = rtransform->getScale()[z];

        double size = rVis->getShapeScale();
        int sampleXY = rVis->getSubsampleXY();
        int sampleZ = rVis->getSubsampleZ();
        int width = rimage->getCImg().width();
        int depth = rimage->getCImg().depth();

        for(int xOffset=0; xOffset < width; xOffset += sampleXY)
        {
            for(int zOffset=0; zOffset < depth; zOffset += sampleZ)
            {
                const T* voxel = rimage->getCImg(rplane->getTime()).data(xOffset, sliceN, zOffset);

                Vec<3,double> base = rtransform->fromImage(Vec<3,Real>(xOffset, sliceN, zOffset));
                Vec<3, double> relativeVec((double)voxel[x], (double)voxel[y], (double)voxel[z]);
                relativeVec = relativeVec * size;

                drawArrow(base, relativeVec);
            }
        }
    }


    //Draw arrows in the X plane
    void drawArrowsX()
    {
        int x=0;
        int y=1;
        int z=2;

        raImage rimage(this->image);
        raPlane rplane(this->plane);
        raTransform rtransform(this->transform);

        int sliceN = rplane->getPlane()[x];
        double voxelSizeX = rtransform->getScale()[x];
        double voxelSizeY = rtransform->getScale()[y];
        double voxelSizeZ = rtransform->getScale()[z];
        raVis rVis(this->vectorVisualization);

        double size = rVis->getShapeScale();
        int sampleXY = rVis->getSubsampleXY();
        int sampleZ = rVis->getSubsampleZ();
        int depth = rimage->getCImg().depth();
        int height = rimage->getCImg().height();

        for(int zOffset=0; zOffset < depth; zOffset += sampleZ)
        {
            for(int yOffset=0; yOffset < height; yOffset += sampleXY)
            {
                const T* voxel = rimage->getCImg(rplane->getTime()).data(sliceN, yOffset, zOffset);

                Vec<3,double> base = rtransform->fromImage(Vec<3,Real>(sliceN, yOffset, zOffset));
                Vec<3, double> relativeVec((double)voxel[x], (double)voxel[y], (double)voxel[z]);
                relativeVec = relativeVec * size;

                drawArrow(base, relativeVec);
            }
        }

    }

    //Draw the boxes around the slices
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

                            //Outline
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


    //Update and draw the slices
    void updateTextures()
    {
        raPlane rplane(this->plane);
        if (!rplane->getDimensions()[0]) return;
        raHisto rhisto(this->histo);

        for (unsigned int i=0; i<3; i++)
        {
            CImg<T> originalPlane = rplane->get_slice(rplane->getPlane()[i],i).resize(cutplane_res,cutplane_res,1,-100,1);


            //Calculates the norm of the vector and sets it as the value of the voxel
            if(!vectorVisualization.getValue().getRgb())
            {
                if(originalPlane)
                {
                    cimg_forXY(originalPlane,x,y)
                    {
                        CImg<T> vector = originalPlane.get_vector_at(x,y,0);

                        vector(0,0,0,0) = (T) vector.magnitude();
                        if(vector.height() > 1)
                            vector(0,1,0,0) = (T) vector(0,0,0,0);
                        if(vector.height() > 2)
                            vector(0,2,0,0) = (T) vector(0,0,0,0);

                        originalPlane.set_vector_at(vector, x, y, 0);
                    }
                }
            }

            CImg<unsigned char> plane;

            plane = convertToUC( originalPlane.cut(rhisto->getClamp()[0],rhisto->getClamp()[1]) );

            if(plane)
            {
                cimg_forXY(plane,x,y)
                {
                    unsigned char *b=cutplane_tex[i]->getImage()->getPixels()+4*(y*cutplane_res+x);


                    for(unsigned int c=0; c<3 && c<(unsigned int)plane.spectrum() ; c++)
                    {

                        b[c]=plane(x,y,0,c);
                    }
                    for(unsigned int c=plane.spectrum(); c<3; c++)
                        b[c]=b[0];
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
