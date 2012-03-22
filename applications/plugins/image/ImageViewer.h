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
    typedef typename TransformType::Coord Coord;
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

    std::string getTemplateName() const  {	return templateName(this);	}
    static std::string templateName(const ImageViewer<ImageTypes>* = NULL)	{ return ImageTypes::Name(); }

    ImageViewer() : Inherited()
        , image(initData(&image,ImageTypes(),"image","input image"))
        , histo(initData(&histo, HistogramType(256,256,false),"histo",""))
        , transform(initData(&transform, TransformType(), "transform" , ""))
        , plane ( initData ( &plane, ImagePlaneType(), "plane" , "" ) )
        , vectorVisualization ( initData (&vectorVisualization, VectorVis(), "vectorvis", ""))
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

        vectorVisualization.setWidget("vectorvis");

        for(unsigned int i=0; i<3; i++)	cutplane_tex[i]=NULL;
    }


    virtual ~ImageViewer()
    {
        for(unsigned int i=0; i<3; i++)	if(cutplane_tex[i]) delete cutplane_tex[i];
    }

    virtual void init()
    {

        // getvisuals
        std::vector<VisuModelType*> visuals;
        sofa::core::objectmodel::BaseContext* context = this->getContext();
        context->get<VisuModelType>(&visuals,core::objectmodel::BaseContext::SearchRoot);

        // set histogram data
        waHisto whisto(this->histo);        whisto->setInput(image.getValue());

        // record user values of plane position
        typename ImagePlaneType::pCoord pc; bool usedefaultplanecoord=true;
        if(this->plane.isSet()) {  raPlane rplane(this->plane); pc=rplane->getPlane(); usedefaultplanecoord=false; }

        // set plane data
        waPlane wplane(this->plane);        wplane->setInput(image.getValue(),transform.getValue(),visuals);

        // set recorded plane pos
        if(!usedefaultplanecoord) wplane->setPlane(pc);

        // enable vecorvis ?
        if(wplane->getDimensions()[3]<2) vectorVisualization.setDisplayed(false);
        else         vectorVisualization.setGroup("Vectors");

        for(unsigned int i=0; i<3; i++)
        {
            cutplane_tex[i]= new helper::gl::Texture(new helper::io::Image,false);
            cutplane_tex[i]->getImage()->init(cutplane_res,cutplane_res,32);
        }

        raVis rvis(this->vectorVisualization);

        whisto->setMergeChannels(!rvis->getRgb());
        wplane->setMergeChannels(!rvis->getRgb());
        wplane->setClamp(whisto->getClamp());

        updateTextures();

        for(unsigned int i=0; i<3; i++)	cutplane_tex[i]->init();

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }


    virtual void reinit()
    {
        waHisto whisto(this->histo);
        waPlane wplane(this->plane);
        raVis rvis(this->vectorVisualization);

        whisto->setMergeChannels(!rvis->getRgb());
        wplane->setMergeChannels(!rvis->getRgb());
        wplane->setClamp(whisto->getClamp());
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
            if(wplane->getDimensions()[3] == 3)
                drawArrows(vparams);
            if(wplane->getDimensions()[3] == 6)
                drawEllipsoid();
        }
        glPushAttrib( GL_LIGHTING_BIT || GL_ENABLE_BIT || GL_LINE_BIT || GL_CURRENT_BIT);
        drawCutplanes();
        glPopAttrib();

    }



protected:

    static const unsigned cutplane_res=256;
    helper::gl::Texture* cutplane_tex[3];

    //Draw vectors as arrows
    void drawArrows(const core::visual::VisualParams* vparams)
    {
        raImage rimage(this->image);
        raPlane rplane(this->plane);
        raTransform rtransform(this->transform);
        raVis rVis(this->vectorVisualization);

        double size = rVis->getShapeScale();
        imCoord dims=rplane->getDimensions();
        Vec<3,int> sampling(rVis->getSubsampleXY(),rVis->getSubsampleXY(),rVis->getSubsampleZ());
        Vec4f colour(1.0,0.5,0.5,1.0);

        unsigned int x,y,z;
        for (z=0; z<3; z++)
        {
            if(z==0) { x=2; y=1;}
            else if(z==1) { x=0; y=2;}
            else { x=0; y=1;}
            Coord ip; ip[z]=rplane->getPlane()[z];
            if(ip[z]<dims[z] && dims[x]>1 && dims[y]>1)
                for(ip[x] = 0; ip[x] < dims[x]; ip[x] += sampling[x])
                    for(ip[y] = 0; ip[y] < dims[y]; ip[y] += sampling[y])
                    {
                        Coord base = rtransform->fromImage(ip);
                        CImg<T> vect = rimage->getCImg(rplane->getTime()).get_vector_at(ip[0],ip[1],ip[2]);
                        Coord relativeVec((double)vect[0], (double)vect[1], (double)vect[2]);
                        vparams->drawTool()->drawArrow(base,base+relativeVec*size,size,colour);
                    }
        }
    }

    //Draw tensors as ellipsoids
    void drawEllipsoid()
    {
        raImage rimage(this->image);
        raPlane rplane(this->plane);
        raTransform rtransform(this->transform);
        raVis rVis(this->vectorVisualization);

        double size = rVis->getShapeScale();
        imCoord dims=rplane->getDimensions();
        Vec<3,int> sampling(rVis->getSubsampleXY(),rVis->getSubsampleXY(),rVis->getSubsampleZ());

        int counter=0;

        unsigned int x,y,z;
        for (z=0; z<3; z++)
        {
            if(z==0) { x=2; y=1;}
            else if(z==1) { x=0; y=2;}
            else { x=0; y=1;}
            Coord ip; ip[z]=rplane->getPlane()[z];
            if(ip[z]<dims[z] && dims[x]>1 && dims[y]>1)
                for(ip[x] = 0; ip[x] < dims[x]; ip[x] += sampling[x])
                    for(ip[y] = 0; ip[y] < dims[y]; ip[y] += sampling[y])
                    {

                        counter++ ;

                        Coord base = rtransform->fromImage(ip);

                        CImg<T> vector = rimage->getCImg(rplane->getTime()).get_vector_at(ip[0], ip[1], ip[2]);

                        //CImg::get_tensor_at() assumes a different tensor input than we expect.
                        // That is why we are generating the tensor manually from the vector instead.
                        CImg<T> tensor;

                        if(rVis->getTensorOrder().compare("LowerTriRowMajor") == 0)
                        {
                            tensor = computeTensorFromLowerTriRowMajorVector(vector);
                        }
                        else if(rVis->getTensorOrder().compare("UpperTriRowMajor") == 0)
                        {
                            tensor = computeTensorFromUpperTriRowMajorVector(vector);
                        }
                        else if(rVis->getTensorOrder().compare("DiagonalFirst") == 0)
                        {
                            tensor = computeTensorFromDiagonalFirstVector(vector);
                        }
                        else
                        {
                            serr << "ImageViewer: Tensor input order \"" << rVis->getTensorOrder() << "\" is not valid." << sout;
                            return;
                        }

                        CImgList<T> eig = tensor.get_symmetric_eigen();
                        const CImg<T> &val = eig[0];
                        const CImg<T> &vec = eig[1];

                        glPushMatrix();

                        GLUquadricObj* ellipsoid = gluNewQuadric();
                        glTranslated(base[0], base[1], base[2]);
                        GLdouble transformMatrix[16];

                        int index=0;
                        for(int i=0; i<vec.width(); i++)
                        {
                            for(int j=0; j<vec.height(); j++)
                            {
                                transformMatrix[index] = (double)vec(i,j);
                                index++;
                            }
                            transformMatrix[index] = 0;
                            index++;
                        }
                        transformMatrix[12] = transformMatrix[13] = transformMatrix[14] = 0;
                        transformMatrix[15] = 1;


                        glMultMatrixd(transformMatrix);
                        glScaled((double)val(0)*size/10, (double)val(1)*size/10, (double)val(2)*size/10);
                        gluSphere(ellipsoid, 1.0, 10, 10);
                        gluDeleteQuadric(ellipsoid);

                        glPopMatrix();

                    }
        }

    }

    CImg<T> computeTensorFromLowerTriRowMajorVector(CImg<T> vector)
    {
        CImg<T> tensor(3,3);
        tensor(0,0) = vector(0);
        tensor(1,0) = tensor(0,1) = vector(1);
        tensor(1,1) = vector(2);
        tensor(2,0) = tensor(0,2) = vector(3);
        tensor(2,1) = tensor(1,2) = vector(4);
        tensor(2,2) = vector(5);

        return tensor;
    }

    CImg<T> computeTensorFromUpperTriRowMajorVector(CImg<T> vector)
    {
        CImg<T> tensor(3,3);
        tensor(0,0) = vector(0);
        tensor(0,1) = tensor(1,0) = vector(1);
        tensor(0,2) = tensor(2,0) = vector(2);
        tensor(1,1) = vector(3);
        tensor(1,2) = tensor(2,1) = vector(4);
        tensor(2,2) = vector(5);

        return tensor;
    }

    CImg<T> computeTensorFromDiagonalFirstVector(CImg<T> vector)
    {
        CImg<T> tensor(3,3);
        tensor(0,0) = vector(0);
        tensor(1,1) = vector(1);
        tensor(2,2) = vector(2);
        tensor(1,0) = tensor(0,1) = vector(3);
        tensor(2,0) = tensor(0,2) = vector(4);
        tensor(2,1) = tensor(1,2) = vector(5);
        return tensor;
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

        for (unsigned int i=0; i<3; i++)
        {
            CImg<unsigned char> cplane = convertToUC( rplane->get_slice(rplane->getPlane()[i],i).resize(cutplane_res,cutplane_res,1,-100,1).cut(rplane->getClamp()[0],rplane->getClamp()[1]) );

            if(cplane)
            {
                cimg_forXY(cplane,x,y)
                {
                    unsigned char *b=cutplane_tex[i]->getImage()->getPixels()+4*(y*cutplane_res+x);
                    for(unsigned int c=0; c<3 && c<(unsigned int)cplane.spectrum() ; c++) b[c]=cplane(x,y,0,c);
                    for(unsigned int c=cplane.spectrum(); c<3; c++) b[c]=b[0];
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
