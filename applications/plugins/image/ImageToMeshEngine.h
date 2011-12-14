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
#ifndef SOFA_IMAGE_IMAGETOMESHENGINE_H
#define SOFA_IMAGE_IMAGETOMESHENGINE_H

#include "initImage.h"
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateEndEvent.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/Texture.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;
using namespace helper;
using namespace core::topology;

/**
 * This class computes a mesh from a depth map image
 */


template <class _ImageTypes>
class ImageToMeshEngine : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageToMeshEngine,_ImageTypes),Inherited);

    typedef SReal Real;

    Data< Real > depthFactor;
    Data< bool > invert;

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;

    typedef ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    typedef _ImageTypes TextureTypes;
    typedef helper::ReadAccessor<Data< TextureTypes > > raTexture;
    Data< TextureTypes > texImage;

    typedef vector<Vec<3,Real> > SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    typedef helper::WriteAccessor<Data< SeqPositions > > waPositions;
    Data< SeqPositions > position;

    typedef fixed_array<Real,2> TexCoord;
    typedef vector<TexCoord> SeqTexCoords;
    typedef helper::ReadAccessor<Data< SeqTexCoords > > raTexCoords;
    typedef helper::WriteAccessor<Data< SeqTexCoords > > waTexCoords;
    Data< SeqTexCoords > texCoord;
    Data< TexCoord > texOffset;

    typedef typename BaseMeshTopology::Triangle Triangle;
    typedef typename BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef helper::ReadAccessor<Data< SeqTriangles > > raTriangles;
    typedef helper::WriteAccessor<Data< SeqTriangles > > waTriangles;
    Data< SeqTriangles > triangles;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ImageToMeshEngine<ImageTypes>* = NULL) { return ImageTypes::Name();    }

    ImageToMeshEngine()    :   Inherited()
        , depthFactor(initData(&depthFactor,(Real)(1.0),"depthFactor","Intensity to depth factor"))
        , invert(initData(&invert,false,"invert","Invert intensities"))
        , image(initData(&image,ImageTypes(),"image",""))
        , transform(initData(&transform,TransformType(),"transform",""))
        , texImage(initData(&texImage,TextureTypes(),"texImage",""))
        , position(initData(&position,SeqPositions(),"position","output positions"))
        , texCoord(initData(&texCoord,SeqTexCoords(),"texCoord","output texture coordinates"))
        , texOffset(initData(&texOffset,TexCoord(0.0,0.0),"texOffset","texture offsets (in [0,1])"))
        , triangles(initData(&triangles,SeqTriangles(),"triangles","output triangles"))
        , time((unsigned int)0)
        , texture(NULL)
    {
        image.setReadOnly(true);
        transform.setReadOnly(true);
        f_listening.setValue(true);
    }

    virtual ~ImageToMeshEngine()
    {
        if(texture) delete texture;
    }

    virtual void init()
    {
        addInput(&image);
        addInput(&transform);
        addOutput(&position);
        addOutput(&texCoord);
        addOutput(&triangles);
        setDirtyValue();

        texture = new helper::gl::Texture(new helper::io::Image,false);
        texture->getImage()->init(texture_res,texture_res,32);
        texture->init();
    }

    virtual void reinit() { update(); }

protected:

    unsigned int time;
    helper::gl::Texture* texture;
    static const unsigned texture_res=256;

    virtual void update()
    {
        cleanDirty();

        raImage in(this->image);
        raTransform inT(this->transform);
        raTexture inTex(this->texImage);
        const unsigned int dimx=in->getDimensions()[0],dimy=in->getDimensions()[1];

        waPositions pos(this->position);
        waTexCoords tc(this->texCoord);
        waTriangles tri(this->triangles);

        // get image at time t
        const CImg<T>& img = in->getCImg(this->time);
        Real f = this->depthFactor.getValue();

        // update texture
        if(inTex->getCImgList())
        {
            const CImg<T>& tex = inTex->getCImg(this->time);
            CImg<unsigned char> plane=convertToUC( tex.get_resize(texture_res,texture_res,1,-100,1) );
            cimg_forXY(plane,x,y)
            {
                unsigned char *b=texture->getImage()->getPixels()+4*(y*texture_res+x);
                for(unsigned int c=0; c<3 && c<(unsigned int)plane.spectrum() ; c++) b[c]=plane(x,y,0,c);
                for(unsigned int c=plane.spectrum(); c<3; c++) b[c]=b[0];
                b[3]=(unsigned char)(-1);
            }
            texture->update();
        }

        // update points
        unsigned int count=0,p1,p2,p3;
        pos.resize(dimx*dimy);
        tc.resize(dimx*dimy);
        vector<bool> isValid(dimx*dimy);
        for(unsigned int y=0; y<dimy; y++)
            for(unsigned int x=0; x<dimx; x++)
            {
                tc[count][0]=(Real)x/(Real)(dimx-1) + texOffset.getValue()[0];
                tc[count][1]=(Real)y/(Real)(dimy-1) + texOffset.getValue()[1];
                T val=this->invert.getValue()?(cimg::type<T>::max()-img(x,y)):img(x,y);
                if(val==cimg::type<T>::max()) {val=0; isValid[count]=false;}
                else isValid[count]=true;
                pos[count++]=inT->fromImage(Coord((Real)x,(Real)y,(Real)val*f));
            }

        // update triangles
        tri.clear();
        for(unsigned int y=0; y<dimy-1; y++)
            for(unsigned int x=0; x<dimx-1; x++)
            {
                p1=x+y*dimx; p2=x+1+y*dimx; p3=x+(y+1)*dimx;
                if(isValid[p1] && isValid[p2] && isValid[p3]) tri.push_back(Triangle(p1,p2,p3));

                p1=x+1+y*dimx; p2=x+1+(y+1)*dimx; p3=x+(y+1)*dimx;
                if(isValid[p1] && isValid[p2] && isValid[p3]) tri.push_back(Triangle(p1,p2,p3));
            }
    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if ( dynamic_cast<simulation::AnimateEndEvent*>(event))
        {
            raImage in(this->image);
            raTransform inT(this->transform);

            // get current time modulo dimt
            const unsigned int dimt=in->getDimensions()[4];
            Real t=inT->toImage(this->getContext()->getTime()) ;
            t-=(Real)((int)((int)t/dimt)*dimt);
            t=(t-floor(t)>0.5)?ceil(t):floor(t); // nearest
            if(t<0) t=0.0; else if(t>=(Real)dimt) t=(Real)dimt-1.0; // clamp

            if(this->time!=(unsigned int)t) { this->time=(unsigned int)t; update(); }
        }
    }

    virtual void draw(const core::visual::VisualParams* vparams)
    {
        if (!vparams->displayFlags().getShowVisualModels()) return;

        raPositions pos(this->position);
        raTexCoords tc(this->texCoord);
        raTriangles tri(this->triangles);
        raImage in(this->image);
        raTexture inTex(this->texImage);

        glPushAttrib( GL_LIGHTING_BIT || GL_ENABLE_BIT || GL_LINE_BIT || GL_CURRENT_BIT);

        float color[]= {0.5,0.5,0.5,0.}, specular[]= {0.,0.,0.,0.};
        glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);
        glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,specular);
        glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,0.0);
        glColor4fv(color);

        glEnable( GL_LIGHTING);

        if(inTex->getCImgList()) { glEnable( GL_TEXTURE_2D ); texture->bind();}

        glBegin(GL_TRIANGLES);
        for (unsigned int i=0; i<tri.size(); ++i)
        {
            const Vec<3,Real>& a = pos[ tri[i][0] ];
            const Vec<3,Real>& b = pos[ tri[i][1] ];
            const Vec<3,Real>& c = pos[ tri[i][2] ];
            Vec<3,Real> n = cross((b-a),(c-a));	n.normalize();
            glNormal3d(n[0],n[1],n[2]);

            glTexCoord2d(tc[tri[i][0]][0],tc[tri[i][0]][1]); glVertex3d(a[0],a[1],a[2]);
            glTexCoord2d(tc[tri[i][1]][0],tc[tri[i][1]][1]); glVertex3d(b[0],b[1],b[2]);
            glTexCoord2d(tc[tri[i][2]][0],tc[tri[i][2]][1]); glVertex3d(c[0],c[1],c[2]);
        }
        glEnd();

        if(inTex->getCImgList()) { texture->unbind(); 	glDisable( GL_TEXTURE_2D ); }

        glPopAttrib();
    }
};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_IMAGETOMESHENGINE_H
