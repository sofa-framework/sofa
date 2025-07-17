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
#ifndef SOFA_IMAGE_DEPTHMAPTOMESHENGINE_H
#define SOFA_IMAGE_DEPTHMAPTOMESHENGINE_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/type/Vec.h>

#if IMAGE_HAVE_SOFA_GL == 1
#include <sofa/gl/Texture.h>
#endif // IMAGE_HAVE_SOFA_GL == 1

namespace sofa
{

namespace component
{

namespace engine
{


/**
 * This class computes a mesh from a depth map image
 */


template <class _ImageTypes>
class DepthMapToMeshEngine : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(DepthMapToMeshEngine,_ImageTypes),Inherited);

    typedef SReal Real;

    Data< Real > depthFactor; ///< Intensity to depth factor
    Data< Real > minThreshold; ///< minimal depth for point creation
    Data< Real > diffThreshold; ///< maximal depth variation for triangle creation

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;

    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    typedef _ImageTypes TextureTypes;
    typedef helper::ReadAccessor<Data< TextureTypes > > raTexture;
    Data< TextureTypes > texImage;

    typedef type::vector<type::Vec<3,Real> > SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    typedef helper::WriteOnlyAccessor<Data< SeqPositions > > waPositions;
    Data< SeqPositions > position; ///< output positions

    typedef type::fixed_array<Real,2> TexCoord;
    typedef type::vector<TexCoord> SeqTexCoords;
    typedef helper::ReadAccessor<Data< SeqTexCoords > > raTexCoords;
    typedef helper::WriteOnlyAccessor<Data< SeqTexCoords > > waTexCoords;
    Data< SeqTexCoords > texCoord; ///< output texture coordinates
    Data< TexCoord > texOffset; ///< texture offsets (in [0,1])

    typedef typename core::topology::BaseMeshTopology::Triangle Triangle;
    typedef typename core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef helper::ReadAccessor<Data< SeqTriangles > > raTriangles;
    typedef helper::WriteOnlyAccessor<Data< SeqTriangles > > waTriangles;
    Data< SeqTriangles > triangles; ///< output triangles

    DepthMapToMeshEngine()    :   Inherited()
        , depthFactor(initData(&depthFactor,(Real)(1.0),"depthFactor","Intensity to depth factor"))
        , minThreshold(initData(&minThreshold,(Real)(0.1),"minThreshold","minimal depth for point creation"))
        , diffThreshold(initData(&diffThreshold,(Real)(1.0),"diffThreshold","maximal depth variation for triangle creation"))
        , image(initData(&image,ImageTypes(),"image",""))
        , transform(initData(&transform,TransformType(),"transform",""))
        , texImage(initData(&texImage,TextureTypes(),"texImage",""))
        , position(initData(&position,SeqPositions(),"position","output positions"))
        , texCoord(initData(&texCoord,SeqTexCoords(),"texCoord","output texture coordinates"))
        , texOffset(initData(&texOffset,TexCoord(0.0,0.0),"texOffset","texture offsets (in [0,1])"))
        , triangles(initData(&triangles,SeqTriangles(),"triangles","output triangles"))
        , time((unsigned int)0)
#if IMAGE_HAVE_SOFA_GL == 1
        , texture(NULL)
#endif // IMAGE_HAVE_SOFA_GL == 1
    {
        image.setReadOnly(true);
        transform.setReadOnly(true);
        f_listening.setValue(true);
    }

    ~DepthMapToMeshEngine() override
    {
#if IMAGE_HAVE_SOFA_GL == 1
        if(texture) delete texture;
#endif // IMAGE_HAVE_SOFA_GL == 1
    }

    void init() override
    {
        addInput(&image);
        addInput(&transform);
        addOutput(&position);
        addOutput(&texCoord);
        addOutput(&triangles);
        setDirtyValue();
    }

    void reinit() override { update(); }

protected:

    unsigned int time;
#if IMAGE_HAVE_SOFA_GL == 1
    sofa::gl::Texture* texture;
    static const unsigned texture_res=256;
#endif // IMAGE_HAVE_SOFA_GL == 1

    void doUpdate() override
    {
        raImage in(this->image);
        raTransform inT(this->transform);
        raTexture inTex(this->texImage);
        const unsigned int dimx=in->getDimensions()[0],dimy=in->getDimensions()[1];

        waPositions pos(this->position);
        waTexCoords tc(this->texCoord);
        waTriangles tri(this->triangles);

        // get image at time t
        const cimg_library::CImg<T>& img = in->getCImg(this->time);
        Real f = this->depthFactor.getValue();

#if IMAGE_HAVE_SOFA_GL == 1
        // update texture
        if(texture && !inTex->isEmpty())
        {
            const cimg_library::CImg<T>& tex = inTex->getCImg(this->time);
            cimg_library::CImg<unsigned char> plane=convertToUC( tex.get_resize(texture_res,texture_res,1,-100,1) );
            cimg_forXY(plane,x,y)
            {
                unsigned char *b=texture->getImage()->getPixels()+4*(y*texture_res+x);
                for(unsigned int c=0; c<3 && c<(unsigned int)plane.spectrum() ; c++) b[c]=plane(x,y,0,c);
                for(unsigned int c=plane.spectrum(); c<3; c++) b[c]=b[0];
                b[3]=(unsigned char)(-1);
            }
            texture->update();
        }
#endif // IMAGE_HAVE_SOFA_GL == 1

        // update points
        unsigned int count=0,p1,p2,p3;
        pos.resize(dimx*dimy);
        tc.resize(dimx*dimy);
        type::vector<bool> isValid(dimx*dimy);
        Real cameraZ= 0.5; // camera position relative to image plane = offset for depth
        Real minT= minThreshold.getValue();
        for(unsigned int y=0; y<dimy; y++)
            for(unsigned int x=0; x<dimx; x++)
            {
                tc[count][0]=(Real)x/(Real)(dimx-1) + texOffset.getValue()[0];
                tc[count][1]=(Real)y/(Real)(dimy-1) + texOffset.getValue()[1];
                Real val=(Real)img(x,y)*f;
                if(val<minT) { isValid[count]=false; val=0;}
                else isValid[count]=true;
                pos[count++]=inT->fromImage(Coord((Real)x,(Real)y,(Real)val-cameraZ));

            }

        // update triangles
        tri.clear();
        Real diffT= diffThreshold.getValue(),diff1,diff2,diff3;
        for(unsigned int y=0; y<dimy-1; y++)
            for(unsigned int x=0; x<dimx-1; x++)
            {
                diff1 = fabs((Real)img(x,y) - (Real)img(x+1,y)); diff2 = fabs((Real)img(x+1,y) - (Real)img(x,y+1)); diff3 = fabs((Real)img(x,y+1) - (Real)img(x,y));
                p1=x+y*dimx; p2=x+1+y*dimx; p3=x+(y+1)*dimx;
                if(isValid[p1] && isValid[p2] && isValid[p3] && diff1<diffT && diff2<diffT && diff3<diffT) tri.push_back(Triangle(p1,p2,p3));

                diff1 = fabs((Real)img(x+1,y) - (Real)img(x+1,y+1)); diff2 = fabs((Real)img(x+1,y+1) - (Real)img(x,y+1)); diff3 = fabs((Real)img(x,y+1) - (Real)img(x+1,y));
                p1=x+1+y*dimx; p2=x+1+(y+1)*dimx; p3=x+(y+1)*dimx;
                if(isValid[p1] && isValid[p2] && isValid[p3] && diff1<diffT && diff2<diffT && diff3<diffT) tri.push_back(Triangle(p1,p2,p3));
            }
    }

    void handleEvent(sofa::core::objectmodel::Event *event) override
    {
        if (simulation::AnimateEndEvent::checkEventType(event))
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

            if(this->time!=(unsigned int)t) { this->time=(unsigned int)t; update(); }
        }
    }

    void draw(const core::visual::VisualParams* vparams) override
    {
#if IMAGE_HAVE_SOFA_GL == 1
        // need a valid opengl context to initialize an opengl texture, a context is not always bound during the init phase so we init the texture here
        if(!texture)
        {
            texture = new sofa::gl::Texture(new helper::io::Image,false);
            texture->getImage()->init(texture_res,texture_res,32);
            texture->init();

            // we need the texture to be updated
            setDirtyValue();
        }

        if (!vparams->displayFlags().getShowVisualModels()) return;

        vparams->drawTool()->saveLastState();

        raPositions pos(this->position);
        raTexCoords tc(this->texCoord);
        raTriangles tri(this->triangles);
        raTexture inTex(this->texImage);

        vparams->drawTool()->setMaterial(sofa::type::RGBAColor{0.5,0.5,0.5,0.});

        vparams->drawTool()->enableLighting();

        if(!inTex->isEmpty()) {
            glEnable( GL_TEXTURE_2D );
            texture->bind();
        }

        glBegin(GL_TRIANGLES);
        for (std::size_t i=0; i<tri.size(); ++i)
        {
            const type::Vec<3,Real>& a = pos[ tri[i][0] ];
            const type::Vec<3,Real>& b = pos[ tri[i][1] ];
            const type::Vec<3,Real>& c = pos[ tri[i][2] ];
            type::Vec<3,Real> n = cross((c-a),(b-a));	n.normalize();
            glNormal3d(n[0],n[1],n[2]);

            glTexCoord2d(tc[tri[i][0]][0],tc[tri[i][0]][1]); glVertex3d(a[0],a[1],a[2]);
            glTexCoord2d(tc[tri[i][1]][0],tc[tri[i][1]][1]); glVertex3d(b[0],b[1],b[2]);
            glTexCoord2d(tc[tri[i][2]][0],tc[tri[i][2]][1]); glVertex3d(c[0],c[1],c[2]);
        }

        if(!inTex->isEmpty()) {
            texture->unbind();
            glDisable( GL_TEXTURE_2D );
        }

        vparams->drawTool()->restoreLastState();
#endif // IMAGE_HAVE_SOFA_GL == 1
    }
};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_DEPTHMAPTOMESHENGINE_H
