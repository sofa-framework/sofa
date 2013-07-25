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
#ifndef SOFA_IMAGE_MERGEBRANCHINGIMAGES_H
#define SOFA_IMAGE_MERGEBRANCHINGIMAGES_H

#include "initImage.h"
#include "BranchingImage.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/rmath.h>
#include <sofa/component/component.h>

#ifdef USING_OMP_PRAGMAS
#include <omp.h>
#endif

namespace sofa
{

namespace component
{

namespace engine
{

using helper::vector;
using defaulttype::Vec;
using defaulttype::Vector3;
using defaulttype::Mat;
using namespace cimg_library;
using namespace helper;

/**
 * This class merges branching images into one
 */


template <class _ImageTypes>
class MergeBranchingImages : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(MergeBranchingImages,_ImageTypes),Inherited);

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;

    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::WriteAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    Data<unsigned int> nbImages;

    helper::vector<Data<ImageTypes>*> inputImages;
    helper::vector<Data<TransformType>*> inputTransforms;

    Data<ImageTypes> image;
    Data<TransformType> transform;

    Data<vector<T> > connectLabels;
    Data<unsigned> connectivity;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const MergeBranchingImages<ImageTypes>* = NULL) { return ImageTypes::Name(); }

    MergeBranchingImages()    :   Inherited()
      , nbImages ( initData ( &nbImages,(unsigned int)0,"nbImages","number of images to merge" ) )
      , image(initData(&image,ImageTypes(),"image","Image"))
      , transform(initData(&transform,TransformType(),"transform","Transform"))
      , connectLabels(initData(&connectLabels,"connectLabels","Pairs of label to be connected accross different input images"))
      , connectivity(initData(&connectivity,(unsigned)27,"connectivity","must be 1, 7 or 27 (27 by default or any incorrect value)"))
    {
        createInputImagesData();
        image.setReadOnly(true);
        transform.setReadOnly(true);
        this->addAlias(&image, "outputImage");
        this->addAlias(&image, "branchingImage");
        this->addAlias(&transform, "outputTransform");
        this->addAlias(&transform, "branchingTransform");
    }

    virtual ~MergeBranchingImages()
    {
        deleteInputDataVector(inputImages);
        deleteInputDataVector(inputTransforms);
    }

    virtual void init()
    {
        addInput(&nbImages);
        createInputImagesData();

        addOutput(&image);
        addOutput(&transform);

        setDirtyValue();
    }

    virtual void reinit()
    {
        createInputImagesData();
        update();
    }


    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
    {
        const char* p = arg->getAttribute(nbImages.getName().c_str());
        if (p)
        {
            std::string nbStr = p;
            sout << "parse: setting nbImages="<<nbStr<<sendl;
            nbImages.read(nbStr);
            createInputImagesData();
        }
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str )
    {
        std::map<std::string,std::string*>::const_iterator it = str.find(nbImages.getName());
        if (it != str.end() && it->second)
        {
            std::string nbStr = *it->second;
            sout << "parseFields: setting nbImages="<<nbStr<<sendl;
            nbImages.read(nbStr);
            createInputImagesData();
        }
        Inherit1::parseFields(str);
    }


protected:

    template<class t>
    void createInputDataVector(unsigned int nb, helper::vector< Data<t>* >& vf, std::string name, std::string help)
    {
        vf.reserve(nb);
        for (unsigned int i=vf.size(); i<nb; i++)
        {
            std::ostringstream oname; oname << name << (1+i); std::string name_i = oname.str();

            Data<t>* d = new Data<t>();
            d->setName(name_i);
            d->setHelpMsg(help.c_str());
            d->setReadOnly(true);

            vf.push_back(d);
            this->addData(d);
            this->addInput(d);
        }
    }
    template<class t>
    void deleteInputDataVector(helper::vector< Data<t>* >& vf)
    {
        for (unsigned int i=0; i<vf.size(); ++i)
        {
            this->delInput(vf[i]);
            delete vf[i];
        }
        vf.clear();
    }

    void createInputImagesData(int nb=-1)
    {
        unsigned int n = (nb < 0) ? nbImages.getValue() : (unsigned int)nb;

        createInputDataVector(n, inputImages, "image", "image");
        createInputDataVector(n, inputTransforms, "transform", "transform");
        if (n != nbImages.getValue())
            nbImages.setValue(n);
    }

    virtual void update()
    {
        cleanDirty();
        createInputImagesData();

        unsigned int nb = nbImages.getValue();
        if(!nb) return;

        // init BB
        Vec<2,Coord> BB = this->getBB(0);
        Coord minScale = this->getScale(0);
        for(unsigned int j=1; j<nb; j++)
        {
            Vec<2,Coord> bb = this->getBB(j);
            for(unsigned int k=0; k<bb[0].size(); k++)
            {
                if(BB[0][k]>bb[0][k]) BB[0][k]=bb[0][k];
                if(BB[1][k]<bb[1][k]) BB[1][k]=bb[1][k];
            }
            for(unsigned int k=0; k<3; k++)
                if( minScale[k] > this->getScale(j)[k] )
                    minScale[k] = this->getScale(j)[k];
        }

        // get input Data
        vector<const ImageTypes*> in;
        vector<const TransformType*> inT;
        for(unsigned int j=0; j<nb; j++)
        {
            raImage inData(this->inputImages[j]);  in.push_back(&inData.ref());
            if(in.back()->isEmpty())  { this->serr<<"Image "<<j<<" not found"<<this->sendl; return; }
            raTransform inTData(this->inputTransforms[j]);   inT.push_back(&inTData.ref());
        }

        // init transform = translated version of inputTransforms[0] with minimum voxel size
        waTransform outT(this->transform);
        outT->operator=(*(inT[0]));
        outT->getTranslation()=BB[0];
        outT->getScale()=minScale;

        // init image
        imCoord dim=in[0]->getDimensions();
        Coord MaxP=outT->toImage(BB[1]); // corner pixel = dim-1
        dim[ImageTypes::DIMENSION_X]=ceil(MaxP[0])+1;
        dim[ImageTypes::DIMENSION_Y]=ceil(MaxP[1])+1;
        dim[ImageTypes::DIMENSION_Z]=ceil(MaxP[2])+1;

        waImage outData(this->image);  ImageTypes& img = outData.wref();
        img.setDimensions(dim);

        // fill image
        typedef typename ImageTypes::ConnectionVoxel ConnectionVoxel;
        typedef typename ImageTypes::VoxelIndex VoxelIndex;
        typedef typename ImageTypes::SuperimposedVoxels SuperimposedVoxels;

        vector<unsigned int> sizes(img.getImageSize()); // buffer to record neighbor offsets due to previously pasted images

        for(unsigned int j=0; j<nb; j++)
        {
            const ImageTypes &inj=*(in[j]);

            bimg_forT(inj,t)
            {
                for(unsigned int i=0;i<sizes.size();i++) sizes[i] = img.imgList[t][i].size();

#ifdef USING_OMP_PRAGMAS
#pragma omp parallel for
#endif
                bimg_foroff1D(inj,i1dj)
                        if(inj.imgList[t][i1dj].size())
                {
                    const SuperimposedVoxels &Vj=inj.imgList[t][i1dj];
                    unsigned int Pj[3];     inj.index1Dto3D(i1dj,Pj[0],Pj[1],Pj[2]);
                    Coord P=outT->toImageInt( inT[j]->fromImage( Coord(Pj[0],Pj[1],Pj[2]) ) );

                    if(img.isInside((int)P[0],(int)P[1],(int)P[2]))
                    {
                        SuperimposedVoxels &V=img.imgList[t][ img.index3Dto1D(P[0],P[1],P[2]) ];
                        for( unsigned v=0 ; v<Vj.size() ; ++v )
                        {
                            V.push_back( Vj[v] , dim[ImageTypes::DIMENSION_S]);

                            for( unsigned n=0 ; n<Vj[v].neighbours.size() ; ++n )
                            {
                                VoxelIndex N=Vj[v].neighbours[n];
                                inj.index1Dto3D(N.index1d,Pj[0],Pj[1],Pj[2]);
                                P=outT->toImageInt( inT[j]->fromImage( Coord(Pj[0],Pj[1],Pj[2]) ) );
                                if(img.isInside((int)P[0],(int)P[1],(int)P[2]))
                                {
                                    N.index1d = img.index3Dto1D(P[0],P[1],P[2]);
                                    N.offset += sizes[N.index1d];
                                    V.last().neighbours[n] = N ;
                                }
                            }
                        }
                    }
                }
            }
        }


        // connect labels based on intensities of first channel
        helper::ReadAccessor<Data< vector<T> > > connectL(this->connectLabels);
        for(unsigned int i=0;i<connectL.size()/2;i++)
        {
            const T a=connectL[2*i], b=connectL[2*i+1];

            if(connectivity.getValue()==1)
            {
                bimg_forT(img,t)
                {
#ifdef USING_OMP_PRAGMAS
#pragma omp parallel for
#endif
                    bimg_foroff1D(img,off1D)
                            for( unsigned va=0 ; va<img.imgList[t][off1D].size() ; ++va )
                            if(img(off1D,va,0,t)==a)
                            for( unsigned vb=0 ; vb<img.imgList[t][off1D].size() ; ++vb )
                            if(img(off1D,vb,0,t)==b)
                    {
                        img.imgList[t][off1D][va].addNeighbour(VoxelIndex(off1D,vb));
                        img.imgList[t][off1D][vb].addNeighbour(VoxelIndex(off1D,va));
                    }

                }
            }
            else if(connectivity.getValue()==7)
            {
                bimg_forT(img,t)
                {
//#ifdef USING_OMP_PRAGMAS
//#pragma omp parallel for
//#endif
                    bimg_foroff1D(img,off1D)
                            for( unsigned va=0 ; va<img.imgList[t][off1D].size() ; ++va )
                            if(img(off1D,va,0,t)==a)
                    {
                        // central voxel
                        for( unsigned vb=0 ; vb<img.imgList[t][off1D].size() ; ++vb )
                            if(img(off1D,vb,0,t)==b)
                            {
                                img.imgList[t][off1D][va].addNeighbour(VoxelIndex(off1D,vb));
                                img.imgList[t][off1D][vb].addNeighbour(VoxelIndex(off1D,va));
                            }
                        // 6 neighbors
                        unsigned x,y,z; img.index1Dto3D(off1D,x,y,z);
                        for( int delta = -1 ; delta <= 1 ; delta+=2 )
                            for( unsigned d = 0 ; d < 3 ; ++d )
                            {
                                int g[3]={0,0,0}; g[d]=delta;
                                if(img.isInside((int)x+g[0],(int)y+g[1],(int)z+g[2]))
                                {
                                    unsigned n = img.index3Dto1D((int)x+g[0],(int)y+g[1],(int)z+g[2]);
                                    for( unsigned vb=0 ; vb<img.imgList[t][n].size() ; ++vb )
                                        if(img(n,vb,0,t)==b)
                                        {
                                            img.imgList[t][off1D][va].addNeighbour(VoxelIndex(n,vb));
                                            img.imgList[t][n][vb].addNeighbour(VoxelIndex(off1D,va));
                                        }
                                }
                            }
                    }
                }
            }
            else
            {
                bimg_forT(img,t)
                {
//#ifdef USING_OMP_PRAGMAS
//#pragma omp parallel for
//#endif
                    bimg_foroff1D(img,off1D)
                            for( unsigned va=0 ; va<img.imgList[t][off1D].size() ; ++va )
                            if(img(off1D,va,0,t)==a)
                    {
                        unsigned x,y,z; img.index1Dto3D(off1D,x,y,z);
                        for( int gx = -1 ; gx <= 1 ; ++gx )
                            for( int gy = -1 ; gy <= 1 ; ++gy )
                                for( int gz = -1 ; gz <= 1 ; ++gz )
                                    if(img.isInside((int)x+gx,(int)y+gy,(int)z+gz))
                                    {
                                        unsigned n = img.index3Dto1D((int)x+gx,(int)y+gy,(int)z+gz);
                                        for( unsigned vb=0 ; vb<img.imgList[t][n].size() ; ++vb )
                                            if(img(n,vb,0,t)==b)
                                            {
                                                img.imgList[t][off1D][va].addNeighbour(VoxelIndex(n,vb));
                                                img.imgList[t][n][vb].addNeighbour(VoxelIndex(off1D,va));
                                            }
                                    }
                    }
                }
            }



        }

        sout << "Created merged image from " << nb << " input images." << sendl;
    }

    Vec<2,Coord> getBB(unsigned int i) // get image corners
    {
        Vec<2,Coord> BB;
        raImage rimage(this->inputImages[i]);
        raTransform rtransform(this->inputTransforms[i]);

        const imCoord dim= rimage->getDimensions();
        Vec<8,Coord> p;
        p[0]=Vector3(0,0,0);
        p[1]=Vector3(dim[0]-1,0,0);
        p[2]=Vector3(0,dim[1]-1,0);
        p[3]=Vector3(dim[0]-1,dim[1]-1,0);
        p[4]=Vector3(0,0,dim[2]-1);
        p[5]=Vector3(dim[0]-1,0,dim[2]-1);
        p[6]=Vector3(0,dim[1]-1,dim[2]-1);
        p[7]=Vector3(dim[0]-1,dim[1]-1,dim[2]-1);

        Coord tp=rtransform->fromImage(p[0]);
        BB[0]=tp;
        BB[1]=tp;
        for(unsigned int j=1; j<8; j++)
        {
            tp=rtransform->fromImage(p[j]);
            for(unsigned int k=0; k<tp.size(); k++)
            {
                if(BB[0][k]>tp[k]) BB[0][k]=tp[k];
                if(BB[1][k]<tp[k]) BB[1][k]=tp[k];
            }
        }
        return BB;
    }

    const Coord getScale(unsigned int i) const
    {
        raTransform rtransform(this->inputTransforms[i]);
        return rtransform->getScale();
    }
};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_MergeBranchingImages_H
