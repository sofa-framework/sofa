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
#ifndef SOFA_IMAGE_MESHTOIMAGEENGINE_H
#define SOFA_IMAGE_MESHTOIMAGEENGINE_H

#include "initImage.h"
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <newmat/newmat.h>
#include <newmat/newmatap.h>

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
 * This class rasterizes a mesh into a boolean image (1: inside mesh, 0: outside)
 */


template <class _ImageTypes>
class MeshToImageEngine : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(MeshToImageEngine,_ImageTypes),Inherited);

    typedef SReal Real;

    Data< Real > voxelSize;
    Data< bool > rotateImage;
    Data< unsigned int > padSize;
    Data< unsigned int > subdiv;

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    Data< ImageTypes > image;

    typedef ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    typedef helper::WriteAccessor<Data< TransformType > > waTransform;
    Data< TransformType > transform;

    typedef vector<Vec<3,Real> > SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    typedef helper::WriteAccessor<Data< SeqPositions > > waPositions;
    Data< SeqPositions > position;

    typedef typename BaseMeshTopology::Triangle Triangle;
    typedef typename BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef helper::ReadAccessor<Data< SeqTriangles > > raTriangles;
    typedef helper::WriteAccessor<Data< SeqTriangles > > waTriangles;
    Data< SeqTriangles > triangles;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const MeshToImageEngine<ImageTypes>* = NULL) { return ImageTypes::Name();    }

    MeshToImageEngine()    :   Inherited()
        , voxelSize(initData(&voxelSize,(Real)(1.0),"voxelSize","voxel Size"))
        , rotateImage(initData(&rotateImage,false,"rotateImage","orient the image bounding box according to the mesh (OBB)"))
        , padSize(initData(&padSize,(unsigned int)(0),"padSize","size of border in number of voxels"))
        , subdiv(initData(&subdiv,(unsigned int)(4),"subdiv","number of subdivisions for face rasterization (if needed, increase to avoid holes)"))
        , image(initData(&image,ImageTypes(),"image",""))
        , transform(initData(&transform,TransformType(),"transform",""))
        , position(initData(&position,SeqPositions(),"position","input positions"))
        , triangles(initData(&triangles,SeqTriangles(),"triangles","input triangles"))
    {
        position.setReadOnly(true);
        triangles.setReadOnly(true);
    }

    virtual ~MeshToImageEngine()
    {
    }

    virtual void init()
    {
        addInput(&position);
        addInput(&triangles);
        addOutput(&image);
        addOutput(&transform);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:

    virtual void update()
    {
        cleanDirty();

        raPositions pos(this->position);        unsigned int nbp = pos.size();
        raTriangles tri(this->triangles);       unsigned int nbtri = tri.size();

        if(!nbp || !nbtri) return;

        waImage iml(this->image);
        waTransform tr(this->transform);

        // update transform
        for(unsigned int j=0; j<3; j++) tr->getScale()[j]=this->voxelSize.getValue();

        Real BB[3][2]= { {pos[0][0],pos[0][0]} , {pos[0][1],pos[0][1]} , {pos[0][2],pos[0][2]} };
        if(!this->rotateImage.getValue()) // use Axis Aligned Bounding Box
        {
            for(unsigned int i=1; i<nbp; i++) for(unsigned int j=0; j<3; j++) { if(BB[j][0]>pos[i][j]) BB[j][0]=pos[i][j]; if(BB[j][1]<pos[i][j]) BB[j][1]=pos[i][j]; }
            for(unsigned int j=0; j<3; j++) tr->getRotation()[j]=(Real)0 ;
            for(unsigned int j=0; j<3; j++) tr->getTranslation()[j]=BB[j][0]-tr->getScale()[j]*this->padSize.getValue();
        }
        else  // use Oriented Bounding Box
        {
            // get mean and covariance
            Coord mean; mean.fill(0);
            for(unsigned int i=0; i<nbp; i++) mean+=pos[i];
            mean/=(Real)nbp;
            Mat<3,3,Real> M; M.fill(0);
            for(unsigned int i=0; i<nbp; i++)  for(unsigned int j=0; j<3; j++)  for(unsigned int k=j; k<3; k++)  M[j][k] += (pos[i][j] - mean[j]) * (pos[i][k] - mean[k]);
            M/=(Real)nbp;
            // get eigen vectors of the covariance matrix
            NEWMAT::SymmetricMatrix e(3); e = 0.0;
            for(unsigned int j=0; j<3; j++) { for(unsigned int k=j; k<3; k++)  e(j+1,k+1) = M[j][k]; for(unsigned int k=0; k<j; k++)  e(k+1,j+1) = e(j+1,k+1); }
            NEWMAT::DiagonalMatrix D(3); D = 0.0;
            NEWMAT::Matrix V(3,3); V = 0.0;
            NEWMAT::Jacobi(e, D, V);
            for(unsigned int j=0; j<3; j++) for(unsigned int k=0; k<3; k++) M[j][k]=V(j+1,k+1);
            if(determinant(M)<0) M*=(Real)-1.0;
            Mat<3,3,Real> MT=M.transposed();

            // get orientation from eigen vectors
            helper::Quater< Real > q; q.fromMatrix(M);
            //  q.toEulerVector() does not work
            if(q[0]*q[0]+q[1]*q[1]==0.5 || q[1]*q[1]+q[2]*q[2]==0.5) {q[3]+=10-3; q.normalize();} // hack to avoid singularities
            tr->getRotation()[0]=atan2(2*(q[3]*q[0]+q[1]*q[2]),1-2*(q[0]*q[0]+q[1]*q[1])) * (Real)180.0 / (Real)M_PI;
            tr->getRotation()[1]=asin(2*(q[3]*q[1]-q[2]*q[0])) * (Real)180.0 / (Real)M_PI;
            tr->getRotation()[2]=atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]*q[1]+q[2]*q[2])) * (Real)180.0 / (Real)M_PI;

//std::cout<<"M="<<M<<std::endl;
//std::cout<<"q="<<q<<std::endl;
//std::cout<<"rot="<<tr->getRotation()<<std::endl;
//helper::Quater< Real > qtest= helper::Quater< Real >::createQuaterFromEuler(tr->getRotation());
//std::cout<<"qtest="<<qtest<<std::endl;
//Mat<3,3,Real> Mtest; qtest.toMatrix(Mtest);
//std::cout<<"Mtest="<<Mtest<<std::endl;

            // get bb
            Coord P=MT*pos[0];
            for(unsigned int i=0; i<3; i++) BB[i][0] = BB[i][1] = P[i];
            for(unsigned int i=1; i<nbp; i++) { P=MT*(pos[i]);  for(unsigned int j=0; j<3; j++) { if(BB[j][0]>P[j]) BB[j][0]=P[j]; if(BB[j][1]<P[j]) BB[j][1]=P[j]; } }
            P=Coord(BB[0][0],BB[1][0],BB[2][0]) - tr->getScale()*this->padSize.getValue();
            tr->getTranslation()=M*(P);
        }

        tr->getOffsetT()=(Real)0.0;
        tr->getScaleT()=(Real)1.0;
        tr->isPerspective()=false;
        tr->update(); // update of internal data

        // update image extents
        unsigned int dim[3];
        for(unsigned int j=0; j<3; j++) dim[j]=1+ceil((BB[j][1]-BB[j][0])/tr->getScale()[j]+(Real)2.0*this->padSize.getValue());
        iml->getCImgList().assign(1,dim[0],dim[1],dim[2],1);
        CImg<T>& im=iml->getCImg();
        T color0=(T)0,color1=(T)1;
        im.fill(color1);

        // draw filled faces
        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: Voxelizing triangles.."<<std::endl;
        for(unsigned int i=0; i<nbtri; i++)
        {
            Coord pts[3];
            for(unsigned int j=0; j<3; j++) pts[j] = (tr->toImage(Coord(pos[tri[i][j]])));
            this->draw_triangle(im,pts[0],pts[1],pts[2],color0,this->subdiv.getValue());
        }

        // flood fill from the exterior point (0,0,0)
        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: Filling object.."<<std::endl;
        CImg<T> im2=im;
        im.draw_fill(0,0,0,&color0);
        cimg_foroff(im2,off) if(!im2[off]) im[off]=color1;

        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: Voxelization done."<<std::endl;
    }


    virtual void draw(const core::visual::VisualParams* vparams)
    {
        if (!vparams->displayFlags().getShowVisualModels()) return;
    }



    void draw_line(CImg<T>& im,const Coord& p0,const Coord& p1,const T& color,const unsigned int subdiv)
    // floating point bresenham
    {
        Coord P0(p0),P1(p1);
        unsigned int dim[3]= {im.width(),im.height(),im.depth()};

        for(unsigned int j=0; j<3; j++)
        {
            if(P0[j]>P1[j]) {Coord tmp(P0); P0=P1; P1=tmp;}
            if (P1[j]<0 || P0[j]>=dim[j]) return;
            if (P0[j]<0) { const double D = 1.0f + P1[j] - P0[j]; for(unsigned int k=0; k<3; k++) if(j!=k) P0[k]-=(int)((float)P0[j]*(1.0f + P1[k] - P0[k])/D);  P0[j] = 0; }
            if (P1[j]>=dim[j]) { const double d = (float)P1[j] - dim[j], D = 1.0f + P1[j] - P0[j]; for(unsigned int k=0; k<3; k++) if(j!=k) P1[k]+=(int)(d*(1.0f + P0[k] - P1[k])/D);  P1[j] = dim[j] - 1; }
        }

        Coord delta = P1 - P0;
        unsigned int dmax = cimg::max(cimg::abs(delta[0]),cimg::abs(delta[1]),cimg::abs(delta[2]));
        dmax*=subdiv; // divide step to avoid possible holes
        Coord dP = delta/(Real)dmax;
        Coord P (P0);
        for (unsigned int t = 0; t<=dmax; ++t)
        {
            im((unsigned int)round(P[0]),(unsigned int)round(P[1]),(unsigned int)round(P[2]))=color;
            P+=dP;
        }
    }

    void draw_triangle(CImg<T>& im,const Coord& p0,const Coord& p1,const Coord& p2,const T& color,const unsigned int subdiv)
    // double bresenham
    {
        Coord P0(p0),P1(p1);
        unsigned int dim[3]= {im.width(),im.height(),im.depth()};

        for(unsigned int j=0; j<3; j++)
        {
            if(P0[j]>P1[j]) {Coord tmp(P0); P0=P1; P1=tmp;}
            if (P1[j]<0 || P0[j]>=dim[j]) return;
            if (P0[j]<0) { const double D = 1.0f + P1[j] - P0[j]; for(unsigned int k=0; k<3; k++) if(j!=k) P0[k]-=(int)((float)P0[j]*(1.0f + P1[k] - P0[k])/D);  P0[j] = 0; }
            if (P1[j]>=dim[j]) { const double d = (float)P1[j] - dim[j], D = 1.0f + P1[j] - P0[j]; for(unsigned int k=0; k<3; k++) if(j!=k) P1[k]+=(int)(d*(1.0f + P0[k] - P1[k])/D);  P1[j] = dim[j] - 1; }
        }

        Coord delta = P1 - P0;
        unsigned int dmax = cimg::max(cimg::abs(delta[0]),cimg::abs(delta[1]),cimg::abs(delta[2]));
        dmax*=subdiv; // divide step to avoid possible holes
        Coord dP = delta/(Real)dmax;
        Coord P (P0);
        for (unsigned int t = 0; t<=dmax; ++t)
        {
            this->draw_line(im,P,p2,color,subdiv);
            P+=dP;
        }
    }

};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_MESHTOIMAGEENGINE_H
