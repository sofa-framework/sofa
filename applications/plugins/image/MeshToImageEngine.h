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
#ifndef SOFA_IMAGE_MeshToImageEngine_H
#define SOFA_IMAGE_MeshToImageEngine_H

#include "initImage.h"
#include "ImageTypes.h"
#include <sofa/helper/rmath.h>
#include <sofa/helper/IndexOpenMP.h>
#include <sofa/core/DataEngine.h>
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/helper/SVector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <newmat/newmat.h>
#include <newmat/newmatap.h>

#ifdef _OPENMP
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
using defaulttype::Mat;
using cimg_library::CImg;
using cimg_library::CImgList;

/**
 * This class rasterizes meshes into a boolean image (1: inside mesh, 0: outside) or a scalar image (val: inside mesh, 0: outside)
 * \todo adjust type of value, closingValue, backgroundValue, roiValue according to ImageTypes
 */
template <class _ImageTypes>
class MeshToImageEngine : public core::DataEngine
{


public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(MeshToImageEngine,_ImageTypes),Inherited);

    typedef SReal Real;

    Data< vector<Real> > voxelSize; // should be a Vec<3,Real>, but it is easier to be backward-compatible that way
    typedef helper::WriteOnlyAccessor<Data< vector<Real> > > waVecReal;
    Data< Vec<3,unsigned> > nbVoxels;
    Data< bool > rotateImage;
    Data< unsigned int > padSize;
    Data< unsigned int > subdiv;

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    typedef helper::WriteOnlyAccessor<Data< ImageTypes > > waImage;
    Data< ImageTypes > image;

    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    typedef helper::WriteOnlyAccessor<Data< TransformType > > waTransform;
    Data< TransformType > transform;

    typedef vector<Vec<3,Real> > SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    typedef helper::WriteOnlyAccessor<Data< SeqPositions > > waPositions;
    helper::vector< Data< SeqPositions > *> vf_positions;

    typedef typename core::topology::BaseMeshTopology::Triangle Triangle;
    typedef typename core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef helper::ReadAccessor<Data< SeqTriangles > > raTriangles;
    typedef helper::WriteOnlyAccessor<Data< SeqTriangles > > waTriangles;
    helper::vector< Data< SeqTriangles >*> vf_triangles;

    typedef typename core::topology::BaseMeshTopology::Edge Edge;
    typedef typename core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef helper::ReadAccessor<Data< SeqEdges > > raEdges;
    typedef helper::WriteOnlyAccessor<Data< SeqEdges > > waEdges;
    helper::vector< Data< SeqEdges >*> vf_edges;

    typedef double ValueType;
    typedef helper::vector<ValueType> SeqValues;
    typedef helper::ReadAccessor<Data< SeqValues > > raValues;
    helper::vector< Data< SeqValues >*> vf_values;

    helper::vector< Data< ValueType >*> vf_InsideValues;

    typedef helper::SVector<typename core::topology::BaseMeshTopology::PointID> SeqIndex; ///< one roi defined as an index list
    typedef helper::vector<SeqIndex> VecSeqIndex;  ///< vector of rois
    helper::vector< Data<VecSeqIndex> *> vf_roiIndices;  ///< vector of rois for each mesh
    helper::vector< Data<SeqValues> *> vf_roiValue;   ///< values for each roi
    typedef helper::ReadAccessor<Data< VecSeqIndex > > raIndex;

    Data< ValueType > backgroundValue;

    Data<unsigned int> f_nbMeshes;

    Data<bool> gridSnap;

    Data<bool> worldGridAligned;


    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const MeshToImageEngine<ImageTypes>* = NULL) { return ImageTypes::Name();    }

    MeshToImageEngine()    :   Inherited()
      , voxelSize(initData(&voxelSize,vector<Real>(3,(Real)1.0),"voxelSize","voxel Size (redondant with and not priority over nbVoxels)"))
      , nbVoxels(initData(&nbVoxels,Vec<3,unsigned>(0,0,0),"nbVoxels","number of voxel (redondant with and priority over voxelSize)"))
      , rotateImage(initData(&rotateImage,false,"rotateImage","orient the image bounding box according to the mesh (OBB)"))
      , padSize(initData(&padSize,(unsigned int)(0),"padSize","size of border in number of voxels"))
      , subdiv(initData(&subdiv,(unsigned int)(4),"subdiv","number of subdivisions for face rasterization (if needed, increase to avoid holes)"))
      , image(initData(&image,ImageTypes(),"image",""))
      , transform(initData(&transform,TransformType(),"transform",""))
      , backgroundValue(initData(&backgroundValue,0.,"backgroundValue","pixel value at background"))
      , f_nbMeshes( initData (&f_nbMeshes, (unsigned)1, "nbMeshes", "number of meshes to voxelize (Note that the last one write on the previous ones)") )
      , gridSnap(initData(&gridSnap,true,"gridSnap","align voxel centers on voxelSize multiples for perfect image merging (nbVoxels and rotateImage should be off)"))
      , worldGridAligned(initData(&worldGridAligned, false, "worldGridAligned", "perform rasterization on a world aligned grid using nbVoxels and voxelSize"))
    {
        createInputMeshesData();
    }

    virtual ~MeshToImageEngine()
    {
        deleteInputDataVector(vf_positions);
        deleteInputDataVector(vf_edges);
        deleteInputDataVector(vf_triangles);
        deleteInputDataVector(vf_values);
        deleteInputDataVector(vf_InsideValues);
        deleteInputDataVector(vf_roiIndices);
        deleteInputDataVector(vf_roiValue);
    }

    virtual void init()
    {
        // backward compatibility (is InsideValue is not set: use first value)
        for( size_t meshId=0; meshId<vf_InsideValues.size() ; ++meshId )
            if(!this->vf_InsideValues[meshId]->isSet())
            {
                this->vf_InsideValues[meshId]->setValue(this->vf_values[meshId]->getValue()[0]);
                serr<<"InsideValue["<<meshId<<"] is not set -> used Value["<<meshId<<"]="<<this->vf_values[meshId]->getValue()[0]<<" instead"<<sendl;
            }


        addInput(&f_nbMeshes);

        createInputMeshesData();

        // HACK to enforce copying linked data so the first read is not done in update(). Because the first read enforces the copy, then tag the data as modified and all update() again.
        for( size_t meshId=0; meshId<f_nbMeshes.getValue() ; ++meshId )
        {
            this->vf_positions[meshId]->getValue();
            this->vf_edges[meshId]->getValue();
            this->vf_triangles[meshId]->getValue();
            this->vf_values[meshId]->getValue();
            this->vf_InsideValues[meshId]->getValue();
            this->vf_roiIndices[meshId]->getValue();
            this->vf_roiValue[meshId]->getValue();
        }

        addOutput(&image);
        addOutput(&transform);
    }

    void clearImage()
    {
        waImage iml(this->image);
        CImg<T>& im= iml->getCImg();
        im.fill((T)0);
    }

    virtual void reinit() { update(); }

protected:

    virtual void update()
    {
        createInputMeshesData();

        // to be backward-compatible, if less than 3 values, fill with the last one
        waVecReal vs( voxelSize ); unsigned vs_lastid=vs.size()-1;
        for( unsigned i=vs.size() ; i<3 ; ++i ) vs.push_back( vs[vs_lastid] );
        vs.resize(3);

        waImage iml(this->image);
        waTransform tr(this->transform);

        // update transform
        Real BB[3][2] = { {std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max()} , {std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max()} , {std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max()} };

        if(worldGridAligned.getValue() == true) // no transformation, simply assign an image of numVoxel*voxelSize
        {
            // min and max centered around origin of transform
            for(int i=0; i< 3; i++)
            {
                BB[i][1] = nbVoxels.getValue()[i]*voxelSize.getValue()[i]*0.5f;
                BB[i][0] = -BB[i][1];
            }
        }
        else if(!this->rotateImage.getValue()) // use Axis Aligned Bounding Box
        {
            for(size_t j=0; j<3; j++) tr->getRotation()[j]=(Real)0 ;

            for( unsigned meshId=0; meshId<f_nbMeshes.getValue() ; ++meshId )
            {
                raPositions pos(*this->vf_positions[meshId]);       unsigned int nbp = pos.size();

                for(size_t i=0; i<nbp; i++) for(size_t j=0; j<3; j++) { if(BB[j][0]>pos[i][j]) BB[j][0]=pos[i][j]; if(BB[j][1]<pos[i][j]) BB[j][1]=pos[i][j]; }
            }

            // enlarge a bit the bb to prevent from numerical precision issues in rasterization
            for(size_t j=0; j<3; j++)
            {
                Real EPSILON = (BB[j][1]-BB[j][0])*1E-10;
                BB[j][1] += EPSILON;
                BB[j][0] -= EPSILON;
            }

            if( nbVoxels.getValue()[0]!=0 && nbVoxels.getValue()[1]!=0 && nbVoxels.getValue()[2]!=0 ) for(size_t j=0; j<3; j++) tr->getScale()[j] = (BB[j][1] - BB[j][0]) / nbVoxels.getValue()[j];
            else for(size_t j=0; j<3; j++) tr->getScale()[j] = this->voxelSize.getValue()[j];

            if(this->gridSnap.getValue())
                if( nbVoxels.getValue()[0]==0 || nbVoxels.getValue()[1]==0 || nbVoxels.getValue()[2]==0 )
                {
                    for(size_t j=0; j<3; j++) BB[j][0] = tr->getScale()[j]*floor(BB[j][0]/tr->getScale()[j]);
                    for(size_t j=0; j<3; j++) BB[j][1] = tr->getScale()[j]*ceil(BB[j][1]/tr->getScale()[j]);
                }

            for(size_t j=0; j<3; j++) tr->getTranslation()[j]=BB[j][0]+tr->getScale()[j]*0.5-tr->getScale()[j]*this->padSize.getValue();
        }
        else  // use Oriented Bounding Box
        {
            unsigned nbpTotal = 0; // total points over all meshes

            // get mean and covariance
            Coord mean; mean.fill(0);
            for( unsigned meshId=0; meshId<f_nbMeshes.getValue() ; ++meshId )
            {
                raPositions pos(*this->vf_positions[meshId]);       unsigned int nbp = pos.size();
                for(size_t i=0; i<nbp; i++) mean+=pos[i];
                nbpTotal += nbp;
            }
            mean/=(Real)nbpTotal;

            Mat<3,3,Real> M; M.fill(0);
            for( unsigned meshId=0; meshId<f_nbMeshes.getValue() ; ++meshId )
            {
                raPositions pos(*this->vf_positions[meshId]);       unsigned int nbp = pos.size();
                for(size_t i=0; i<nbp; i++)  for(size_t j=0; j<3; j++)  for(size_t k=j; k<3; k++)  M[j][k] += (pos[i][j] - mean[j]) * (pos[i][k] - mean[k]);
            }
            M/=(Real)nbpTotal;

            // get eigen vectors of the covariance matrix
            NEWMAT::SymmetricMatrix e(3); e = 0.0;
            for(size_t j=0; j<3; j++) { for(size_t k=j; k<3; k++)  e(j+1,k+1) = M[j][k]; for(size_t k=0; k<j; k++)  e(k+1,j+1) = e(j+1,k+1); }
            NEWMAT::DiagonalMatrix D(3); D = 0.0;
            NEWMAT::Matrix V(3,3); V = 0.0;
            NEWMAT::Jacobi(e, D, V);
            for(size_t j=0; j<3; j++) for(size_t k=0; k<3; k++) M[j][k]=V(j+1,k+1);
            if(determinant(M)<0) M*=(Real)-1.0;
            Mat<3,3,Real> MT=M.transposed();

            // get orientation from eigen vectors
            helper::Quater< Real > q; q.fromMatrix(M);
            //  q.toEulerVector() does not work
            if(q[0]*q[0]+q[1]*q[1]==0.5 || q[1]*q[1]+q[2]*q[2]==0.5) {q[3]+=10-3; q.normalize();} // hack to avoid singularities
            tr->getRotation()[0]=atan2(2*(q[3]*q[0]+q[1]*q[2]),1-2*(q[0]*q[0]+q[1]*q[1])) * (Real)180.0 / (Real)M_PI;
            tr->getRotation()[1]=asin(2*(q[3]*q[1]-q[2]*q[0])) * (Real)180.0 / (Real)M_PI;
            tr->getRotation()[2]=atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]*q[1]+q[2]*q[2])) * (Real)180.0 / (Real)M_PI;

            // get bb
            Coord P;
            for( unsigned meshId=0; meshId<f_nbMeshes.getValue() ; ++meshId )
            {
                raPositions pos(*this->vf_positions[meshId]);       unsigned int nbp = pos.size();
                for(size_t i=0; i<nbp; i++) { P=MT*(pos[i]);  for(size_t j=0; j<3; j++) { if(BB[j][0]>P[j]) BB[j][0]=P[j]; if(BB[j][1]<P[j]) BB[j][1]=P[j]; } }
            }

            // enlarge a bit the bb to prevent from numerical precision issues in rasterization
            for(size_t j=0; j<3; j++)
            {
                Real EPSILON = (BB[j][1]-BB[j][0])*1E-10;
                BB[j][1] += EPSILON;
                BB[j][0] -= EPSILON;
            }

            if( nbVoxels.getValue()[0]!=0 && nbVoxels.getValue()[1]!=0 && nbVoxels.getValue()[2]!=0 ) for(size_t j=0; j<3; j++) tr->getScale()[j] = (BB[j][1] - BB[j][0]) / nbVoxels.getValue()[j];
            else for(size_t j=0; j<3; j++) tr->getScale()[j] = this->voxelSize.getValue()[j];

            P=Coord(BB[0][0],BB[1][0],BB[2][0]) + tr->getScale()*0.5 - tr->getScale()*this->padSize.getValue();
            tr->getTranslation()=M*(P);
        }

        tr->getOffsetT()=(Real)0.0;
        tr->getScaleT()=(Real)1.0;
        tr->isPerspective()=false;
        tr->update(); // update of internal data

        // update image extents
        unsigned int dim[3];
        for(size_t j=0; j<3; j++) dim[j]=ceil((BB[j][1]-BB[j][0])/tr->getScale()[j]+(Real)2.0*this->padSize.getValue());

        if(this->worldGridAligned.getValue()==true)
            for(size_t j=0; j<3; j++)
            {
                dim[j]=ceil((BB[j][1]-BB[j][0])/this->voxelSize.getValue()[j]);
                tr->getScale()[j]= this->voxelSize.getValue()[j];
            }
        
        if(iml->getCImgList().size() == 0) iml->getCImgList().assign(1,dim[0],dim[1],dim[2],1);
        else  iml->getCImgList()(0).assign(dim[0],dim[1],dim[2],1);  // Just realloc the memory of the image to suit new size

        // Keep it as a pointer since the code will be called recursively
        CImg<T>& im = iml->getCImg();
        im.fill( (T)backgroundValue.getValue() );

        for( size_t meshId=0 ; meshId<f_nbMeshes.getValue() ; ++meshId )        rasterizeAndFill ( meshId, im, tr );

        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: "<<this->getName()<<": Voxelization done"<<std::endl;

        cleanDirty();
    }


    // regular rasterization like first implementation, with inside filled by the unique value
    void rasterizeAndFill( const unsigned int &meshId, CImg<T>& im, const waTransform& tr )
    {
        //        vf_positions[meshId]->cleanDirty();
        //        vf_triangles[meshId]->cleanDirty();
        //        vf_edges[meshId]->cleanDirty();
        //        TODO - need this ?
        //        vf_roi[meshId]->cleanDirty();
        //        vf_roiValues[meshId]->cleanDirty();

        raPositions pos(*this->vf_positions[meshId]);       unsigned int nbp = pos.size();
        raTriangles tri(*this->vf_triangles[meshId]);       unsigned int nbtri = tri.size();
        raEdges edg(*this->vf_edges[meshId]);               unsigned int nbedg = edg.size();
        if(!nbp || (!nbtri && !nbedg) ) { serr<<"no topology defined for mesh "<<meshId<<sendl; return; }
        unsigned int nbval = this->vf_values[meshId]->getValue().size();

        raIndex roiIndices(*this->vf_roiIndices[meshId]);
        if(roiIndices.size() && !this->vf_roiValue[meshId]->getValue().size()) serr<<"at least one roiValue for mesh "<<meshId<<" needs to be specified"<<sendl;
        if(this->f_printLog.getValue())  for(size_t r=0;r<roiIndices.size();++r) std::cout<<"MeshToImageEngine: "<<this->getName()<<"mesh "<<meshId<<"\t ROI "<<r<<"\t number of vertices= " << roiIndices[r].size() << "\t value= "<<getROIValue(meshId,r)<<std::endl;

        /// colors definition
        T FillColor = (T)getValue(meshId,0);
        T InsideColor = (T)this->vf_InsideValues[meshId]->getValue();
//        T OutsideColor = (T)this->backgroundValue.getValue();

        /// draw surface
        CImg<bool> mask;
        mask.assign( im.width(), im.height(), im.depth(), 1 );
        mask.fill(false);

        // draw edges
        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: "<<this->getName()<<":  Voxelizing edges (mesh "<<meshId<<")..."<<std::endl;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(sofa::helper::IndexOpenMP<unsigned int>::type i=0; i<nbedg; i++)
        {
            Coord pts[2];
            for(size_t j=0; j<2; j++) pts[j] = (tr->toImage(Coord(pos[edg[i][j]])));
            T currentColor = FillColor;
            for(size_t r=0;r<roiIndices.size();++r)
            {
                bool isRoi = true;
                for(size_t j=0; j<2; j++)  if(std::find(roiIndices[r].begin(), roiIndices[r].end(), edg[i][j])==roiIndices[r].end()) { isRoi=false; break; }
                if (isRoi) currentColor = (T)getROIValue(meshId,r);
            }
            if (nbval>1 && currentColor == FillColor)  draw_line(im,mask,pts[0],pts[1],getValue(meshId,edg[i][0]),getValue(meshId,edg[i][1]),this->subdiv.getValue()); // edge rasterization with interpolated values (if not in roi)
            else draw_line(im,mask,pts[0],pts[1],currentColor,this->subdiv.getValue());
        }

        //  draw filled faces
        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: "<<this->getName()<<":  Voxelizing triangles (mesh "<<meshId<<")..."<<std::endl;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(sofa::helper::IndexOpenMP<unsigned int>::type i=0; i<nbtri; i++)
        {
            Coord pts[3];
            for(size_t j=0; j<3; j++) pts[j] = (tr->toImage(Coord(pos[tri[i][j]])));
            T currentColor = FillColor;
            for(size_t r=0;r<roiIndices.size();++r)
            {
                bool isRoi = true;
                for(size_t j=0; j<3; j++) if(std::find(roiIndices[r].begin(), roiIndices[r].end(), tri[i][j])==roiIndices[r].end()) { isRoi=false; break; }
                if (isRoi) currentColor = (T)getROIValue(meshId,r);
            }
            if (nbval>1 && currentColor == FillColor)  // triangle rasterization with interpolated values (if not in roi)
            {
                draw_triangle(im,mask,pts[0],pts[1],pts[2],getValue(meshId,tri[i][0]),getValue(meshId,tri[i][1]),getValue(meshId,tri[i][2]),this->subdiv.getValue());
                draw_triangle(im,mask,pts[1],pts[2],pts[0],getValue(meshId,tri[i][1]),getValue(meshId,tri[i][2]),getValue(meshId,tri[i][0]),this->subdiv.getValue());  // fill along two directions to be sure that there is no hole
            }
            else
            {
                draw_triangle(im,mask,pts[0],pts[1],pts[2],currentColor,this->subdiv.getValue());
                draw_triangle(im,mask,pts[1],pts[2],pts[0],currentColor,this->subdiv.getValue());  // fill along two directions to be sure that there is no hole
            }
        }


        /// fill inside
        if(!isClosed(tri.ref())) sout<<"mesh["<<meshId<<"] might be open, let's try to fill it anyway"<<sendl;
//        else
        {
            // flood fill from the exterior point (0,0,0) with the color outsideColor
            if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: "<<this->getName()<<":  Filling object (mesh "<<meshId<<")..."<<std::endl;
            bool colorTrue=true;
            mask.draw_fill(0,0,0,&colorTrue);
            cimg_foroff(mask,off) if(!mask[off]) im[off]=InsideColor;
        }
    }


    /// retrieve input value of vertex 'index' of mesh 'meshId'
    ValueType getValue( const unsigned int &meshId, const unsigned int &index ) const
    {
        if(!this->vf_values[meshId]->getValue().size()) return (ValueType)1.0;
        return ( index<this->vf_values[meshId]->getValue().size() )? this->vf_values[meshId]->getValue()[index] : this->vf_values[meshId]->getValue()[0];
    }

    /// retrieve value of roi 'index' of mesh 'meshId'
    ValueType getROIValue( const unsigned int &meshId, const unsigned int &index ) const
    {
        if(!this->vf_roiValue[meshId]->getValue().size()) return (ValueType)1.0;
        return ( index<this->vf_roiValue[meshId]->getValue().size() )? this->vf_roiValue[meshId]->getValue()[index] : this->vf_roiValue[meshId]->getValue()[0];
    }


    /// check if mesh is closed (ie. all edges are present twice in triangle list)
    bool isClosed( const SeqTriangles& tri ) const
    {
        typedef std::pair<unsigned int,unsigned int> edge;
        typedef std::set< edge > edgeset;
        typedef typename edgeset::iterator edgesetit;

        edgeset edges;
        for(size_t i=0; i<tri.size(); i++)
            for(size_t j=0; j<3; j++)
            {
                unsigned int p1=tri[i][(j==0)?2:j-1],p2=tri[i][j];
                edgesetit it=edges.find(edge(p2,p1));
                if(it==edges.end()) edges.insert(edge(p1,p2));
                else edges.erase(it);
            }
        if(edges.empty())  return true;
        else return false;
    }



    virtual void draw(const core::visual::VisualParams* /*vparams*/)
    {
    }


    template<class PixelT>
    bool isInsideImage(CImg<PixelT>& img, unsigned int x, unsigned int y, unsigned z)
    {
        //		if(x<0) return false;
        //		if(y<0) return false;
        //		if(z<0) return false;
        if(x>=(unsigned int)img.width() ) return false;
        if(y>=(unsigned int)img.height()) return false;
        if(z>=(unsigned int)img.depth() ) return false;
        return true;
    }

    template<class PixelT>
    void draw_line(CImg<PixelT>& im,CImg<bool>& mask,const Coord& p0,const Coord& p1,const PixelT& color,const unsigned int subdiv)
    // floating point bresenham
    {
        Coord P0(p0),P1(p1);

        Coord delta = P1 - P0;
        unsigned int dmax = cimg_library::cimg::max(cimg_library::cimg::abs(delta[0]),cimg_library::cimg::abs(delta[1]),cimg_library::cimg::abs(delta[2]));
        dmax*=subdiv; // divide step to avoid possible holes
        Coord dP = delta/(Real)dmax;
        Coord P (P0);
        for (unsigned int t = 0; t<=dmax; ++t)
        {
            unsigned int x=(unsigned int)sofa::helper::round(P[0]), y=(unsigned int)sofa::helper::round(P[1]), z=(unsigned int)sofa::helper::round(P[2]);
            if(isInsideImage<PixelT>(im,x,y,z))
            {
                im(x,y,z)=color;
                mask(x,y,z)=true;
            }
            P+=dP;
        }
    }

    template<class PixelT>
    void draw_line(CImg<PixelT>& im,CImg<bool>& mask,const Coord& p0,const Coord& p1,const Real& color0,const Real& color1,const unsigned int subdiv)
    // floating point bresenham
    {
        Coord P0(p0),P1(p1);

        Coord delta = P1 - P0;
        unsigned int dmax = cimg_library::cimg::max(cimg_library::cimg::abs(delta[0]),cimg_library::cimg::abs(delta[1]),cimg_library::cimg::abs(delta[2]));
        dmax*=subdiv; // divide step to avoid possible holes
        Coord dP = delta/(Real)dmax;
        Coord P (P0);
        for (unsigned int t = 0; t<=dmax; ++t)
        {
            Real u = (dmax == 0) ? Real(0.5) : (Real)t / (Real)dmax;
            PixelT    color = (PixelT)(color0 * (1.0 - u) + color1 * u);
            unsigned int x=(unsigned int)sofa::helper::round(P[0]), y=(unsigned int)sofa::helper::round(P[1]), z=(unsigned int)sofa::helper::round(P[2]);
            if(isInsideImage<PixelT>(im,x,y,z))
            {
                im(x,y,z)=color;
                mask(x,y,z)=true;
            }
            P+=dP;
        }
    }

    template<class PixelT>
    void draw_triangle(CImg<PixelT>& im,CImg<bool>& mask,const Coord& p0,const Coord& p1,const Coord& p2,const PixelT& color,const unsigned int subdiv)
    // double bresenham
    {
        Coord P0(p0),P1(p1);

        Coord delta = P1 - P0;
        unsigned int dmax = cimg_library::cimg::max(cimg_library::cimg::abs(delta[0]),cimg_library::cimg::abs(delta[1]),cimg_library::cimg::abs(delta[2]));
        dmax*=subdiv; // divide step to avoid possible holes
        Coord dP = delta/(Real)dmax;
        Coord P (P0);
        for (unsigned int t = 0; t<=dmax; ++t)
        {
            this->draw_line(im,mask,P,p2,color,subdiv);
            P+=dP;
        }
    }

    template<class PixelT>
    void draw_triangle(CImg<PixelT>& im,CImg<bool>& mask,const Coord& p0,const Coord& p1,const Coord& p2,const Real& color0,const Real& color1,const Real& color2,const unsigned int subdiv)
    // double bresenham
    {
        Coord P0(p0),P1(p1);

        Coord delta = P1 - P0;
        unsigned int dmax = cimg_library::cimg::max(cimg_library::cimg::abs(delta[0]),cimg_library::cimg::abs(delta[1]),cimg_library::cimg::abs(delta[2]));
        dmax*=subdiv; // divide step to avoid possible holes
        Coord dP = delta/(Real)dmax;
        Coord P (P0);
        for (unsigned int t = 0; t<=dmax; ++t)
        {
            Real u = (dmax == 0) ? Real(0.5) : (Real)t / (Real)dmax;
            PixelT    color = (PixelT)(color0 * (1.0 - u) + color1 * u);
            this->draw_line(im,mask,P,p2,color,color2,subdiv);
            P+=dP;
        }
    }


public:
    void createInputMeshesData()
    {
        unsigned int n = f_nbMeshes.getValue();

        createInputDataVector(n, vf_positions, "position", "input positions for mesh ", SeqPositions(), true);
        createInputDataVector(n, vf_edges, "edges", "input edges for mesh ", SeqEdges(), true);
        createInputDataVector(n, vf_triangles, "triangles", "input triangles for mesh ", SeqTriangles(), true);

        ValueType defaultValue = (ValueType)1.0;
        SeqValues defaultValues; defaultValues.push_back(defaultValue);
        createInputDataVector(n, vf_values, "value", "pixel value on mesh surface ", defaultValues, false);

        createInputDataVector(n, vf_InsideValues, "insideValue", "pixel value inside the mesh", defaultValue, false);

        createInputDataVector(n, vf_roiIndices, "roiIndices", "List of Regions Of Interest, vertex indices ", VecSeqIndex(), false);
        createInputDataVector(n, vf_roiValue, "roiValue", "pixel value for ROIs, list of values ", SeqValues(), false);
    }

protected:
    template<class U>
    void createInputDataVector(unsigned int nb, helper::vector< Data<U>* >& vf, std::string name, std::string help, const U&defaultValue, bool readOnly=false)
    {
        vf.reserve(nb);
        for (unsigned int i=vf.size(); i<nb; ++i)
        {
            std::ostringstream oname, ohelp;
            oname << name;
            ohelp << help;

            if( i>0 ) // to keep backward-compatible with the previous definition voxelizing only one input mesh
            {
                oname << (i+1);
                ohelp << (i+1);
            }

            std::string name_i = oname.str();
            std::string help_i = ohelp.str();
            Data<U>* d = new Data<U>(help_i.c_str(), true, false);
            d->setName(name_i);
            d->setReadOnly(readOnly);
            d->setValue(defaultValue);
            d->unset();
            vf.push_back(d);
            this->addData(d);
            this->addInput(d);
        }
    }

    template<class U>
    void deleteInputDataVector(helper::vector< Data<U>* >& vf)
    {
        for (unsigned int i=0; i<vf.size(); ++i)
        {
            this->delInput(vf[i]);
            delete vf[i];
        }
        vf.clear();
    }

public:

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
    {
        const char* p = arg->getAttribute(f_nbMeshes.getName().c_str());
        if (p)
        {
            std::string nbStr = p;
            sout << "parse: setting nbMeshes="<<nbStr<<sendl;
            f_nbMeshes.read(nbStr);
            createInputMeshesData();
        }
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str )
    {
        std::map<std::string,std::string*>::const_iterator it = str.find(f_nbMeshes.getName());
        if (it != str.end() && it->second)
        {
            std::string nbStr = *it->second;
            sout << "parseFields: setting nbMeshes="<<nbStr<<sendl;
            f_nbMeshes.read(nbStr);
            createInputMeshesData();
        }
        Inherit1::parseFields(str);
    }

};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_IMAGE_MeshToImageEngine_CPP)
extern template class SOFA_IMAGE_API MeshToImageEngine<sofa::defaulttype::ImageB>;
extern template class SOFA_IMAGE_API MeshToImageEngine<sofa::defaulttype::ImageUC>;
extern template class SOFA_IMAGE_API MeshToImageEngine<sofa::defaulttype::ImageUS>;
extern template class SOFA_IMAGE_API MeshToImageEngine<sofa::defaulttype::ImageD>;
#endif




} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_MeshToImageEngine_H
