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
#include <sofa/core/DataEngine.h>
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <newmat/newmat.h>
#include <newmat/newmatap.h>

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
using defaulttype::Mat;
using cimg_library::CImg;
using cimg_library::CImgList;

/**
 * This class rasterizes meshes into a boolean image (1: inside mesh, 0: outside) or a scalar image (val: inside mesh, 0: outside)
 */


template <class _ImageTypes>
class MeshToImageEngine : public core::DataEngine
{


public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(MeshToImageEngine,_ImageTypes),Inherited);

    typedef SReal Real;

    Data< vector<Real> > voxelSize; // should be a Vec<3,Real>, but it is easier to be backward-compatible that way
    typedef helper::WriteAccessor<Data< vector<Real> > > waVecReal;
    Data< Vec<3,unsigned> > nbVoxels;
    Data< bool > rotateImage;
    Data< unsigned int > padSize;
    Data< unsigned int > subdiv;

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    Data< ImageTypes > image;

    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    typedef helper::WriteAccessor<Data< TransformType > > waTransform;
    Data< TransformType > transform;

    typedef vector<Vec<3,Real> > SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    typedef helper::WriteAccessor<Data< SeqPositions > > waPositions;
    helper::vector< Data< SeqPositions > *> vf_positions;
    Data< SeqPositions > closingPosition;

    typedef typename core::topology::BaseMeshTopology::Triangle Triangle;
    typedef typename core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef helper::ReadAccessor<Data< SeqTriangles > > raTriangles;
    typedef helper::WriteAccessor<Data< SeqTriangles > > waTriangles;
    helper::vector< Data< SeqTriangles >*> vf_triangles;
    Data< SeqTriangles > closingTriangles; // closing could be done per mesh

    typedef typename core::topology::BaseMeshTopology::Edge Edge;
    typedef typename core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef helper::ReadAccessor<Data< SeqEdges > > raEdges;
    typedef helper::WriteAccessor<Data< SeqEdges > > waEdges;
    helper::vector< Data< SeqEdges >*> vf_edges;

    typedef helper::vector<double> SeqValues;
    typedef helper::ReadAccessor<Data< SeqValues > > raValues;
    helper::vector< Data< SeqValues >*> vf_values;

    helper::vector< Data< bool >*> vf_fillInside;

    typedef helper::vector<size_t> SeqIndex;
    typedef helper::ReadAccessor<Data< SeqIndex > > raIndex;
    helper::vector< Data< SeqIndex >*> vf_roiVertices;
    helper::vector< Data< double >*> vf_roiValue;


    Data< double > closingValue;
    Data< double > backgroundValue;

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
        , closingPosition(initData(&closingPosition,SeqPositions(),"closingPosition","ouput closing positions"))
        , closingTriangles(initData(&closingTriangles,SeqTriangles(),"closingTriangles","ouput closing triangles"))
        , closingValue(initData(&closingValue,1.,"closingValue","pixel value at closings"))
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
        deleteInputDataVector(vf_fillInside);
        deleteInputDataVector(vf_roiVertices);
        deleteInputDataVector(vf_roiValue);
    }

    virtual void init()
    {
        addInput(&f_nbMeshes);
        createInputMeshesData();

        // HACK to enforce copying linked data so the first read is not done in update(). Because the first read enforces the copy, then tag the data as modified and all update() again.
        for( size_t meshId=0; meshId<f_nbMeshes.getValue() ; ++meshId )
        {
            this->vf_positions[meshId]->getValue();
            this->vf_edges[meshId]->getValue();
            this->vf_triangles[meshId]->getValue();
            this->vf_roiVertices[meshId]->getValue();
            this->vf_roiValue[meshId]->getValue();
        }

        addOutput(&closingPosition);
        addOutput(&closingTriangles);
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
        cleanDirty();

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

            //std::cout<<"M="<<M<<std::endl;
            //std::cout<<"q="<<q<<std::endl;
            //std::cout<<"rot="<<tr->getRotation()<<std::endl;
            //helper::Quater< Real > qtest= helper::Quater< Real >::createQuaterFromEuler(tr->getRotation());
            //std::cout<<"qtest="<<qtest<<std::endl;
            //Mat<3,3,Real> Mtest; qtest.toMatrix(Mtest);
            //std::cout<<"Mtest="<<Mtest<<std::endl;

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
		
		if(this->worldGridAligned.getValue()==true) {
			for(size_t j=0; j<3; j++) {	
				dim[j]=ceil((BB[j][1]-BB[j][0])/this->voxelSize.getValue()[j]);
				tr->getScale()[j]= this->voxelSize.getValue()[j];
			}
		}
        
		if(iml->getCImgList().size() == 0)
			iml->getCImgList().assign(1,dim[0],dim[1],dim[2],1);
		else
			// Just realloc the memory of the image to suit new size
			iml->getCImgList()(0).assign(dim[0],dim[1],dim[2],1);

		// Keep it as a pointer since the code will be called recursively
        CImg<T>& im = iml->getCImg();
        im.fill( (T)backgroundValue.getValue() );

        for( size_t meshId=0 ; meshId<f_nbMeshes.getValue() ; ++meshId )
        {
            if( /*!vf_fillInside[meshId]->getValue() ||*/ vf_values[meshId]->getValue().size() > 1 )
                rasterizeSeveralValues( meshId, im, tr );
            else
                rasterizeUniqueValueAndFill( meshId, im, tr );
        }

        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: "<<this->getName()<<": Voxelization done"<<std::endl;

        cleanDirty();
    }


    // regular rasterization like first implementation, with inside filled by the unique value
    void rasterizeUniqueValueAndFill( unsigned meshId, CImg<T>& im, const waTransform& tr )
    {
		vf_positions[meshId]->cleanDirty();
		vf_triangles[meshId]->cleanDirty();
		vf_edges[meshId]->cleanDirty();
//        TODO - need this ?
//        vf_roi[meshId]->cleanDirty();
//        vf_roiValues[meshId]->cleanDirty();
		closingTriangles.cleanDirty();

        raPositions pos(*this->vf_positions[meshId]);       unsigned int nbp = pos.size();
        raTriangles tri(*this->vf_triangles[meshId]);       unsigned int nbtri = tri.size();
        raEdges edg(*this->vf_edges[meshId]);               unsigned int nbedg = edg.size();
        raIndex roiVertices(*this->vf_roiVertices[meshId]); bool hasRoi = roiVertices.size()>0;


        if(!nbp || (!nbtri && !nbedg) ) return;

        // colors definition for painting voxels
        unsigned char defaultColor = 0;
        unsigned char fillColor = 1;
        unsigned char closingColor = 2;
        unsigned char roiColor = 3;
        unsigned char outsideColor = 4;

        CImg<unsigned char> imCurrent;
        imCurrent.assign( im.width(), im.height(), im.depth(), 1 );
        imCurrent.fill(defaultColor);


        //        bool isTransformSet=false;
        //        if(this->transform.isSet()) { isTransformSet=true; if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: Voxelize using existing transform.."<<std::endl;}

        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: "<<this->getName()<<":  ROI number of vertices: " << roiVertices.size() << std::endl;

        // draw edges
        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: "<<this->getName()<<":  Voxelizing edges (mesh "<<meshId<<")..."<<std::endl;



#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif
        for(unsigned int i=0; i<nbedg; i++)
        {
            Coord pts[2];
            for(size_t j=0; j<2; j++) pts[j] = (tr->toImage(Coord(pos[edg[i][j]])));
            unsigned char currentColor = fillColor;
            if (hasRoi) {
                bool isRoi = true;
                for(size_t j=0; j<2; j++) { // edge is in roi if both ends are in roi
                    if(std::find(roiVertices.begin(), roiVertices.end(), edg[i][j])==roiVertices.end()) {
                        isRoi=false;
                        break;
                    }
                }
                if (isRoi)
                    currentColor = roiColor;
            }
            draw_line(imCurrent,pts[0],pts[1],currentColor,this->subdiv.getValue());
        }

//            // draw filled faces
        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: "<<this->getName()<<":  Voxelizing triangles (mesh "<<meshId<<")..."<<std::endl;

#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif
        for(unsigned int i=0; i<nbtri; i++)
        {
            Coord pts[3];
            for(size_t j=0; j<3; j++) pts[j] = (tr->toImage(Coord(pos[tri[i][j]])));
            unsigned char currentColor = fillColor;
            if (hasRoi) {
                bool isRoi = true;
                for(size_t j=0; j<3; j++) { // triangle is in roi if all 3 vertices are in roi
                    if(std::find(roiVertices.begin(), roiVertices.end(), tri[i][j])==roiVertices.end()) {
                        isRoi=false;
                        break;
                    }
                }
                if (isRoi) {
                    currentColor = roiColor;
                }
            }
            draw_triangle(imCurrent,pts[0],pts[1],pts[2],currentColor,this->subdiv.getValue());
            draw_triangle(imCurrent,pts[1],pts[2],pts[0],currentColor,this->subdiv.getValue());  // fill along two directions to be sure that there is no hole
        }

        // draw closing faces with closingColor

        raTriangles cltri(this->closingTriangles);
        unsigned previousClosingTriSize = cltri.size();


        this->closeMesh( meshId );

        raPositions clpos(this->closingPosition);

#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif
        for(unsigned int i=previousClosingTriSize; i<cltri.size(); i++)
        {
            Coord pts[3];
            for(size_t j=0; j<3; j++) pts[j] = (tr->toImage(Coord(clpos[cltri[i][j]])));
            draw_triangle(imCurrent,pts[0],pts[1],pts[2],closingColor,this->subdiv.getValue());
            draw_triangle(imCurrent,pts[1],pts[2],pts[0],closingColor,this->subdiv.getValue());  // fill along two directions to be sure that there is no hole
        }


        T trueFillColor = (T)this->vf_values[meshId]->getValue()[0];
        T trueClosingColor;
        if( this->closingValue.getValue()==0 ) trueClosingColor=trueFillColor;
        else trueClosingColor = (T)this->closingValue.getValue();
        T trueRoiColor;
        if( this->vf_roiValue[meshId]->getValue()==0 ) trueRoiColor=trueClosingColor;
        else trueRoiColor = (T)this->vf_roiValue[meshId]->getValue();


        if( vf_fillInside[meshId]->getValue() )
        {
            // flood fill from the exterior point (0,0,0) with the color outsideColor so every voxel==outsideColor are outside
            if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: "<<this->getName()<<":  Filling object (mesh "<<meshId<<")..."<<std::endl;

            imCurrent.draw_fill(0,0,0,&outsideColor);
            cimg_foroff(imCurrent,off)
            {
                if( imCurrent[off]!=outsideColor ) // not outside
                {
                    if( imCurrent[off]==defaultColor || imCurrent[off]==fillColor ) im[off]=trueFillColor; // inside or rasterized
                    else if( imCurrent[off]==closingColor ) im[off]=trueClosingColor; // closing
                    else if( imCurrent[off]==roiColor ) {
                        im[off]=trueRoiColor; // Roi
                    }
                }
            }
        }
        else
        {
            cimg_foroff(imCurrent,off)
            {
                if( imCurrent[off]!=defaultColor) // not outside and not inside
                {
                    if( imCurrent[off]==fillColor ) im[off]=trueFillColor; // rasterized
                    else if( imCurrent[off]==closingColor ) im[off]=trueClosingColor; // closing
                    else if( imCurrent[off]==roiColor ) im[off]=trueRoiColor; // roi
                }
            }
        }
    }




    // pure triangle/edge rasterization with interpolated values without inside filling
    void rasterizeSeveralValues( unsigned meshId, CImg<T>& im, const waTransform& tr )
    {
        raPositions pos(*this->vf_positions[meshId]);       unsigned int nbp = pos.size();
        raTriangles tri(*this->vf_triangles[meshId]);       unsigned int nbtri = tri.size();
        raEdges edg(*this->vf_edges[meshId]);               unsigned int nbedg = edg.size();
        raValues val(*this->vf_values[meshId]);				unsigned int nbval = val.size();

        if(!nbp || (!nbtri && !nbedg) ) return;


        //        bool isTransformSet=false;
        //        if(this->transform.isSet()) { isTransformSet=true; if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: Voxelize using existing transform.."<<std::endl;}



        // draw all rasterized object with color 1

        // draw edges
        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: "<<this->getName()<<":  Voxelizing "<<nbedg<<" edges (mesh "<<meshId<<")..."<<std::endl;

#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif
        for(unsigned int i=0; i<nbedg; i++)
        {
            Coord pts[2];
            T colors[2];
            for(size_t j=0; j<2; j++)
            {
                pts[j] = (tr->toImage(Coord(pos[edg[i][j]])));

                if( edg[i][j]<nbval ) colors[j] = (T)val[edg[i][j]];
                else colors[j] = val[nbval-1];
            }
            draw_line(im,pts[0],pts[1],colors[0],colors[1],this->subdiv.getValue());
        }

//            // draw filled faces
        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: "<<this->getName()<<":  Voxelizing "<<nbtri<<" triangles (mesh "<<meshId<<")..."<<std::endl;

#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for
#endif
        for(unsigned int i=0; i<nbtri; i++)
        {
            Coord pts[3];
            T colors[3];
            for(size_t j=0; j<3; j++)
            {
                pts[j] = (tr->toImage(Coord(pos[tri[i][j]])));
                if( tri[i][j]<nbval ) colors[j] = (T)val[tri[i][j]];
                else colors[j] = val[nbval-1];
            }
            draw_triangle(im,pts[0],pts[1],pts[2],colors[0],colors[1],colors[2],this->subdiv.getValue());
            draw_triangle(im,pts[1],pts[2],pts[0],colors[1],colors[2],colors[0],this->subdiv.getValue());  // fill along two directions to be sure that there is no hole
        }




        // draw closing faces with color 2

		// This will avoid to cause recursion double deletion
		closingTriangles.cleanDirty();

        raTriangles cltri(this->closingTriangles);
        unsigned previousClosingTriSize = cltri.size();


        T colorClosing = (T)this->closingValue.getValue();

        if( colorClosing )
        {
            this->closeMesh( meshId );

            raPositions clpos(this->closingPosition);

    #ifdef USING_OMP_PRAGMAS
            #pragma omp parallel for
    #endif
            for(unsigned int i=previousClosingTriSize; i<cltri.size(); i++)
            {
                Coord pts[3];
                for(size_t j=0; j<3; j++) pts[j] = (tr->toImage(Coord(clpos[cltri[i][j]])));

                this->draw_triangle(im,pts[0],pts[1],pts[2],colorClosing,this->subdiv.getValue());
                this->draw_triangle(im,pts[1],pts[2],pts[0],colorClosing,this->subdiv.getValue());  // fill along two directions to be sure that there is no hole
            }
        }
        else
        {
            // colorClosing=0 interpolate the color

            SeqValues clValues;
            this->closeMesh( meshId, &clValues );

            raPositions clpos(this->closingPosition);


    #ifdef USING_OMP_PRAGMAS
            #pragma omp parallel for
    #endif
            for(unsigned int i=previousClosingTriSize; i<cltri.size(); i++)
            {
                Coord pts[3];
                T colors[3];

                for(size_t j=0; j<3; j++)
                {
                    pts[j] = (tr->toImage(Coord(clpos[cltri[i][j]])));
                    if( !colorClosing ) colors[j] = (T)clValues[cltri[i][j]];
                }

                this->draw_triangle(im,pts[0],pts[1],pts[2],colors[0],colors[1],colors[2],this->subdiv.getValue());
                this->draw_triangle(im,pts[1],pts[2],pts[0],colors[0],colors[1],colors[2],this->subdiv.getValue());  // fill along two directions to be sure that there is no hole
            }

        }


    }



    virtual void draw(const core::visual::VisualParams* vparams)
    {
        if (!vparams->displayFlags().getShowVisualModels()) return;
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
    void draw_line(CImg<PixelT>& im,const Coord& p0,const Coord& p1,const PixelT& color,const unsigned int subdiv)
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
			if(isInsideImage<PixelT>(im, (unsigned int)sofa::helper::round(P[0]),(unsigned int)sofa::helper::round(P[1]),(unsigned int)sofa::helper::round(P[2])))
				im((unsigned int)sofa::helper::round(P[0]),(unsigned int)sofa::helper::round(P[1]),(unsigned int)sofa::helper::round(P[2]))=color;
            
			P+=dP;
        }
    }

    template<class PixelT>
    void draw_line(CImg<PixelT>& im,const Coord& p0,const Coord& p1,const Real& color0,const Real& color1,const unsigned int subdiv)
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
            PixelT color;
            if( dmax==0 )
            {
                color = (PixelT)(0.5*(color0+color1));
            }
            else
            {
				Real u = (dmax == 0) ? Real(0.0) : (Real)t / (Real)dmax;
                color = (PixelT)(color0 * (1.0 - u) + color1 * u);
            }
			
			if(isInsideImage<PixelT>(im, (unsigned int)sofa::helper::round(P[0]),(unsigned int)sofa::helper::round(P[1]),(unsigned int)sofa::helper::round(P[2])))
				im((unsigned int)sofa::helper::round(P[0]),(unsigned int)sofa::helper::round(P[1]),(unsigned int)sofa::helper::round(P[2]))=color;
            P+=dP;
        }
    }

    template<class PixelT>
    void draw_triangle(CImg<PixelT>& im,const Coord& p0,const Coord& p1,const Coord& p2,const PixelT& color,const unsigned int subdiv)
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
            this->draw_line(im,P,p2,color,subdiv);
            P+=dP;
        }
    }

    template<class PixelT>
    void draw_triangle(CImg<PixelT>& im,const Coord& p0,const Coord& p1,const Coord& p2,const Real& color0,const Real& color1,const Real& color2,const unsigned int subdiv)
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

            PixelT color;
            if( dmax==0 )
            {
                color = (PixelT)(0.5*(color0+color1));
            }
            else
            {
                Real u = (Real)t / (Real)dmax;
                color = (PixelT)(color0 * (1.0 - u) + color1 * u);
            }

            this->draw_line(im,P,p2,color,color2,subdiv);
            P+=dP;
        }
    }


    void closeMesh( unsigned meshId, SeqValues* clValues=NULL )
    {
        raPositions pos(*this->vf_positions[meshId]);
        raTriangles tri(*this->vf_triangles[meshId]);
        raValues    val(*this->vf_values[meshId]   );

        waPositions clpos(this->closingPosition);
        waTriangles cltri(this->closingTriangles);

        typedef std::pair<unsigned int,unsigned int> edge;
        typedef std::set< edge > edgeset;
        typedef typename edgeset::iterator edgesetit;

        // get list of border edges
        edgeset edges;
        for(size_t i=0; i<tri.size(); i++)
            for(size_t j=0; j<3; j++)
            {
                unsigned int p1=tri[i][(j==0)?2:j-1],p2=tri[i][j];
                edgesetit it=edges.find(edge(p2,p1));
                if(it==edges.end()) edges.insert(edge(p1,p2));
                else edges.erase(it);
            }
        if(edges.empty()) 
            return; // no hole

        // get loops
        typedef std::map<unsigned int,unsigned int> edgemap;
        edgemap emap;
        for(edgesetit it=edges.begin(); it!=edges.end(); it++)
            emap[it->first]=it->second;

        typename edgemap::iterator it=emap.begin();
        std::vector<std::vector<unsigned int> > loops; loops.resize(1);
        loops.back().push_back(it->first);
        while(!emap.empty())
        {
            unsigned int i=it->second;
            loops.back().push_back(i);  // insert point in loop
            emap.erase(it);
            if(!emap.empty())
            {
                if(i==loops.back().front())  loops.push_back(std::vector<unsigned int>());  //  loop termination
                it=emap.find(i);
                if(it==emap.end())
                {
                    it=emap.begin();
                    loops.back().push_back(it->first);
                }
            }
        }
        if(this->f_printLog.getValue()) std::cout<<"MeshToImageEngine: found "<< loops.size()<<" loops"<<std::endl;
        //for(size_t i=0;i<loops.size();i++) for(size_t j=0;j<loops[i].size();j++) std::cout<<"loop "<<i<<","<<j<<":"<<loops[i][j]<<std::endl;

        // insert points at loop centroids and triangles connecting loop edges and centroids
        for(size_t i=0; i<loops.size(); i++)
            if(loops[i].size()>2)
            {
                Coord centroid;
                double centroidValue = 0.0;
                size_t indexCentroid=clpos.size()+loops[i].size()-1;
                for(size_t j=0; j<loops[i].size()-1; j++)
                {
                    unsigned int posIdx = loops[i][j];
                    clpos.push_back(pos[posIdx]);
                    centroid+=pos[posIdx];

                    if( clValues )
                    {
                        clValues->push_back( val[posIdx] );
                        centroidValue+=val[posIdx]; // TODO weight by distance to perform real barycentric interpolation
                    }

                    cltri.push_back(Triangle(indexCentroid,clpos.size()-1,j?clpos.size()-2:indexCentroid-1));
                }
                centroid/=(Real)(loops[i].size()-1);
                clpos.push_back(centroid);

                if( clValues )
                {
                    centroidValue /= (Real)(loops[i].size()-1); // TODO normalize by sum of weight
                    clValues->push_back( centroidValue );
                }
            }
    }







public:
    void createInputMeshesData()
    {
        unsigned int n = f_nbMeshes.getValue();

        createInputDataVector(n, vf_positions, "position", "input positions for mesh ", SeqPositions(), true);
        createInputDataVector(n, vf_edges, "edges", "input edges for mesh ", SeqEdges(), true);
        createInputDataVector(n, vf_triangles, "triangles", "input triangles for mesh ", SeqTriangles(), true);

        SeqValues defaultValues; defaultValues.push_back(1.0);
        createInputDataVector(n, vf_values, "value", "pixel value for mesh ", defaultValues, false);

        createInputDataVector(n, vf_fillInside, "fillInside", "fill the inside? (only valable for unique value)", true, false);

        createInputDataVector(n, vf_roiVertices, "roiVertices", "Region Of Interest, vertices index ", SeqIndex(), false);
        createInputDataVector(n, vf_roiValue, "roiValue", "pixel value for ROI ", 1.0, false);
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

            if( i>0 ) // to stay backward-compatible with the previous definition voxelizing only one input mesh
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
