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
#ifndef FLEXIBLE_DiffusionShapeFunction_H
#define FLEXIBLE_DiffusionShapeFunction_H

#include "../initFlexible.h"
#include "../shapeFunction/BaseShapeFunction.h"
#include "../shapeFunction/BaseImageShapeFunction.h"
#include "../types/PolynomialBasis.h"

#include <image/ImageTypes.h>
#include <image/ImageAlgorithms.h>

#include <sofa/helper/OptionsGroup.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <string>

//#include <Eigen/SparseCore>
//#include <Eigen/SparseCholesky>
//#include <Eigen/IterativeLinearSolvers>

#define HARMONIC 0
#define BIHARMONIC 1
#define ANISOTROPIC 2

#define GAUSS_SEIDEL 0
#define LDLT 1
#define CG 2
#define biCG 3



#ifdef SOFA_HAVE_MGDIFFUSI0N
#define MULTIGRID 4
#include <DiffusionSolver.h>
#endif


namespace sofa
{
namespace component
{
namespace shapefunction
{

using core::behavior::BaseShapeFunction;
using defaulttype::Mat;
using defaulttype::Vec;

/**
Shape functions computed using heat diffusion in images
  */


/// Default implementation does not compile
template <int imageTypeLabel>
struct DiffusionShapeFunctionSpecialization
{
};

/// Specialization for regular Image
template <>
struct DiffusionShapeFunctionSpecialization<defaulttype::IMAGELABEL_IMAGE>
{
    template<class DiffusionShapeFunction>
    static void init(DiffusionShapeFunction* This)
    {
        typedef typename DiffusionShapeFunction::ImageTypes ImageTypes;
        typedef typename DiffusionShapeFunction::raImage raImage;
        typedef typename DiffusionShapeFunction::DistTypes DistTypes;
        typedef typename DiffusionShapeFunction::waDist waDist;
        //typedef typename DiffusionShapeFunction::DistT DistT;
        typedef typename DiffusionShapeFunction::IndTypes IndTypes;
        typedef typename DiffusionShapeFunction::waInd waInd;

        // retrieve data
        raImage in(This->image);
        if(in->isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        //        const typename ImageTypes::CImgT& inimg = in->getCImg(0);  // suppose time=0

        // init dist
        typename DiffusionShapeFunction::imCoord dim = in->getDimensions(); dim[ImageTypes::DIMENSION_S]=dim[ImageTypes::DIMENSION_T]=1;

        waDist distData(This->f_distances);         distData->setDimensions(dim);
        //        typename DistTypes::CImgT& dist = distData->getCImg();

        // init indices and weights images
        unsigned int nbref=This->f_nbRef.getValue();        dim[ImageTypes::DIMENSION_S]=nbref;

        waInd indData(This->f_index); indData->setDimensions(dim);
        typename IndTypes::CImgT& indices = indData->getCImg(); indices.fill(0);

        waDist weightData(This->f_w);         weightData->setDimensions(dim);
        typename DistTypes::CImgT& weights = weightData->getCImg(); weights.fill(0);
    }
    /*
    template<class DiffusionShapeFunction>
    static Eigen::SparseMatrix<double> fillH(DiffusionShapeFunction* This, const CImg<double>& stencil, const CImg<unsigned long> ind, const unsigned long N)
    {
        Eigen::SparseMatrix<double> mat(N,N);         // default is column major

        // count non null entries
        unsigned int sparsity=0;    cimg_forXYZ(stencil,dx,dy,dz) if(stencil(dx,dy,dz)) sparsity++;
        unsigned int ox=0.5*(stencil.width()-1),oy=0.5*(stencil.height()-1),oz=0.5*(stencil.depth()-1);
        unsigned int border=max(ox,max(oy,oz));

        mat.reserve(Eigen::VectorXi::Constant(N,sparsity));

        cimg_for_insideXYZ(label,x,y,z,border)
                if(label(x,y,z)==1)
        {
            cimg_forXYZ(stencil,dx,dy,dz)
                    if(stencil(dx,dy,dz))
            {
                if(label(x+dx-ox,y+dy-oy,z+dz-oz)==1) mat.coeffRef(ind(x,y,z),ind(x+dx-ox,y+dy-oy,z+dz-oz))+=stencil(dx,dy,dz);
                else if(label(x+dx-ox,y+dy-oy,z+dz-oz)==0) mat.coeffRef(ind(x,y,z),ind(x,y,z))+=stencil(dx,dy,dz); // neumann bc
            }
        }
        mat.makeCompressed();
        return mat;
    }

    /// Apply Dirichlet conditions based on chosen parent
    template<class DiffusionShapeFunction>
    static Eigen::VectorXd fillb(DiffusionShapeFunction* This, const unsigned index)
                                 //const CImg<double>& stencil,const CImg<unsigned long> ind, const unsigned long N, const unsigned int index,const double temp)
    {
        Eigen::VectorXd b = Eigen::VectorXd::Zero(N);

        unsigned int ox=0.5*(stencil.width()-1),oy=0.5*(stencil.height()-1),oz=0.5*(stencil.depth()-1);
        unsigned int border=max(ox,max(oy,oz));

        cimg_for_insideXYZ(label,x,y,z,border)
                if(label(x,y,z)==1)
        {
            cimg_forXYZ(stencil,dx,dy,dz)
                    if(stencil(dx,dy,dz))
                    if(label(x+dx-ox,y+dy-oy,z+dz-oz)==index) b(ind(x,y,z))-=temp*stencil(dx,dy,dz);

        }

        return b;
    }
*/

    /// update weights and indices images based on computed diffusion image and current node index
    template<class DiffusionShapeFunction>
    static void updateWeights(DiffusionShapeFunction* This, const unsigned index)
    {
        // retrieve data
        typename DiffusionShapeFunction::raDist distData(This->f_distances);  const typename DiffusionShapeFunction::DistTypes::CImgT& dist = distData->getCImg();
        typename DiffusionShapeFunction::waInd indData(This->f_index);        typename DiffusionShapeFunction::IndTypes::CImgT& indices = indData->getCImg();
        typename DiffusionShapeFunction::waDist weightData(This->f_w);        typename DiffusionShapeFunction::DistTypes::CImgT& weights = weightData->getCImg();

        // copy from dist
        unsigned int nbref=This->f_nbRef.getValue();
        cimg_forXYZ(dist,x,y,z)
        {
            if( dist(x,y,z)>This->d_weightThreshold.getValue() ) // neglecting too small values
            {
                unsigned int j=0;
                while(j!=nbref && weights(x,y,z,j)>=dist(x,y,z)) j++;
                if(j!=nbref)
                {
                    if(j!=nbref-1) for(unsigned int k=nbref-1; k>j; k--) { weights(x,y,z,k)=weights(x,y,z,k-1); indices(x,y,z,k)=indices(x,y,z,k-1); }
                    weights(x,y,z,j)=dist(x,y,z);
                    indices(x,y,z,j)=index+1;
                }
            }
        }
    }


    /// normalize weights to partition unity
    template<class DiffusionShapeFunction>
    static void normalizeWeights(DiffusionShapeFunction* This)
    {
        typename DiffusionShapeFunction::waDist weightData(This->f_w);
        typename DiffusionShapeFunction::DistTypes::CImgT& weights = weightData->getCImg();
        cimg_forXYZ(weights,x,y,z)
        {
            typename DiffusionShapeFunction::DistT totW=0;
            cimg_forC(weights,c) totW+=weights(x,y,z,c);
            if(totW) cimg_forC(weights,c) weights(x,y,z,c)/=totW;
        }
    }


    /// init temperature according to dof position (to have interpolating weights) and provided boundary condition images
    template<class DiffusionShapeFunction>
    static void initTemp(DiffusionShapeFunction* This, const unsigned index)
    {
        // retrieve data
        typename DiffusionShapeFunction::raImage in(This->image);
        typename DiffusionShapeFunction::raTransform inT(This->transform);
        if(in->isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        const typename DiffusionShapeFunction::ImageTypes::CImgT& inimg = in->getCImg(0);  // suppose time=0

        typename DiffusionShapeFunction::waDist distData(This->f_distances);     typename DiffusionShapeFunction::DistTypes::CImgT& dist = distData->getCImg();

        typename DiffusionShapeFunction::raVecCoord parent(This->f_position);
        if(!parent.size()) { This->serr<<"Parent nodes not found"<<This->sendl; return; }

        // init temperatures
        dist.fill(-1);
        cimg_foroff(inimg,off) if(inimg[off]) dist[off]=/*DiffusionShapeFunction::MAXTEMP**/0.5;

        for(unsigned int i=0; i<parent.size(); i++)
        {
            typename DiffusionShapeFunction::Coord p = inT->toImageInt(parent[i]);
            if(in->isInside(p[0],p[1],p[2])) dist(p[0],p[1],p[2])=(i==index)?/*DiffusionShapeFunction::MAXTEMP*/1:0;
        }

        if(index<This->nbBoundaryConditions.getValue())
        {
            typename DiffusionShapeFunction::raDist bcData(This->f_boundaryConditions[index]);
            if(!bcData->isEmpty())
            {
                const  typename DiffusionShapeFunction::DistTypes::CImgT& bc = bcData->getCImg();
                cimg_foroff(bc,off)
                        if(bc[off]>=0)
                        dist[off]=bc[off];
            }
        }






//        DIRTY CODE TO EXPORT IMAGES TO BE DIFFUSED
//        CImg<float> img = dist; // convert in float

//        std::stringstream ss;
//        ss << "/tmp/DiffusionShapeFunction"<<index<<"_"<<dist.width()<<"_"<<dist.height()<<"_"<<dist.depth()<<"_"<<sizeof(float)<<".raw";
//        img.save( ss.str().c_str() );

//        CImg<char> mask(dist.width(),dist.height(),dist.depth(),1,-1);// outside
//        cimg_foroff(mask,off)
//                if( inimg[off]!=0 ) mask[off]=1; // inside

//        for(unsigned int i=0; i<parent.size(); i++)
//        {
//            typename DiffusionShapeFunction::Coord p = inT->toImageInt(parent[i]);
//            if(in->isInside(p[0],p[1],p[2])) mask(p[0],p[1],p[2])=0; // dirichlet
//        }

//        if(index<This->nbBoundaryConditions.getValue())
//        {
//            typename DiffusionShapeFunction::raDist bcData(This->f_boundaryConditions[index]);
//            if(!bcData->isEmpty())
//            {
//                const  typename DiffusionShapeFunction::DistTypes::CImgT& bc = bcData->getCImg();
//                cimg_foroff(bc,off)
//                        if(bc[off]>=0) mask[off]=0; // dirichlet
//            }
//        }

//        std::stringstream ss2;
//        ss2 << "/tmp/DiffusionShapeFunction"<<index<<"_mask_"<<dist.width()<<"_"<<dist.height()<<"_"<<dist.depth()<<"_"<<sizeof(char)<<".raw";
//        mask.save( ss2.str().c_str() );





    }


#ifdef SOFA_HAVE_MGDIFFUSI0N


    template<class DiffusionShapeFunction>
    static void buildDiffusionProblem(DiffusionShapeFunction* This, const unsigned index, CImg<float>& values, CImg<char>& mask)
    {
        // retrieve data
        typename DiffusionShapeFunction::raImage in(This->image);
        typename DiffusionShapeFunction::raTransform inT(This->transform);
        if(in->isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        const typename DiffusionShapeFunction::ImageTypes::CImgT& inimg = in->getCImg(0);  // suppose time=0

        typename DiffusionShapeFunction::raVecCoord parent(This->f_position);
        if(!parent.size()) { This->serr<<"Parent nodes not found"<<This->sendl; return; }

        // init temperatures
        values.resize(inimg.width(),inimg.height(),inimg.depth(),1);
        values.fill(0);

        mask.resize(inimg.width(),inimg.height(),inimg.depth(),1);
        mask.fill(DiffusionSolver<float>::OUTSIDE);
        cimg_foroff(mask,off)
                if( inimg[off]!=0 ) mask[off] = DiffusionSolver<float>::INSIDE;


        for(unsigned int i=0; i<parent.size(); i++)
        {
            typename DiffusionShapeFunction::Coord p = inT->toImageInt(parent[i]);
            if(in->isInside(p[0],p[1],p[2]))
            {
                values(p[0],p[1],p[2])=(i==index)?1:0;
                mask(p[0],p[1],p[2]) = DiffusionSolver<float>::DIRICHLET;
            }
        }

        if(index<This->nbBoundaryConditions.getValue())
        {
            typename DiffusionShapeFunction::raDist bcData(This->f_boundaryConditions[index]);
            if(!bcData->isEmpty())
            {
                const  typename DiffusionShapeFunction::DistTypes::CImgT& bc = bcData->getCImg();
                cimg_foroff(bc,off)
                {
                    if(bc[off]>=0)
                    {
                        values[off]=bc[off];
                        mask[off] = DiffusionSolver<float>::DIRICHLET; // dirichlet
                    }
                }
            }
        }

    }

    template<class DiffusionShapeFunction>
    static void solveMG(DiffusionShapeFunction* This, CImg<float>& values, CImg<char>& mask)
    {
        DiffusionSolver<float>::solveMG( values, mask, This->tolerance.getValue(), 0, This->iterations.getValue() );

        if( This->d_outsideDiffusion.getValue() )
        {
            cimg_for_insideXYZ(mask,x,y,z,1) // at least a one pixel outside border
            {
                char& m = mask(x,y,z);
                if( m == DiffusionSolver<float>::OUTSIDE ) m = DiffusionSolver<float>::INSIDE;
                else m = DiffusionSolver<float>::DIRICHLET;
            }

            DiffusionSolver<float>::solveMG( values, mask, This->tolerance.getValue(), 0, This->iterations.getValue() );
        }

        // convert from float // TODO improve that
        typename DiffusionShapeFunction::waDist distData(This->f_distances);
        typename DiffusionShapeFunction::DistTypes::CImgT& dist = distData->getCImg();
        dist = values;
    }

    template<class DiffusionShapeFunction>
    static void solveGS(DiffusionShapeFunction* This, CImg<float>& values, CImg<char>& mask)
    {
        DiffusionSolver<float>::solveGS( values, mask, This->iterations.getValue(), This->tolerance.getValue()/*, This->d_weightThreshold.getValue()*/ );

        if( This->d_outsideDiffusion.getValue() )
        {
            cimg_for_insideXYZ(mask,x,y,z,1) // at least a one pixel outside border
            {
                char& m = mask(x,y,z);
                if( m == DiffusionSolver<float>::OUTSIDE ) m = DiffusionSolver<float>::INSIDE;
                else m = DiffusionSolver<float>::DIRICHLET;
            }

            DiffusionSolver<float>::solveGS( values, mask, This->iterations.getValue(), This->tolerance.getValue(), This->d_weightThreshold.getValue() );
        }

        // convert from float // TODO improve that
        typename DiffusionShapeFunction::waDist distData(This->f_distances);
        typename DiffusionShapeFunction::DistTypes::CImgT& dist = distData->getCImg();
        dist = values;
    }



#endif

    /**
    * do one gauss seidel diffusion step : each pixel is replaced by the weighted average of its neighbors
    * weights are based on the (biased) distance : w= e(-d'²/sigma²) with d'=d/min(stiffness)
    * distances must be initialized to 1 or 0 at constrained pixels, to -1 outside, and between 0 and 1 elsewhere
    */
    /*
    template<typename real,typename T>
    real GaussSeidelDiffusionStep (CImg<real>& distances, const sofa::defaulttype::Vec<3,real>& voxelsize, const CImg<T>* biasFactor=NULL)
    {
        typedef sofa::defaulttype::Vec<3,int> iCoord;
        const iCoord dim(distances.width(),distances.height(),distances.depth());

        // init
        sofa::defaulttype::Vec<6,  iCoord > offset; // image coord offsets related to neighbors
        sofa::defaulttype::Vec<6,  real > lD;      // precomputed local distances (supposing that the transformation is linear)
        offset[0]=iCoord(-1,0,0);           lD[0]=voxelsize[0];
        offset[1]=iCoord(+1,0,0);           lD[1]=voxelsize[0];
        offset[2]=iCoord(0,-1,0);           lD[2]=voxelsize[1];
        offset[3]=iCoord(0,+1,0);           lD[3]=voxelsize[1];
        offset[4]=iCoord(0,0,-1);           lD[4]=voxelsize[2];
        offset[5]=iCoord(0,0,+1);           lD[5]=voxelsize[2];
        unsigned int nbOffset=offset.size();

        const real inv_variance=5.0; // =1/sigma² arbitrary

        real res=0; // return maximum absolute change
        cimg_for_insideXYZ(distances,x,y,z,1)
                if(distances(x,y,z)>0 && distances(x,y,z)<1)
        {
            real b1; if(biasFactor) b1=(real)(*biasFactor)(x,y,z); else  b1=1.0;
            real val=0;
            real W=0;
            for(unsigned int i=0;i<nbOffset;i++)
            {
                real b2; if(biasFactor) b2=(real)(*biasFactor)(x+offset[0],y+offset[1],z+offset[2]); else  b2=1.0;
                real d = lD[i]*1.0/sofa::helper::rmin(b1,b2);
                real w=exp(-d*d*inv_variance);
                if(distances(x+offset[0],y+offset[1],z+offset[2])!=-1.0) val+= w*distances(x+offset[0],y+offset[1],z+offset[2]);
                else val+= w*distances(x,y,z);
                W+=w;
            }
            if(W!=0) val /= W;
            if(res<fabs(val-distances(x,y,z))) res=fabs(val-distances(x,y,z));
            distances(x,y,z)=val;
        }
        return res;
    }*/

    template<class DiffusionShapeFunction>
    static double GaussSeidelStep(DiffusionShapeFunction* This, const unsigned index) //, const CImg<typename DiffusionShapeFunction::DistT>& stencil
    {
        typedef typename DiffusionShapeFunction::DistT DistT;

        // laplacian stencil
        CImg<DistT> stencil(3,3,3);
        stencil.fill(0);
        stencil(1,1,1)=-6.0;
        stencil(0,1,1)=stencil(2,1,1)=stencil(1,0,1)=stencil(1,2,1)=stencil(1,1,0)=stencil(1,1,2)=1.0;

        typename DiffusionShapeFunction::waDist distData(This->f_distances);     typename DiffusionShapeFunction::DistTypes::CImgT& dist = distData->getCImg();

        unsigned int ox=0.5*(stencil.width()-1),oy=0.5*(stencil.height()-1),oz=0.5*(stencil.depth()-1);
        unsigned int border=std::max(ox,std::max(oy,oz));

        const  typename DiffusionShapeFunction::DistTypes::CImgT* bc = NULL;
        if(index<This->nbBoundaryConditions.getValue())
        {
            typename DiffusionShapeFunction::raDist bcData(This->f_boundaryConditions[index]);
            if(!bcData->isEmpty()) bc = &(bcData->getCImg());
        }

        double res=0; // return maximum absolute change

        //#ifdef USING_OMP_PRAGMAS
        //        #pragma omp parallel for
        //#endif
        cimg_for_insideXYZ(dist,x,y,z,border)
                if(dist(x,y,z)>0 && dist(x,y,z)<1)
                if(!bc || (*bc)(x,y,z)==(DistT)-1)
        {
            DistT val=0;
            cimg_forXYZ(stencil,dx,dy,dz)
                    if(stencil(dx,dy,dz))
                    if(dx!=(int)ox || dy!=(int)oy || dz!=(int)oz)
            {
                if(dist(x+dx-ox,y+dy-oy,z+dz-oz)!=-1.0) val+=dist(x+dx-ox,y+dy-oy,z+dz-oz)*stencil(dx,dy,dz);
                else val+=dist(x,y,z)*stencil(dx,dy,dz);  // neumann boundary conditions
            }
            val = -val/stencil(ox,oy,oz);
            if(res<fabs(val-dist(x,y,z))) res=fabs(val-dist(x,y,z));
            dist(x,y,z)=val;
        }
        return res;
    }
};


template <class ShapeFunctionTypes_,class ImageTypes_>
class DiffusionShapeFunction : public BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_>
{
    friend struct DiffusionShapeFunctionSpecialization<defaulttype::IMAGELABEL_IMAGE>;
//    friend struct DiffusionShapeFunctionSpecialization<defaulttype::IMAGELABEL_BRANCHINGIMAGE>;

public:
    SOFA_CLASS(SOFA_TEMPLATE2(DiffusionShapeFunction, ShapeFunctionTypes_,ImageTypes_) , SOFA_TEMPLATE2(BaseImageShapeFunction, ShapeFunctionTypes_,ImageTypes_));
    typedef BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_> Inherit;

    /** @name  Shape function types */
    //@{
    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    typedef helper::ReadAccessor<Data<vector<Coord> > > raVecCoord;
    //@}

    /** @name  Image data */
    //@{
    typedef ImageTypes_ ImageTypes;
    typedef typename Inherit::T T;
    typedef typename Inherit::imCoord imCoord;
    typedef typename Inherit::raImage raImage;

    typedef typename Inherit::raTransform raTransform;

    typedef typename Inherit::DistT DistT;
    typedef typename Inherit::DistTypes DistTypes;
    typedef typename Inherit::raDist raDist;
    typedef typename Inherit::waDist waDist;
    Data< DistTypes > f_distances;
    Data<unsigned int> nbBoundaryConditions;
    helper::vector<Data<DistTypes>*> f_boundaryConditions;


    typedef typename Inherit::IndT IndT;
    typedef typename Inherit::IndTypes IndTypes;
    typedef typename Inherit::waInd waInd;
    //@}

    /** @name  Options */
    //@{
    Data<helper::OptionsGroup> method;
    Data<helper::OptionsGroup> solver;
    Data<unsigned int> iterations;
    Data<Real> tolerance;
    Data<Real> d_weightThreshold; ///< neglect smaller weights (another way to limit parents with nbref)
    Data<bool> biasDistances;


    Data<bool> d_clearData;
    Data<bool> d_outsideDiffusion;

//    static const DistT MAXTEMP;
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const DiffusionShapeFunction<ShapeFunctionTypes_,ImageTypes_>* = NULL) { return ShapeFunctionTypes_::Name()+std::string(",")+ImageTypes_::Name(); }


    virtual void init()
    {
        Inherit::init();

        createBoundaryConditionsData();

        // init weight and indice image
        DiffusionShapeFunctionSpecialization<ImageTypes::label>::init( this );

        if (this->method.getValue().getSelectedId() == HARMONIC)
        {

#ifdef SOFA_HAVE_MGDIFFUSI0N

            CImg<float> values;
            CImg<char> mask;

            //#ifdef USING_OMP_PRAGMAS
            //        #pragma omp parallel for
            //#endif
            for(unsigned int i=0; i<this->f_position.getValue().size(); i++)
            {
                DiffusionShapeFunctionSpecialization<ImageTypes::label>::buildDiffusionProblem(this,i,values,mask);

                if (this->solver.getValue().getSelectedId() == MULTIGRID)
                    DiffusionShapeFunctionSpecialization<ImageTypes::label>::solveMG(this,values,mask);
                else
                    DiffusionShapeFunctionSpecialization<ImageTypes::label>::solveGS(this,values,mask);

                DiffusionShapeFunctionSpecialization<ImageTypes::label>::updateWeights(this,i);
            }


#else

            if (this->solver.getValue().getSelectedId() == GAUSS_SEIDEL)
            {
                for(unsigned int i=0; i<this->f_position.getValue().size(); i++)
                {
                    DiffusionShapeFunctionSpecialization<ImageTypes::label>::initTemp(this,i);




                    double res=this->tolerance.getValue();
                    unsigned int nbit=0;
                    for(unsigned int it=0; it<this->iterations.getValue(); it++)
                        if(res>=this->tolerance.getValue())
                        {
                            res = DiffusionShapeFunctionSpecialization<ImageTypes::label>::GaussSeidelStep(this,i);
                            nbit++;
                        }
                    if(this->f_printLog.getValue()) std::cout<<this->getName()<<": dof "<<i<<": performed "<<nbit<<" iterations, residual="<<res<<std::endl;

                    DiffusionShapeFunctionSpecialization<ImageTypes::label>::updateWeights(this,i);
                }
            }
            /* else
            {
                // assemble system Hx = b
                Eigen::SparseMatrix<double> H = fillH(label,stencil,ind,N);
                Eigen::VectorXd b = fillb(label,stencil,ind,N,index,temp);

                // solve using Eigen
                if (this->solver.getValue().getSelectedId() == LDLT)
                {
                    Eigen::SimplicialCholesky<Eigen::SparseMatrix<double> > chol(H);
                    std::cout<<"done"<<std::endl;
                    std::cout<<"solve"<<std::endl;
                    Eigen::VectorXd X = chol.solve(b);
                    //    Eigen::VectorXd X = chol.solve(X1+b);
                    std::cout<<"done"<<std::endl;
                }
                else if (this->solver.getValue().getSelectedId() == CG)
                {
                        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,  Eigen::IncompleteLUT<double> > cg;
                        cg.setMaxIterations(500);
                        cg.setTolerance(1e-8);
                        cg.compute(H);
                        std::cout<<"done"<<std::endl;
                        std::cout<<"solve"<<std::endl;
                        Eigen::VectorXd X = cg.solve(b);
                        std::cout<<"done"<<std::endl;
                        std::cout << "#iterations:     " << cg.iterations() << std::endl;
                        std::cout << "estimated error: " << cg.error()      << std::endl;
                }
                else if (this->solver.getValue().getSelectedId() == biCG)
                {
                            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double> > cg(H);
                                cg.setTolerance(1e-8);
                            std::cout<<"done"<<std::endl;
                                std::cout<<"solve"<<std::endl;
                                Eigen::VectorXd X = cg.solve(b);
                                std::cout<<"done"<<std::endl;
                                std::cout << "#iterations:     " << cg.iterations() << std::endl;
                                std::cout << "estimated error: " << cg.error()      << std::endl;
                }
            }*/
#endif

        }

        DiffusionShapeFunctionSpecialization<ImageTypes::label>::normalizeWeights( this );

        if(this->d_clearData.getValue())
        {
            waDist dist(this->f_distances); dist->clear();
        }
    }


    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
    {
        const char* p = arg->getAttribute(nbBoundaryConditions.getName().c_str());
        if (p)
        {
            std::string nbStr = p;
            sout << "parse: setting nbBoundaryConditions="<<nbStr<<sendl;
            nbBoundaryConditions.read(nbStr);
            createBoundaryConditionsData();
        }
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str )
    {
        std::map<std::string,std::string*>::const_iterator it = str.find(nbBoundaryConditions.getName());
        if (it != str.end() && it->second)
        {
            std::string nbStr = *it->second;
            sout << "parseFields: setting nbBoundaryConditions="<<nbStr<<sendl;
            nbBoundaryConditions.read(nbStr);
            createBoundaryConditionsData();
        }
        Inherit1::parseFields(str);
    }

protected:
    DiffusionShapeFunction()
        :Inherit()
        , f_distances(initData(&f_distances,DistTypes(),"distances",""))
        , nbBoundaryConditions(initData(&nbBoundaryConditions,(unsigned int)0,"nbBoundaryConditions","Number of boundary condition images provided"))
        , method ( initData ( &method,"method","method" ) )
        , solver ( initData ( &solver,"solver","solver (param)" ) )
        , iterations(initData(&iterations,(unsigned int)100,"iterations","Max number of iterations for iterative solvers"))
        , tolerance(initData(&tolerance,(Real)1e-6,"tolerance","Error tolerance for iterative solvers"))
        , d_weightThreshold(initData(&d_weightThreshold,(Real)0,"weightThreshold","Thresold to neglect too small weights"))
        , biasDistances(initData(&biasDistances,false,"bias","Bias distances using inverse pixel values"))
        , d_clearData(initData(&d_clearData,true,"clearData","clear diffusion image after computation?"))
        , d_outsideDiffusion(initData(&d_outsideDiffusion,false,"outsideDiffusion","propagate shape function outside of the object? (can be useful for embeddings)"))
    {
        helper::OptionsGroup methodOptions(3,"0 - Harmonic"
                                           ,"1 - bi-Harmonic"
                                           ,"2 - Anisotropic"
                                           );
        methodOptions.setSelectedItem(HARMONIC);
        method.setValue(methodOptions);
        method.setGroup("parameters");

        helper::OptionsGroup solverOptions(4
#ifdef SOFA_HAVE_MGDIFFUSI0N
                                          +1
#endif
                                           ,"0 - Gauss-Seidel"
                                           ,"1 - LDLT"
                                           ,"2 - CG"
                                           ,"3 - biCG"
#ifdef SOFA_HAVE_MGDIFFUSI0N
                                           ,"4 - Multigrid"
#endif
                                           );
        solverOptions.setSelectedItem(GAUSS_SEIDEL);
        solver.setValue(solverOptions);
        solver.setGroup("parameters");

        biasDistances.setGroup("parameters");

        createBoundaryConditionsData();

    }

    virtual ~DiffusionShapeFunction()
    {
        deleteInputDataVector(f_boundaryConditions);

    }



    template<class t>
    void createInputDataVector(unsigned int nb, helper::vector< Data<t>* >& vf, std::string name, std::string help)
    {
        vf.reserve(nb);
        for (unsigned int i=vf.size(); i<nb; i++)
        {
            std::ostringstream oname; oname << name << (1+i); std::string name_i = oname.str();

            Data<t>* d = new Data<t>();
            d->setName(name_i);
            d->setHelp(help.c_str());
            d->setReadOnly(true);

            vf.push_back(d);
            this->addData(d);
        }
    }
    template<class t>
    void deleteInputDataVector(helper::vector< Data<t>* >& vf)
    {
        for (unsigned int i=0; i<vf.size(); ++i) delete vf[i];
        vf.clear();
    }

    void createBoundaryConditionsData()
    {
        createInputDataVector(this->nbBoundaryConditions.getValue(), f_boundaryConditions, "boundaryConditions", "boundaryConditions");
    }
};

//template<class ShapeFunctionTypes,class ImageTypes>
//const typename DiffusionShapeFunction<ShapeFunctionTypes,ImageTypes>::DistT DiffusionShapeFunction<ShapeFunctionTypes,ImageTypes>::MAXTEMP = (DistT)1.0;

}
}
}


#endif

