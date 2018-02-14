/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef FLEXIBLE_DiffusionShapeFunction_H
#define FLEXIBLE_DiffusionShapeFunction_H

#include <Flexible/config.h>
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

//#define HARMONIC 0
//#define BIHARMONIC 1
//#define ANISOTROPIC 2

#define GAUSS_SEIDEL 0
#define JACOBI 1
#define CG 2
#include <image/extlibs/DiffusionSolver/DiffusionSolver.h>



namespace sofa
{
namespace component
{
namespace shapefunction
{


/**
Shape functions computed using heat diffusion in images

@author Matthieu Nesme

  */


/// Default implementation does not compile
template <class ImageType>
struct DiffusionShapeFunctionSpecialization
{
};


/// Specialization for regular Image
template <class T>
struct DiffusionShapeFunctionSpecialization<defaulttype::Image<T>>
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
                while(j<nbref && weights(x,y,z,j)>=dist(x,y,z)) j++; // find the right ordered place
                if(j<nbref) // no too small weight
                {
                    /*if(j!=nbref-1) */for(unsigned int k=nbref-1; k>j; k--) { weights(x,y,z,k)=weights(x,y,z,k-1); indices(x,y,z,k)=indices(x,y,z,k-1); } // ending weights are moved back
                    weights(x,y,z,j)=dist(x,y,z); indices(x,y,z,j)=index+1; // current weight is inserted
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
        cimg_foroff(inimg,off) if(inimg[off]) dist[off] = 0;


        for(unsigned int i=0; i<parent.size(); i++)
        {
            typename DiffusionShapeFunction::Coord p = inT->toImageInt(parent[i]);
            if(in->isInside(p[0],p[1],p[2])) dist(p[0],p[1],p[2])=(i==index)?1:0;
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

    }



    template<class DiffusionShapeFunction>
    static void buildMaterialImage(DiffusionShapeFunction* This, cimg_library::CImg<float>& material)
    {
        typename DiffusionShapeFunction::raImage in(This->image);
        if(in->isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        material = in->getCImg(0);
    }

    template<class DiffusionShapeFunction>
    static void buildDiffusionProblem(DiffusionShapeFunction* This, const unsigned index, cimg_library::CImg<float>& values, cimg_library::CImg<char>& mask)
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
        cimg_foroff(mask,off)
            mask[off] = !inimg[off] ? DiffusionSolver<float>::OUTSIDE : DiffusionSolver<float>::INSIDE;


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
    static void solveGS(DiffusionShapeFunction* This, cimg_library::CImg<float>& values, cimg_library::CImg<char>& mask, cimg_library::CImg<float>* material=NULL)
    {
        typename DiffusionShapeFunction::TransformType::Coord spacing = This->transform.getValue().getScale();
        DiffusionSolver<float>::solveGS( values, mask, spacing[0], spacing[1], spacing[2], This->iterations.getValue(), This->tolerance.getValue(), 1.5, material /*This->d_weightThreshold.getValue()*/ );

        if( This->d_outsideDiffusion.getValue() )
        {
            cimg_for_insideXYZ(mask,x,y,z,1) // at least a one pixel outside border
            {
                char& m = mask(x,y,z);
                if( m == DiffusionSolver<float>::OUTSIDE ) m = DiffusionSolver<float>::INSIDE;
                else m = DiffusionSolver<float>::DIRICHLET;
            }

            DiffusionSolver<float>::solveGS( values, mask, spacing[0], spacing[1], spacing[2], This->iterations.getValue(), This->tolerance.getValue(), 1.5 /*This->d_weightThreshold.getValue()*/ );
        }

        // convert from float // TODO improve that
        typename DiffusionShapeFunction::waDist distData(This->f_distances);
        typename DiffusionShapeFunction::DistTypes::CImgT& dist = distData->getCImg();
        dist = values;
    }

    template<class DiffusionShapeFunction>
    static void solveJacobi(DiffusionShapeFunction* This, cimg_library::CImg<float>& values, cimg_library::CImg<char>& mask, cimg_library::CImg<float>* material=NULL)
    {
        typename DiffusionShapeFunction::TransformType::Coord spacing = This->transform.getValue().getScale();
        DiffusionSolver<float>::solveJacobi( values, mask, spacing[0], spacing[1], spacing[2], This->iterations.getValue(), This->tolerance.getValue(), material );

        if( This->d_outsideDiffusion.getValue() )
        {
            cimg_for_insideXYZ(mask,x,y,z,1) // at least a one pixel outside border
            {
                char& m = mask(x,y,z);
                if( m == DiffusionSolver<float>::OUTSIDE ) m = DiffusionSolver<float>::INSIDE;
                else m = DiffusionSolver<float>::DIRICHLET;
            }

            DiffusionSolver<float>::solveJacobi( values, mask, spacing[0], spacing[1], spacing[2], This->iterations.getValue(), This->tolerance.getValue() );
        }

        // convert from float // TODO improve that
        typename DiffusionShapeFunction::waDist distData(This->f_distances);
        typename DiffusionShapeFunction::DistTypes::CImgT& dist = distData->getCImg();
        dist = values;
    }

    template<class DiffusionShapeFunction>
    static void solveCG(DiffusionShapeFunction* This, cimg_library::CImg<float>& values, cimg_library::CImg<char>& mask, cimg_library::CImg<float>* material=NULL)
    {
        typename DiffusionShapeFunction::TransformType::Coord spacing = This->transform.getValue().getScale();
        DiffusionSolver<float>::solveCG( values, mask, spacing[0], spacing[1], spacing[2], This->iterations.getValue(), This->tolerance.getValue(), material );

        if( This->d_outsideDiffusion.getValue() )
        {
            cimg_for_insideXYZ(mask,x,y,z,1) // at least a one pixel outside border
            {
                char& m = mask(x,y,z);
                if( m == DiffusionSolver<float>::OUTSIDE ) m = DiffusionSolver<float>::INSIDE;
                else m = DiffusionSolver<float>::DIRICHLET;
            }

            DiffusionSolver<float>::solveCG( values, mask, spacing[0], spacing[1], spacing[2], This->iterations.getValue(), This->tolerance.getValue() );
        }

        // convert from float // TODO improve that
        typename DiffusionShapeFunction::waDist distData(This->f_distances);
        typename DiffusionShapeFunction::DistTypes::CImgT& dist = distData->getCImg();
        dist = values;
    }

};


template <class ShapeFunctionTypes_,class ImageTypes_>
class DiffusionShapeFunction : public BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_>
{
    friend struct DiffusionShapeFunctionSpecialization<ImageTypes_>;

public:
    SOFA_CLASS(SOFA_TEMPLATE2(DiffusionShapeFunction, ShapeFunctionTypes_,ImageTypes_) , SOFA_TEMPLATE2(BaseImageShapeFunction, ShapeFunctionTypes_,ImageTypes_));
    typedef BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_> Inherit;

    /** @name  Shape function types */
    //@{
    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    typedef helper::ReadAccessor<Data<helper::vector<Coord> > > raVecCoord;
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
    Data<unsigned int> nbBoundaryConditions; ///< Number of boundary condition images provided
    helper::vector<Data<DistTypes>*> f_boundaryConditions;


    typedef typename Inherit::IndT IndT;
    typedef typename Inherit::IndTypes IndTypes;
    typedef typename Inherit::waInd waInd;
    //@}

    /** @name  Options */
    //@{
    Data<helper::OptionsGroup> method;
    Data<helper::OptionsGroup> solver; ///< solver (param)
    Data<unsigned int> iterations; ///< Max number of iterations for iterative solvers
    Data<Real> tolerance; ///< Error tolerance for iterative solvers
    Data<Real> d_weightThreshold; ///< neglect smaller weights (another way to limit parents with nbref)
    Data<bool> biasDistances; ///< Bias distances using inverse pixel values


    Data<bool> d_clearData; ///< clear diffusion image after computation?
    Data<bool> d_outsideDiffusion; ///< propagate shape function outside of the object? (can be useful for embeddings)

    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const DiffusionShapeFunction<ShapeFunctionTypes_,ImageTypes_>* = NULL) { return ShapeFunctionTypes_::Name()+std::string(",")+ImageTypes_::Name(); }


    virtual void init()
    {
        Inherit::init();

        createBoundaryConditionsData();

        // init weight and indice image
        DiffusionShapeFunctionSpecialization<ImageTypes>::init( this );

//        if (this->method.getValue().getSelectedId() == HARMONIC)
        {

//            DiffusionSolver<float>::setNbThreads( 1 );
            DiffusionSolver<float>::setDefaultNbThreads();

            cimg_library::CImg<float> values;
            cimg_library::CImg<char> mask;
            cimg_library::CImg<float> material, *materialPtr = NULL;

            if( biasDistances.getValue() )
            {
                DiffusionShapeFunctionSpecialization<ImageTypes>::buildMaterialImage(this,material);
                materialPtr = &material;
            }

//            if( materialPtr ) materialPtr->display("materialPtr");




            for(unsigned int i=0; i<this->f_position.getValue().size(); i++)
            {
                DiffusionShapeFunctionSpecialization<ImageTypes>::buildDiffusionProblem(this,i,values,mask);

//                values.display("values");
//                mask.display("mask");

#ifndef NDEBUG
                // checking that there is at least a one pixel outside border
                // it is a limitation that dramatically improves performances by removing boundary testing
                cimg_forXYZ(mask,x,y,z)
                    if( x==0 || y==0 || z==0 || x==mask.width()-1 || y==mask.height()-1 || z==mask.depth()-1 )
                        assert( mask(x,y,z) == DiffusionSolver<float>::OUTSIDE && "DiffusionShapeFunction mask must have at least one pixel outside border" );
#endif

                switch( this->solver.getValue().getSelectedId() )
                {
                    case JACOBI:
                        DiffusionShapeFunctionSpecialization<ImageTypes>::solveJacobi(this,values,mask,materialPtr);
                        break;
                    case CG:
                        DiffusionShapeFunctionSpecialization<ImageTypes>::solveCG(this,values,mask,materialPtr);
                        break;
                    case GAUSS_SEIDEL:
                    default:
                        DiffusionShapeFunctionSpecialization<ImageTypes>::solveGS(this,values,mask,materialPtr);
                        break;
                }

//                values.display("diffused");

                DiffusionShapeFunctionSpecialization<ImageTypes>::updateWeights(this,i);
            }


        }

        DiffusionShapeFunctionSpecialization<ImageTypes>::normalizeWeights( this );

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
//        , method ( initData ( &method,"method","method" ) )
        , solver ( initData ( &solver,"solver","solver (param)" ) )
        , iterations(initData(&iterations,(unsigned int)100,"iterations","Max number of iterations for iterative solvers"))
        , tolerance(initData(&tolerance,(Real)1e-6,"tolerance","Error tolerance for iterative solvers"))
        , d_weightThreshold(initData(&d_weightThreshold,(Real)0,"weightThreshold","Thresold to neglect too small weights"))
        , biasDistances(initData(&biasDistances,false,"bias","Bias distances using inverse pixel values"))
        , d_clearData(initData(&d_clearData,true,"clearData","clear diffusion image after computation?"))
        , d_outsideDiffusion(initData(&d_outsideDiffusion,false,"outsideDiffusion","propagate shape function outside of the object? (can be useful for embeddings)"))
    {
//        helper::OptionsGroup methodOptions(3,"0 - Harmonic"
//                                           ,"1 - bi-Harmonic"
//                                           ,"2 - Anisotropic"
//                                           );
//        methodOptions.setSelectedItem(HARMONIC);
//        method.setValue(methodOptions);
//        method.setGroup("parameters");

        helper::OptionsGroup solverOptions(3
                                           ,"0 - Gauss-Seidel"
                                           ,"1 - Jacobi"
                                           ,"2 - CG" );
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

}
}
}


#endif

