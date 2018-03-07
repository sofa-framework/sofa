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
#ifndef SOFA_DEFAULTTYPE_DUALQUATBLENTYPES_INL
#define SOFA_DEFAULTTYPE_DUALQUATBLENTYPES_INL

#include <sofa/helper/DualQuat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/gl/template.h>


namespace sofa
{
namespace defaulttype
{

template<class Real,int Dim>
Vec<Dim*Dim,Real>& MattoVec(Mat<Dim,Dim,Real>& m) { return *reinterpret_cast<Vec<Dim*Dim,Real>*>( &m[0]); }
template<class Real,int Dim>
const Vec<Dim*Dim,Real>& MattoVec(const Mat<Dim,Dim,Real>& m) { return *reinterpret_cast<const Vec<Dim*Dim,Real>*>( &m[0]); }

template<class Real,int Dim>
Mat<Dim,Dim,Real>& VectoMat(Vec<Dim*Dim,Real>& v) { return *reinterpret_cast<Mat<Dim,Dim,Real>*>( &v[0]); }
template<class Real,int Dim>
const Mat<Dim,Dim,Real>& VectoMat(const Vec<Dim*Dim,Real>& v) { return *reinterpret_cast<const Mat<Dim,Dim,Real>*>( &v[0]); }

//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdAffineTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 0
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdAffineTypes<3,InReal> In;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    bool avoidWarning; // TODO no attribute leads to gcc warnings. Remove this one when the class will be implemented.

    void init( const OutCoord& /*InitialPos*/, const Vec<nbRef,unsigned int>& /*Index*/, const VecInCoord& /*InitialTransform*/, const Vec<nbRef,Real>& /*w*/, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
    }

    OutCoord apply( const VecInCoord& /*d*/ )  // Called in Apply
    {
        OutCoord result;
        return result;
    }

    OutDeriv mult( const VecInDeriv& /*d*/ ) // Called in ApplyJ
    {
        OutDeriv result;
        return result;
    }

    void addMultTranspose( VecInDeriv& /*res*/, const OutDeriv& /*d*/ ) // Called in ApplyJT
    {
    }

    void addMultTranspose( ParentJacobianRow& /*parentJacobianRow*/, const OutDeriv& /*childJacobianVec*/ ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};




//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine -> Affine =  -> DefGradient1 with dw=0
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdAffineTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 3
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdAffineTypes<3,InReal> In;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    bool avoidWarning; // TODO no attribute leads to gcc warnings. Remove this one when the class will be implemented.

    void init( const OutCoord& /*InitialPos*/, const Vec<nbRef,unsigned int>& /*Index*/, const VecInCoord& /*InitialTransform*/, const Vec<nbRef,Real>& /*w*/, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
    }

    OutCoord apply( const VecInCoord& /*d*/ )  // Called in Apply
    {
        OutCoord result;
        return result;
    }

    OutDeriv mult( const VecInDeriv& /*d*/ ) // Called in ApplyJ
    {
        OutDeriv result;
        return result;
    }

    void addMultTranspose( VecInDeriv& /*res*/, const OutDeriv& /*d*/ ) // Called in ApplyJT
    {
    }

    void addMultTranspose( ParentJacobianRow& /*parentJacobianRow*/, const OutDeriv& /*childJacobianVec*/ ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine -> Rigid =  -> Affine with polarDecomposition
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdAffineTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 4
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdAffineTypes<3,InReal> In;
    typedef typename StdAffineTypes<3,OutReal>::Coord Affine;
    typedef Mat<3,3,OutReal> Mat33;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;
    typedef helper::Quater<OutReal> Quat;

    bool avoidWarning; // TODO no attribute leads to gcc warnings. Remove this one when the class will be implemented.

    void init( const OutCoord& /*InitialPos*/, const Vec<nbRef,unsigned int>& /*Index*/, const VecInCoord& /*InitialTransform*/, const Vec<nbRef,Real>& /*w*/, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
    }

    OutCoord apply( const VecInCoord& /*d*/ )  // Called in Apply
    {
        OutCoord result;
        return result;
    }

    OutDeriv mult( const VecInDeriv& /*d*/ ) // Called in ApplyJ
    {
        OutDeriv result;
        return result;
    }

    void addMultTranspose( VecInDeriv& /*res*/, const OutDeriv& /*d*/ ) // Called in ApplyJT
    {
    }

    void addMultTranspose( ParentJacobianRow& /*parentJacobianRow*/, const OutDeriv& /*childJacobianVec*/ ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdAffineTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 1
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdAffineTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<Out::material_dimensions,Out::material_dimensions, InReal> MaterialFrame;
    typedef Vec<Out::material_dimensions, MaterialFrame> MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    bool avoidWarning; // TODO no attribute leads to gcc warnings. Remove this one when the class will be implemented.

    void init( const OutCoord& /*InitialPos*/, const Vec<nbRef,unsigned int>& /*Index*/, const VecInCoord& /*InitialTransform*/, const Vec<nbRef,Real>& /*w*/, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
    }

    OutCoord apply( const VecInCoord& /*d*/ )  // Called in Apply
    {
        OutCoord result;
        return result;
    }

    OutDeriv mult( const VecInDeriv& /*d*/ ) // Called in ApplyJ
    {
        OutDeriv result;
        return result;
    }

    void addMultTranspose( VecInDeriv& /*res*/, const OutDeriv& /*d*/ ) // Called in ApplyJT
    {
    }

    void addMultTranspose( ParentJacobianRow& /*parentJacobianRow*/, const OutDeriv& /*childJacobianVec*/ ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Affine->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class  _Material, int nbRef>
struct DualQuatBlending<
        StdAffineTypes<3,typename _Material::Real>,
        Out,
        _Material, nbRef,2
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdAffineTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<Out::material_dimensions,Out::material_dimensions, InReal> MaterialFrame;
    typedef Vec<Out::material_dimensions, MaterialFrame> MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    bool avoidWarning; // TODO no attribute leads to gcc warnings. Remove this one when the class will be implemented.

    void init( const OutCoord& /*InitialPos*/, const Vec<nbRef,unsigned int>& /*Index*/, const VecInCoord& /*InitialTransform*/, const Vec<nbRef,Real>& /*w*/, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
    }

    OutCoord apply( const VecInCoord& /*d*/ )  // Called in Apply
    {
        OutCoord result;
        return result;
    }

    OutDeriv mult( const VecInDeriv& /*d*/ ) // Called in ApplyJ
    {
        OutDeriv result;
        return result;
    }

    void addMultTranspose( VecInDeriv& /*res*/, const OutDeriv& /*d*/ ) // Called in ApplyJT
    {
    }

    void addMultTranspose( ParentJacobianRow& /*parentJacobianRow*/, const OutDeriv& /*childJacobianVec*/ ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdAffineTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdQuadraticTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,0
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdQuadraticTypes<3,InReal> In;
    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    bool avoidWarning; // TODO no attribute leads to gcc warnings. Remove this one when the class will be implemented.

    void init( const OutCoord& /*InitialPos*/, const Vec<nbRef,unsigned int>& /*Index*/, const VecInCoord& /*InitialTransform*/, const Vec<nbRef,Real>& /*w*/, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
    }

    OutCoord apply( const VecInCoord& /*d*/ )  // Called in Apply
    {
        OutCoord result;
        return result;
    }

    OutDeriv mult( const VecInDeriv& /*d*/ ) // Called in ApplyJ
    {
        OutDeriv result;
        return result;
    }

    void addMultTranspose( VecInDeriv& /*res*/, const OutDeriv& /*d*/ ) // Called in ApplyJT
    {
    }

    void addMultTranspose( ParentJacobianRow& /*parentJacobianRow*/, const OutDeriv& /*childJacobianVec*/ ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic -> Affine  =  -> DefGradient1 with dw=0
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdQuadraticTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,3
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdQuadraticTypes<3,InReal> In;
    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    typedef typename In::Affine Affine;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<In::spatial_dimensions*In::spatial_dimensions,3,Real> QuadraticMat; // mat 9x3
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    bool avoidWarning; // TODO no attribute leads to gcc warnings. Remove this one when the class will be implemented.

    void init( const OutCoord& /*InitialPos*/, const Vec<nbRef,unsigned int>& /*Index*/, const VecInCoord& /*InitialTransform*/, const Vec<nbRef,Real>& /*w*/, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
    }

    OutCoord apply( const VecInCoord& /*d*/ )  // Called in Apply
    {
        OutCoord result;
        return result;
    }

    OutDeriv mult( const VecInDeriv& /*d*/ ) // Called in ApplyJ
    {
        OutDeriv result;
        return result;
    }

    void addMultTranspose( VecInDeriv& /*res*/, const OutDeriv& /*d*/ ) // Called in ApplyJT
    {
    }

    void addMultTranspose( ParentJacobianRow& /*parentJacobianRow*/, const OutDeriv& /*childJacobianVec*/ ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdQuadraticTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,1
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdQuadraticTypes<3,InReal> In;
    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    typedef typename In::Affine Affine;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<9,3,Real> MaterialFrame2;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    bool avoidWarning; // TODO no attribute leads to gcc warnings. Remove this one when the class will be implemented.

    void init( const OutCoord& /*InitialPos*/, const Vec<nbRef,unsigned int>& /*Index*/, const VecInCoord& /*InitialTransform*/, const Vec<nbRef,Real>& /*w*/, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
    }

    OutCoord apply( const VecInCoord& /*d*/ )  // Called in Apply
    {
        OutCoord result;
        return result;
    }

    OutDeriv mult( const VecInDeriv& /*d*/ ) // Called in ApplyJ
    {
        OutDeriv result;
        return result;
    }

    void addMultTranspose( VecInDeriv& /*res*/, const OutDeriv& /*d*/ ) // Called in ApplyJT
    {
    }

    void addMultTranspose( ParentJacobianRow& /*parentJacobianRow*/, const OutDeriv& /*childJacobianVec*/ ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Quadratic->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdQuadraticTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,2
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdQuadraticTypes<3,InReal> In;
    //typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<Out::material_dimensions,Out::material_dimensions, InReal> MaterialFrame;
    typedef Vec<Out::material_dimensions, MaterialFrame> MaterialFrameGradient;
    typedef Mat<9,3,Real> MaterialFrame2;
    typedef Vec<3,MaterialFrame2> MaterialFrameGradient2;
    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    typedef typename In::Affine Affine;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    bool avoidWarning; // TODO no attribute leads to gcc warnings. Remove this one when the class will be implemented.

    void init( const OutCoord& /*InitialPos*/, const Vec<nbRef,unsigned int>& /*Index*/, const VecInCoord& /*InitialTransform*/, const Vec<nbRef,Real>& /*w*/, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
    }

    OutCoord apply( const VecInCoord& /*d*/ )  // Called in Apply
    {
        OutCoord result;
        return result;
    }

    OutDeriv mult( const VecInDeriv& /*d*/ ) // Called in ApplyJ
    {
        OutDeriv result;
        return result;
    }

    void addMultTranspose( VecInDeriv& /*res*/, const OutDeriv& /*d*/ ) // Called in ApplyJT
    {
    }

    void addMultTranspose( ParentJacobianRow& /*parentJacobianRow*/, const OutDeriv& /*childJacobianVec*/ ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};



// Warning !!!! Onlly the declaration is done. The actual definitions of the methods come from Quadratic->Affine (type case 3)
template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdQuadraticTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,5
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdQuadraticTypes<3,InReal> In;
    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    typedef typename In::Affine Affine;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<In::spatial_dimensions*In::spatial_dimensions,3,Real> QuadraticMat; // mat 9x3
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;

    bool avoidWarning; // TODO no attribute leads to gcc warnings. Remove this one when the class will be implemented.

    void init( const OutCoord& /*InitialPos*/, const Vec<nbRef,unsigned int>& /*Index*/, const VecInCoord& /*InitialTransform*/, const Vec<nbRef,Real>& /*w*/, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
    }

    OutCoord apply( const VecInCoord& /*d*/ )  // Called in Apply
    {
        OutCoord result;
        return result;
    }

    OutDeriv mult( const VecInDeriv& /*d*/ ) // Called in ApplyJ
    {
        OutDeriv result;
        return result;
    }

    void addMultTranspose( VecInDeriv& /*res*/, const OutDeriv& /*d*/ ) // Called in ApplyJT
    {
    }

    void addMultTranspose( ParentJacobianRow& /*parentJacobianRow*/, const OutDeriv& /*childJacobianVec*/ ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
    }

    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 5>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdQuadraticTypes<3,typename _Material::Real>,Out, _Material, nbRef, 5>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};





//////////////////////////////////////////////////////////////////////////////////
////  Rigid->Vec
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdRigidTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,0
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdRigidTypes<3,InReal> In;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;
    typedef typename In::Deriv InDeriv;
    typedef helper::DualQuatCoord3<InReal> DQCoord;

    struct JacobianBlock
    {
        /** Dual quat skinning: p = sum_i w_i q_i*q0_i^-1 / |sum_i w_i q_i*q0_i^-1| (p_0)
          */

        Real w;
        Mat<4,4,Real> T0; Mat<4,4,Real> TE; ///< Real/Dual part of blended quaternion Jacobian : db = [T0,TE] dq

        Mat<3,3,Real> Pa; ///< dp = Pa.Omega_i  : affine part
        Mat<3,3,Real> Pt; ///< dp = Pt.dt_i : translation part
    };

    Vec<nbRef,unsigned int> index;
    OutCoord P0;	///< initial position
    DQCoord b;		///< linearly blended dual quaternions : b= sum_i w_i q_i*q0_i^-1
    DQCoord bn;		///< normalized dual quaternion : bn=b/|b|

    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        P0 = InitialPos;
        unsigned int i = 0 ;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            Jb[i].w= w[i];
            DQCoord T0inv=InitialTransform[index[i]]; T0inv.invert();
            T0inv.multLeft_getJ(Jb[i].T0,Jb[i].TE);
            Jb[i].T0*=w[i]; Jb[i].TE*=w[i];
        }
        if ( i<nbRef ) Jb[i].w=(Real)0; // used for loop terminations
        apply(InitialTransform);
    }


    OutCoord apply( const VecInCoord& in )  // Called in Apply
    {
        OutCoord result;

        DQCoord q;		// frame current position in DQ form
        b.clear();
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            // weighted relative transform : w_i.q_i*q0_i^-1
            q.getDual() = Jb[i].TE*q.getOrientation() + Jb[i].T0*q.getDual();
            q.getOrientation() = Jb[i].T0*q.getOrientation();
            b+=q;
        }

        bn=b; bn.normalize();
        Mat<4,4,Real> N0; Mat<4,4,Real> NE; // Real/Dual part of the normalization Jacobian : dbn = [N0,NE] db
        b.normalize_getJ( N0 , NE );

        result = bn.pointToParent( P0 );
        Mat<3,4,Real> Q0; Mat<3,4,Real> QE; // Real/Dual part of the transformation Jacobian : dP = [Q0,QE] dbn
        bn.pointToParent_getJ( Q0 , QE , P0 );

        Mat<3,4,Real> QN0 = Q0*N0 + QE*NE , QNE = QE * N0;
        Mat<4,3,Real> TL0 , TLE;

        for ( unsigned int i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            q.velocity_getJ ( TL0 , TLE );  // Real/Dual part of quaternion Jacobian : dq_i = [L0,LE] [Omega_i, dt_i]
            TLE  = Jb[i].TE * TL0 + Jb[i].T0 * TLE;
            TL0  = Jb[i].T0 * TL0 ;
            // dP = QNTL [Omega_i, dt_i]
            Jb[i].Pa = QN0 * TL0 + QNE * TLE;
            Jb[i].Pt = QNE * TL0 ;
        }

        return result;
    }

    OutDeriv mult( const VecInDeriv& in ) // Called in ApplyJ
    {
        OutDeriv result;

        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            result += Jb[i].Pt * getLinear( in[index[i]] )  + Jb[i].Pa * getAngular(in[index[i]]);
        }
        return result;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            getLinear(res[index[i]])  += Jb[i].Pt.transposed() * d;
            getAngular(res[index[i]]) += Jb[i].Pa.transposed() * d;
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            InDeriv parentJacobianVec;
            getLinear(parentJacobianVec)  += Jb[i].Pt.transposed() * childJacobianVec;
            getAngular(parentJacobianVec) += Jb[i].Pa.transposed() * childJacobianVec;
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 0>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid -> Affine =  -> Rigid
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdRigidTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 3
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdRigidTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<3,3, InReal> MaterialFrame;
    typedef Vec<3, MaterialFrame> MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<3,3,Real> Mat33;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;
    typedef typename In::Deriv InDeriv;
    typedef helper::DualQuatCoord3<InReal> DQCoord;


    struct JacobianBlock
    {
        /** Dual quat skinning: p = sum_i w_i q_i*q0_i^-1 / |sum_i w_i q_i*q0_i^-1| (p_0)
          */
        Real w;

        Mat<4,4,Real> T0; Mat<4,4,Real> TE; ///< Real/Dual part of blended quaternion Jacobian : db = [T0,TE] dq

        Mat<3,3,Real> Pa; ///< dp = Pa.Omega_i  : rotation part
        Mat<3,3,Real> Pt; ///< dp = Pt.dt_i : translation part
        Mat<9,3,Real> Fa; ///< dA = Fa.Omega_i  : rotation part
    };

    Vec<nbRef,unsigned int> index;
    OutCoord P0;	///< initial position
    DQCoord b;		///< linearly blended dual quaternions : b= sum_i w_i q_i*q0_i^-1
    DQCoord bn;		///< normalized dual quaternion : bn=b/|b|
    Mat33 R;

    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        P0 = InitialPos;
        unsigned int i = 0 ;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            Jb[i].w= w[i];
            DQCoord T0inv=InitialTransform[index[i]]; T0inv.invert();
            T0inv.multLeft_getJ(Jb[i].T0,Jb[i].TE);
            Jb[i].T0*=w[i]; Jb[i].TE*=w[i];
        }
        if ( i<nbRef ) Jb[i].w=(Real)0; // used for loop terminations

        apply(InitialTransform);
    }

    OutCoord apply( const VecInCoord& in )  // Called in Apply
    {
        OutCoord res;

        unsigned int i,j,k,kk;

        DQCoord q;		// frame current position in DQ form
        b.clear();
        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            // weighted relative transform : w_i.q_i*q0_i^-1
            q.getDual()= Jb[i].TE*q.getOrientation() + Jb[i].T0*q.getDual();
            q.getOrientation() = Jb[i].T0*q.getOrientation();
            b+=q;
        }

        bn=b; bn.normalize();

        res.getCenter() = bn.pointToParent( P0.getCenter() );
        bn.toRotationMatrix(R);
        res.getAffine()= R*P0.getAffine();

        Mat<4,4,Real> N0; Mat<4,4,Real> NE; // Real/Dual part of the normalization Jacobian : dbn = [N0,NE] db
        b.normalize_getJ( N0 , NE );

        Mat<3,4,Real> Q0; Mat<3,4,Real> QE; // Real/Dual part of the transformation Jacobian : dP = [Q0,QE] dbn
        bn.pointToParent_getJ( Q0 , QE , P0.getCenter() );

        Mat<3,4,Real> QN0 = Q0*N0 + QE*NE , QNE = QE*N0;
        Mat<4,3,Real> L0 , LE , TL0 , TLE , Fa;

        Mat<4,4,Real> H0,HE;
        bn.inverse().multRight_getJ( H0,HE);
        Mat33 dA;

        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            q.velocity_getJ ( L0 , LE );  // Real/Dual part of quaternion Jacobian : dq_i = [L0,LE] [Omega_i, dt_i]
            TLE  = Jb[i].TE * L0 + Jb[i].T0 * LE;
            TL0  = Jb[i].T0 * L0 ;

            // dP = QNTL [Omega_i, dt_i]
            Jb[i].Pa = QN0 * TL0 + QNE * TLE;
            Jb[i].Pt = QNE * TL0 ;

            // Omega = 2*H-(bn^-1)NTL0.Omega_i
            Fa = H0 * N0 * TL0 * (Real) 2;

            for ( j=0; j<3 ; j++ )
            {
                helper::Quater<Real> qut(Fa[0][j],Fa[1][j],Fa[2][j],Fa[3][j]);
                qut.toMatrix(dA);
                for ( k=0; k<3 ; k++ ) for ( kk=0; kk<3 ; kk++ ) Jb[i].Fa[3*k+kk][j]=dA[k][kk];
            }
        }

        return res;
    }

    OutDeriv mult( const VecInDeriv& in ) // Called in ApplyJ
    {
        OutDeriv res;

        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            res.getVCenter() += Jb[i].Pt * getLinear( in[index[i]] )  + Jb[i].Pa * getAngular(in[index[i]]);
            *reinterpret_cast< Vec<9,Real>* >(&res.getVAffine()[0][0]) += Jb[i].Fa * getAngular(in[index[i]]);
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            getLinear(res[index[i]]) +=  Jb[i].Pt.transposed() * d.getVCenter();
            getAngular(res[index[i]]) += Jb[i].Pa.transposed() * d.getVCenter();
            getAngular(res[index[i]]) += Jb[i].Fa.transposed() * (*reinterpret_cast<const Vec<9,Real>* >(&d.getVAffine()[0][0]));
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            InDeriv parentJacobianVec;

            getLinear(parentJacobianVec) +=  Jb[i].Pt.transposed() * childJacobianVec.getVCenter();
            getAngular(parentJacobianVec) += Jb[i].Pa.transposed() * childJacobianVec.getVCenter();
            getAngular(parentJacobianVec) += Jb[i].Fa.transposed() * (*reinterpret_cast<const Vec<9,Real>* >(&childJacobianVec.getVAffine()[0][0]));
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }



    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 3>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid -> Rigid
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class _Material, int nbRef>
struct DualQuatBlending<
        StdRigidTypes<3,typename _Material::Real>,
        Out, _Material, nbRef, 4
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdRigidTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<3,3, InReal> MaterialFrame;
    typedef Vec<3, MaterialFrame> MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<3,3,Real> Mat33;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;
    typedef typename In::Deriv InDeriv;
    typedef helper::DualQuatCoord3<InReal> DQCoord;


    struct JacobianBlock
    {
        /** Dual quat skinning: p = sum_i w_i q_i*q0_i^-1 / |sum_i w_i q_i*q0_i^-1| (p_0)
          */
        Real w;

        Mat<4,4,Real> T0; Mat<4,4,Real> TE; ///< Real/Dual part of blended quaternion Jacobian : db = [T0,TE] dq

        Mat<3,3,Real> Pa; ///< dp = Pa.Omega_i  : rotation part
        Mat<3,3,Real> Pt; ///< dp = Pt.dt_i : translation part

        Mat<3,3,Real> Fa; ///< Omega = Omega_n = 2 * dbn o bn^-1 = Fa.Omega_i  : rotation part
    };

    Vec<nbRef,unsigned int> index;
    OutCoord P0;	///< initial position
    DQCoord b;		///< linearly blended dual quaternions : b= sum_i w_i q_i*q0_i^-1
    DQCoord bn;		///< normalized dual quaternion : bn=b/|b|

    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& /*dw*/, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        P0 = InitialPos;
        unsigned int i = 0 ;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            Jb[i].w= w[i];
            DQCoord T0inv=InitialTransform[index[i]]; T0inv.invert();
            T0inv.multLeft_getJ(Jb[i].T0,Jb[i].TE);
            Jb[i].T0*=w[i]; Jb[i].TE*=w[i];
        }
        if ( i<nbRef ) Jb[i].w=(Real)0; // used for loop terminations

        apply(InitialTransform);
    }

    OutCoord apply( const VecInCoord& in )  // Called in Apply
    {
        OutCoord res;

        unsigned int i,k;

        DQCoord q;		// frame current position in DQ form
        b.clear();
        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            // weighted relative transform : w_i.q_i*q0_i^-1
            q.getDual()= Jb[i].TE*q.getOrientation() + Jb[i].T0*q.getDual();
            q.getOrientation() = Jb[i].T0*q.getOrientation();
            b+=q;
        }

        bn=b; bn.normalize();

        res.getCenter() = bn.pointToParent( P0.getCenter() );
        helper::Quater<Real> qbn(bn.getOrientation()[0],bn.getOrientation()[1],bn.getOrientation()[2],bn.getOrientation()[3]);
        res.getOrientation()=qbn*P0.getOrientation();

        Mat<4,4,Real> N0; Mat<4,4,Real> NE; // Real/Dual part of the normalization Jacobian : dbn = [N0,NE] db
        b.normalize_getJ( N0 , NE );

        Mat<3,4,Real> Q0; Mat<3,4,Real> QE; // Real/Dual part of the transformation Jacobian : dP = [Q0,QE] dbn
        bn.pointToParent_getJ( Q0 , QE , P0.getCenter() );

        Mat<3,4,Real> QN0 = Q0*N0 + QE*NE , QNE = QE*N0;
        Mat<4,3,Real> L0 , LE , TL0 , TLE , Fa;

        Mat<4,4,Real> H0,HE;
        bn.inverse().multRight_getJ( H0,HE);

        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            q.velocity_getJ ( L0 , LE );  // Real/Dual part of quaternion Jacobian : dq_i = [L0,LE] [Omega_i, dt_i]
            TLE  = Jb[i].TE * L0 + Jb[i].T0 * LE;
            TL0  = Jb[i].T0 * L0 ;

            // dP = QNTL [Omega_i, dt_i]
            Jb[i].Pa = QN0 * TL0 + QNE * TLE;
            Jb[i].Pt = QNE * TL0 ;

            // Omega = 2*H-(bn^-1)NTL0.Omega_i
            Fa = H0 * N0 * TL0 * (Real) 2;
            for ( k=0; k<3 ; k++ ) Jb[i].Fa[k] = Fa[k]; // skip last line (w=0)
        }

        return res;
    }

    OutDeriv mult( const VecInDeriv& in ) // Called in ApplyJ
    {
        OutDeriv res;

        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            res.getVCenter() += Jb[i].Pt * getLinear( in[index[i]] )  + Jb[i].Pa * getAngular(in[index[i]]);
            res.getAngular() += Jb[i].Fa * getAngular(in[index[i]]);
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        /* To derive this method, rewrite the product Jacobian * InDeriv as a matrix * Vec12 product, and apply the transpose of this matrix
          */
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            getLinear(res[index[i]]) +=  Jb[i].Pt.transposed() * d.getVCenter();
            getAngular(res[index[i]]) += Jb[i].Pa.transposed() * d.getVCenter();
            getAngular(res[index[i]]) += Jb[i].Fa.transposed() * d.getAngular();
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT to build a contraint equation on the independent DOF
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            InDeriv parentJacobianVec;

            getLinear(parentJacobianVec) +=  Jb[i].Pt.transposed() * childJacobianVec.getVCenter();
            getAngular(parentJacobianVec) += Jb[i].Pa.transposed() * childJacobianVec.getVCenter();
            getAngular(parentJacobianVec) += Jb[i].Fa.transposed() * childJacobianVec.getAngular();
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }



    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 4>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }
};



//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid->DeformationGradient first order
//////////////////////////////////////////////////////////////////////////////////

template<class Out, class  _Material, int nbRef>
struct DualQuatBlending<
        StdRigidTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,1
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdRigidTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<Out::material_dimensions,Out::material_dimensions, InReal> MaterialFrame;
    typedef Vec<Out::material_dimensions, MaterialFrame> MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<3,3,Real> Mat33;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;
    typedef typename In::Deriv InDeriv;
    typedef helper::DualQuatCoord3<InReal> DQCoord;

    struct JacobianBlock
    {
        /** Dual quat skinning: p = sum_i w_i q_i*q0_i^-1 / |sum_i w_i q_i*q0_i^-1| (p_0)
          */

        Real w;
        MaterialDeriv dw;

        Mat<4,4,Real> T0; Mat<4,4,Real> TE; ///< Real/Dual part of blended quaternion Jacobian : db = [T0,TE] dq

        Mat<3,3,Real> Pa; ///< dp = Pa.Omega_i  : affine part
        Mat<3,3,Real> Pt; ///< dp = Pt.dt_i : translation part

        Mat<9,3,Real> Fa; ///< dF = Fa.Omega_i  : affine part
        Mat<9,3,Real> Ft; ///< dF = Ft.dt_i : translation part
    };

    Vec<nbRef,unsigned int> index;
    SpatialCoord P0;	///< initial position
    DQCoord b;		///< linearly blended dual quaternions : b= sum_i w_i q_i*q0_i^-1
    DQCoord bn;		///< normalized dual quaternion : bn=b/|b|

    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  /*ddw*/)
    {
        index = Index;
        P0 = InitialPos.getCenter();
        unsigned int i = 0 ;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            Jb[i].w= w[i];
            DQCoord T0inv=InitialTransform[index[i]]; T0inv.invert();
            T0inv.multLeft_getJ(Jb[i].T0,Jb[i].TE);
            Jb[i].T0*=w[i]; Jb[i].TE*=w[i];
            Jb[i].dw=dw[i];
        }
        if ( i<nbRef ) Jb[i].w=(Real)0; // used for loop terminations

        apply(InitialTransform);
    }

    OutCoord apply( const VecInCoord& in )  // Called in Apply
    {
        OutCoord res;

        unsigned int i,j,k,kk;

        DQCoord q;		// frame current position in DQ form
        b.clear();
        Mat<4,3,Real> W0 , WE; // Real/Dual part of the blended quaternion spatial deriv : db = [W0,WE] dp
        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            // weighted relative transform : w_i.q_i*q0_i^-1
            q.getDual()= Jb[i].TE*q.getOrientation() + Jb[i].T0*q.getDual();
            q.getOrientation() = Jb[i].T0*q.getOrientation();
            b+=q;
            for ( j=0; j<4 ; j++ )
                for ( k=0; k<3 ; k++ )
                {
                    W0[j][k]= q.getOrientation()[j]*Jb[i].dw[k]/Jb[i].w;
                    WE[j][k]= q.getDual()[j]*Jb[i].dw[k]/Jb[i].w;
                }
        }

        bn=b; bn.normalize();
        Mat<4,4,Real> N0; Mat<4,4,Real> NE; // Real/Dual part of the normalization Jacobian : dbn = [N0,NE] db
        b.normalize_getJ( N0 , NE );

        res.getCenter() = bn.pointToParent( P0 );
        Mat<3,4,Real> Q0; Mat<3,4,Real> QE; // Real/Dual part of the transformation Jacobian : dP = [Q0,QE] dbn
        bn.pointToParent_getJ( Q0 , QE , P0 );

        Mat<3,4,Real> QN0 = Q0*N0 + QE*NE , QNE = QE*N0;
        bn.toRotationMatrix(res.getMaterialFrame());
        res.getMaterialFrame() += QN0*W0 + QNE*WE; // defgradient F = R + QNW
        Mat<4,3,Real> L0 , LE , TL0 , TLE;

        Mat<3,4,Real> QNT0 , QNTE ;
        Mat<4,3,Real> NTL0 , NTLE ;
        Mat<4,3,Real> NW0 = N0*W0 , NWE = NE*W0 + N0*WE;
        Mat<3,4,Real> dQ0 , dQE;
        Mat<4,4,Real> dN0 , dNE;
        Mat<4,3,Real> dW0 , dWE;
        DQCoord dq ;
        Mat<3,3,Real> dF;

        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            q=in[index[i]];
            q.velocity_getJ ( L0 , LE );  // Real/Dual part of quaternion Jacobian : dq_i = [L0,LE] [Omega_i, dt_i]
            TLE  = Jb[i].TE * L0 + Jb[i].T0 * LE;
            TL0  = Jb[i].T0 * L0 ;
            // dP = QNTL [Omega_i, dt_i]
            Jb[i].Pa = QN0 * TL0 + QNE * TLE;
            Jb[i].Pt = QNE * TL0 ;

            NTL0 = N0 * TL0 ;				NTLE = NE * TL0 + N0 * TLE;
            QNT0 = QN0 * Jb[i].T0 + QNE * Jb[i].TE;		QNTE = QNE * Jb[i].T0 ;
            QNT0/=Jb[i].w; QNTE/=Jb[i].w; // remove pre-multiplication of T by w

            // dF = dR + dQNW + QdNW + QNdW
            Jb[i].Fa.fill(0);
            Jb[i].Ft.fill(0);
            for ( j=0; j<3 ; j++ )
            {
                // dR
                for ( k=0; k<4 ; k++ )  {dq.getOrientation()[k]=NTL0[k][j]; dq.getDual()[k]=NTLE[k][j];}
                dF=bn.rotation_applyH(dq); for ( k=0; k<9 ; k++ ) Jb[i].Fa[k][j]+=MattoVec(dF)[k];

                // dQNW
                dq.pointToParent_getJ( dQ0 , dQE , P0 );
                dF = dQ0*NW0 + dQE*NWE ;  for ( k=0; k<9 ; k++ ) Jb[i].Fa[k][j]+=MattoVec(dF)[k];
                dq.getDual()=dq.getOrientation(); dq.getOrientation().fill(0);
                dq.pointToParent_getJ( dQ0 , dQE , P0 );
                dF = dQ0*NW0 ;  for ( k=0; k<9 ; k++ ) Jb[i].Ft[k][j]+=MattoVec(dF)[k];

                // QdNW
                for ( k=0; k<4 ; k++ )  {dq.getOrientation()[k]=TL0[k][j]; dq.getDual()[k]=TLE[k][j];}
                b.normalize_getdJ ( dN0 , dNE , dq );
                dF = Q0*dN0*W0 + QE*(dNE*W0 + dN0*WE);  for ( k=0; k<9 ; k++ ) Jb[i].Fa[k][j]+=MattoVec(dF)[k];
                dq.getDual()=dq.getOrientation(); dq.getOrientation().fill(0);
                b.normalize_getdJ ( dN0 , dNE , dq );
                dF = QE*dNE*W0 ;  for ( k=0; k<9 ; k++ ) Jb[i].Ft[k][j]+=MattoVec(dF)[k];

                // QNdW
                for ( k=0; k<4 ; k++ )
                    for ( kk=0; kk<3 ; kk++ )
                    {
                        dW0[k][kk]= L0[k][j]*Jb[i].dw[kk];
                        dWE[k][kk]= LE[k][j]*Jb[i].dw[kk];
                    }
                dF = QNT0*dW0 + QNTE*dWE;  for ( k=0; k<9 ; k++ ) Jb[i].Fa[k][j]+=MattoVec(dF)[k];
                dF = QNTE*dW0;  for ( k=0; k<9 ; k++ ) Jb[i].Ft[k][j]+=MattoVec(dF)[k];
            }
        }

        return res;
    }


    OutDeriv mult( const VecInDeriv& in ) // Called in ApplyJ
    {
        OutDeriv res;

        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            res.getCenter() += Jb[i].Pt * getLinear( in[index[i]] )  + Jb[i].Pa * getAngular(in[index[i]]);
            MattoVec(res.getMaterialFrame()) += Jb[i].Ft * getLinear( in[index[i]] )  + Jb[i].Fa * getAngular(in[index[i]]);
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            getLinear(res[index[i]])  += Jb[i].Pt.transposed() * d.getCenter();
            getAngular(res[index[i]]) += Jb[i].Pa.transposed() * d.getCenter();
            getLinear(res[index[i]])  += Jb[i].Ft.transposed() * MattoVec(d.getMaterialFrame());
            getAngular(res[index[i]]) += Jb[i].Fa.transposed() * MattoVec(d.getMaterialFrame());
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0.; i++ )
        {

            InDeriv parentJacobianVec;
            getLinear(parentJacobianVec)  += Jb[i].Pt.transposed() * childJacobianVec.getCenter();
            getAngular(parentJacobianVec) += Jb[i].Pa.transposed() * childJacobianVec.getCenter();
            getLinear(parentJacobianVec)  += Jb[i].Ft.transposed() * MattoVec(childJacobianVec.getMaterialFrame());
            getAngular(parentJacobianVec) += Jb[i].Fa.transposed() * MattoVec(childJacobianVec.getMaterialFrame());
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 1>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};


//////////////////////////////////////////////////////////////////////////////////
////  Specialization on Rigid->DeformationGradient second order
//////////////////////////////////////////////////////////////////////////////////


template<class Out, class  _Material, int nbRef>
struct DualQuatBlending<
        StdRigidTypes<3,typename _Material::Real>,
        Out, _Material, nbRef,2
        >
{
    typedef _Material Material;
    typedef typename Material::Real InReal;
    typedef typename Out::Real OutReal;
    typedef InReal Real;
    typedef typename Material::SGradient MaterialDeriv;
    typedef typename Material::SHessian MaterialMat;
    typedef StdRigidTypes<3,InReal> In;
    typedef Vec<Out::spatial_dimensions, InReal> SpatialCoord; // = Vec3
    typedef Mat<Out::material_dimensions,Out::material_dimensions, InReal> MaterialFrame;
    typedef Vec<Out::material_dimensions, MaterialFrame> MaterialFrameGradient;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef Mat<3,3,Real> Mat33;
    typedef typename In::MatrixDeriv::RowIterator ParentJacobianRow;
    typedef typename In::Deriv InDeriv;
    typedef helper::DualQuatCoord3<InReal> DQCoord;

    struct JacobianBlock
    {
        /** Dual quat skinning: p = sum_i w_i q_i*q0_i^-1 / |sum_i w_i q_i*q0_i^-1| (p_0)
          */

        Real w;
        MaterialDeriv dw;
        MaterialMat ddw;

        Mat<4,4,Real> T0; Mat<4,4,Real> TE; ///< Real/Dual part of blended quaternion Jacobian : db = [T0,TE] dq

        Mat<3,3,Real> Pa; ///< dp = Pa.Omega_i  : affine part
        Mat<3,3,Real> Pt; ///< dp = Pt.dt_i : translation part

        Mat<9,3,Real> Fa; ///< dF = Fa.Omega_i  : affine part
        Mat<9,3,Real> Ft; ///< dF = Ft.dt_i : translation part

        Vec<3,Mat<9,3,Real> > dFa; ///< d gradF_k = dFa_k.Omega_i  : affine part
        Vec<3,Mat<9,3,Real> > dFt; ///< d gradF_k = dFt_k.dt_i : translation part
    };

    Vec<nbRef,unsigned int> index;
    SpatialCoord P0;	///< initial position
    DQCoord b;		///< linearly blended dual quaternions : b= sum_i w_i q_i*q0_i^-1
    DQCoord bn;		///< normalized dual quaternion : bn=b/|b|

    Vec<nbRef,JacobianBlock> Jb;

    void init( const OutCoord& InitialPos, const Vec<nbRef,unsigned int>& Index, const VecInCoord& InitialTransform, const Vec<nbRef,Real>& w, const Vec<nbRef,MaterialDeriv>& dw, const Vec<nbRef,MaterialMat>&  ddw)
    {
        index = Index;
        P0 = InitialPos.getCenter();
        unsigned int i = 0 ;
        for ( ; i<nbRef && w[i]>0; i++ )
        {
            Jb[i].w= w[i];
            DQCoord T0inv=InitialTransform[index[i]]; T0inv.invert();
            T0inv.multLeft_getJ(Jb[i].T0,Jb[i].TE);
            Jb[i].T0*=w[i]; Jb[i].TE*=w[i];
            Jb[i].dw=dw[i];
            Jb[i].ddw=ddw[i];
        }
        if ( i<nbRef ) Jb[i].w=(Real)0; // used for loop terminations
        apply(InitialTransform);
    }






    OutCoord apply( const Vec<nbRef,DQCoord>& in )
    {
        OutCoord res;

        unsigned int i,j,k,kk;

        DQCoord q,b2,bn2;
        b2.clear();
        Mat<4,3,Real> W0 , WE; // Real/Dual part of the blended quaternion spatial deriv : db = [W0,WE] dp
        /// specific to D332
        Vec<3,Mat<4,3,Real> > gradW0 , gradWE; // spatial deriv of W
        ///
        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {
            // weighted relative transform : w_i.q_i*q0_i^-1
            q=in[i];
            q.getDual()= Jb[i].TE*q.getOrientation() + Jb[i].T0*q.getDual();
            q.getOrientation() = Jb[i].T0*q.getOrientation();
            b2+=q;
            for ( j=0; j<4 ; j++ )
                for ( k=0; k<3 ; k++ )
                {
                    W0[j][k]= q.getOrientation()[j]*Jb[i].dw[k]/Jb[i].w;
                    WE[j][k]= q.getDual()[j]*Jb[i].dw[k]/Jb[i].w;
                    /// specific to D332
                    for ( kk=0; kk<3 ; kk++ )
                    {
                        gradW0[kk][j][k]= q.getOrientation()[j]*Jb[i].ddw[k][kk]/Jb[i].w;
                        gradWE[kk][j][k]= q.getDual()[j]*Jb[i].ddw[k][kk]/Jb[i].w;
                    }
                    ///
                }
        }

        bn2=b2; bn2.normalize();
        Mat<4,4,Real> N0; Mat<4,4,Real> NE; // Real/Dual part of the normalization Jacobian : dbn = [N0,NE] db
        b2.normalize_getJ( N0 , NE );
        res.getCenter() = bn2.pointToParent( P0 );
        Mat<3,4,Real> Q0; Mat<3,4,Real> QE; // Real/Dual part of the transformation Jacobian : dP = [Q0,QE] dbn
        bn2.pointToParent_getJ( Q0 , QE , P0 );
        Mat<3,4,Real> QN0 = Q0*N0 + QE*NE , QNE = QE*N0;
        bn2.toRotationMatrix(res.getMaterialFrame());
        res.getMaterialFrame() += QN0*W0 + QNE*WE; // defgradient F = R + QNW

        Mat<4,3,Real> NW0 = N0*W0 , NWE = NE*W0 + N0*WE;
        Mat<3,4,Real> dQ0 , dQE;
        Mat<4,4,Real> dN0 , dNE;
        Mat<3,3,Real> dF;
        DQCoord dq ;

        /// specific to D332
        // gradF = gradR + gradQ.N.W + Q.gradN.W + Q.N.gradW
        for ( j=0; j<3 ; j++ )
        {
            res.getMaterialFrameGradient()[j].fill(0);
            // gradR
            for ( k=0; k<4 ; k++ )  {dq.getOrientation()[k]=NW0[k][j]; dq.getDual()[k]=NWE[k][j];}
            dF=bn2.rotation_applyH(dq); res.getMaterialFrameGradient()[j]+=dF;
            // gradQ.N.W
            dq.pointToParent_getJ( dQ0 , dQE , P0 );
            dF = dQ0*NW0 + dQE*NWE ;  res.getMaterialFrameGradient()[j]+=dF;
            // Q.gradN.W
            for ( k=0; k<4 ; k++ )  {dq.getOrientation()[k]=W0[k][j]; dq.getDual()[k]=WE[k][j];}
            b2.normalize_getdJ ( dN0 , dNE , dq );
            dF = Q0*dN0*W0 + QE*(dNE*W0 + dN0*WE);  res.getMaterialFrameGradient()[j]+=dF;
            // Q.N.gradW
            dF = QN0*gradW0[j] + QNE*gradWE[j];  res.getMaterialFrameGradient()[j]+=dF;
        }

        return res;
    }




    OutCoord apply( const VecInCoord& in )  // Called in Apply
    {
        OutCoord res;

        unsigned int i,j,k,kk;

        Vec<nbRef,DQCoord> DQin;
        for ( i=0; i<nbRef && Jb[i].w>0.; i++ ) DQin[i]=in[index[i]];
        res=apply( DQin );

        Mat<4,3,Real> L0 , LE ;
        Mat<3,3,Real> dF;
        DQCoord dq;
        OutCoord res2;
        Real mult=(Real)0.000001;

        // here we do not compute the Jacobian dF/dq_i as in D331 (it is too complex for gradF)
        // so we estimate dF/dq_i ~ [ F(q_i+mult*dq_i)- F(q_i) ] /mult

        for ( i=0; i<nbRef && Jb[i].w>0.; i++ )
        {

            DQin[i].velocity_getJ ( L0 , LE );  // Real/Dual part of quaternion Jacobian : dq_i = [L0,LE] [Omega_i, dt_i]

            for ( j=0; j<3 ; j++ )
            {

                for ( k=0; k<4 ; k++ )  {dq.getOrientation()[k]=L0[k][j]; dq.getDual()[k]=LE[k][j];}
                DQin[i]+=dq*mult;

                res2=apply( DQin );

                for ( k=0; k<3 ; k++ ) Jb[i].Pa[k][j]=(res2.getCenter()[k]-res.getCenter()[k])/mult;
                dF=(res2.getMaterialFrame()-res.getMaterialFrame())/mult;
                for ( k=0; k<9 ; k++ ) Jb[i].Fa[k][j]=MattoVec(dF)[k];

                for ( kk=0; kk<3 ; kk++ )
                {
                    dF=(res2.getMaterialFrameGradient()[kk]-res.getMaterialFrameGradient()[kk])/mult;
                    for ( k=0; k<9 ; k++ ) Jb[i].dFa[kk][k][j]=MattoVec(dF)[k];
                }

                DQin[i]+=dq*(-mult);

                DQin[i].getDual()+=dq.getOrientation()*mult;

                res2=apply( DQin );

                for ( k=0; k<3 ; k++ ) Jb[i].Pt[k][j]=(res2.getCenter()[k]-res.getCenter()[k])/mult;
                dF=(res2.getMaterialFrame()-res.getMaterialFrame())/mult;
                for ( k=0; k<9 ; k++ ) Jb[i].Ft[k][j]=MattoVec(dF)[k];

                for ( kk=0; kk<3 ; kk++ )
                {
                    dF=(res2.getMaterialFrameGradient()[kk]-res.getMaterialFrameGradient()[kk])/mult;
                    for ( k=0; k<9 ; k++ ) Jb[i].dFt[kk][k][j]=MattoVec(dF)[k];
                }

                DQin[i].getDual()+=dq.getOrientation()*(-mult);

            }
        }

        return res;
    }


    OutDeriv mult( const VecInDeriv& in ) // Called in ApplyJ
    {
        OutDeriv res;

        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            res.getCenter() += Jb[i].Pt * getLinear( in[index[i]] )  + Jb[i].Pa * getAngular(in[index[i]]);
            MattoVec(res.getMaterialFrame()) += Jb[i].Ft * getLinear( in[index[i]] )  + Jb[i].Fa * getAngular(in[index[i]]);
            for (unsigned int k = 0; k < 3; ++k)
                MattoVec(res.getMaterialFrameGradient()[k]) += Jb[i].dFt[k] * getLinear( in[index[i]] )  + Jb[i].dFa[k] * getAngular(in[index[i]]);
        }
        return res;
    }

    void addMultTranspose( VecInDeriv& res, const OutDeriv& d ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0; i++ )
        {
            getLinear(res[index[i]])  += Jb[i].Pt.transposed() * d.getCenter();
            getAngular(res[index[i]]) += Jb[i].Pa.transposed() * d.getCenter();
            getLinear(res[index[i]])  += Jb[i].Ft.transposed() * MattoVec(d.getMaterialFrame());
            getAngular(res[index[i]]) += Jb[i].Fa.transposed() * MattoVec(d.getMaterialFrame());
            for (unsigned int k = 0; k < 3; ++k)
            {
                getLinear(res[index[i]])  += Jb[i].dFt[k].transposed() * MattoVec(d.getMaterialFrameGradient()[k]);
                getAngular(res[index[i]]) += Jb[i].dFa[k].transposed() * MattoVec(d.getMaterialFrameGradient()[k]);
            }
        }
    }

    void addMultTranspose( ParentJacobianRow& parentJacobianRow, const OutDeriv& childJacobianVec ) // Called in ApplyJT
    {
        for ( unsigned int i=0; i<nbRef && Jb[i].w>0.; i++ )
        {

            InDeriv parentJacobianVec;
            getLinear(parentJacobianVec)  += Jb[i].Pt.transposed() * childJacobianVec.getCenter();
            getAngular(parentJacobianVec) += Jb[i].Pa.transposed() * childJacobianVec.getCenter();
            getLinear(parentJacobianVec)  += Jb[i].Ft.transposed() * MattoVec(childJacobianVec.getMaterialFrame());
            getAngular(parentJacobianVec) += Jb[i].Fa.transposed() * MattoVec(childJacobianVec.getMaterialFrame());
            for (unsigned int k = 0; k < 3; ++k)
            {
                getLinear(parentJacobianVec)  += Jb[i].dFt[k].transposed() * MattoVec(childJacobianVec.getMaterialFrameGradient()[k]);
                getAngular(parentJacobianVec) += Jb[i].dFa[k].transposed() * MattoVec(childJacobianVec.getMaterialFrameGradient()[k]);
            }
            parentJacobianRow.addCol(index[i],parentJacobianVec);
        }
    }
    inline friend std::ostream& operator<< ( std::ostream& o, const DualQuatBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return o;
    }

    inline friend std::istream& operator>> ( std::istream& i, DualQuatBlending<StdRigidTypes<3,typename _Material::Real>,Out, _Material, nbRef, 2>& /*e*/ )
    {
        // Not implemented !!  Just needed to compile
        return i;
    }

};

//////////////////////////////////////////////////////////////////////////////////
}
}

#endif
