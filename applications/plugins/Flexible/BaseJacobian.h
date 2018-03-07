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
#ifndef FLEXIBLE_BaseJacobian_H
#define FLEXIBLE_BaseJacobian_H

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/MatSym.h>
#include <Eigen/Core>
//#include <Eigen/Dense>

namespace sofa
{

namespace defaulttype
{


/** Template class used to implement one jacobian block
*/
template<class TIn, class TOut>
class BaseJacobianBlock
{
public:
    typedef TIn In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Real Real;

    typedef TOut Out;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;

    typedef Mat<Out::deriv_total_size,In::deriv_total_size,Real> MatBlock;
    typedef Mat<In::deriv_total_size,In::deriv_total_size,Real> KBlock;

    // Called in Apply
    virtual void addapply( OutCoord& result, const InCoord& data )=0;
    // Called in ApplyJ
    virtual void addmult( OutDeriv& result,const InDeriv& data )=0;
    // Called in ApplyJT
    virtual void addMultTranspose( InDeriv& result, const OutDeriv& data )=0;
    // Called in getJ
    virtual MatBlock getJ()=0;

    // Geometric Stiffness = dJ^T.fc
    virtual KBlock getK(const OutDeriv& childForce, bool stabilization=false)=0;
    // compute $ df += K dx $
    virtual void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const SReal& kfactor )=0;

protected:


    //////////////////////////////////////////////////////////////////////////////////
    ////  macros
    //////////////////////////////////////////////////////////////////////////////////
    #define V3(type) StdVectorTypes<Vec<3,type>,Vec<3,type>,type>
    #define V2(type) StdVectorTypes<Vec<2,type>,Vec<2,type>,type>
    #define EV3(type) ExtVectorTypes<Vec<3,type>,Vec<3,type>,type>

    #define Rigid3(type)  StdRigidTypes<3,type>
    #define Affine3(type)  StdAffineTypes<3,type>
    #define Quadratic3(type)  StdQuadraticTypes<3,type>

    #define F331(type)  DefGradientTypes<3,3,0,type>
    #define F321(type)  DefGradientTypes<3,2,0,type>
    #define F311(type)  DefGradientTypes<3,1,0,type>
    #define F332(type)  DefGradientTypes<3,3,1,type>
    #define F221(type)  DefGradientTypes<2,2,0,type>

    #define E321(type)  StrainTypes<3,2,0,type>
    #define E311(type)  StrainTypes<3,1,0,type>
    #define E221(type)  StrainTypes<2,2,0,type>

    #define I331(type)  InvariantStrainTypes<3,3,0,type>

    #define U331(type)  PrincipalStretchesStrainTypes<3,3,0,type>
    #define U321(type)  PrincipalStretchesStrainTypes<3,2,0,type>


    //////////////////////////////////////////////////////////////////////////////////
    ////  helpers
    //////////////////////////////////////////////////////////////////////////////////


    template<typename Real>
    static Eigen::Matrix<Real,6,9,Eigen::RowMajor> assembleJ(const defaulttype::Mat<3,3,Real>& f) // 3D->3D
    {
        static const unsigned int spatial_dimensions = 3;
        static const unsigned int material_dimensions = 3;
        static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2;
        typedef Eigen::Matrix<Real,strain_size,spatial_dimensions*material_dimensions,Eigen::RowMajor> JBlock;
        JBlock J=JBlock::Zero();
        for( unsigned int k=0; k<spatial_dimensions; k++ )
            for(unsigned int j=0; j<material_dimensions; j++)
                J(j,j+material_dimensions*k)=f[k][j];
        for( unsigned int k=0; k<spatial_dimensions; k++ )
        {
            J(3,material_dimensions*k+1)=J(5,material_dimensions*k+2)=f[k][0];
            J(3,material_dimensions*k)=J(4,material_dimensions*k+2)=f[k][1];
            J(5,material_dimensions*k)=J(4,material_dimensions*k+1)=f[k][2];
        }
        return J;
    }

    template<typename Real>
    static Eigen::Matrix<Real,3,4,Eigen::RowMajor> assembleJ(const defaulttype::Mat<2,2,Real>& f) // 2D->2D
    {
        static const unsigned int spatial_dimensions = 2;
        static const unsigned int material_dimensions = 2;
        static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2;
        typedef Eigen::Matrix<Real,strain_size,spatial_dimensions*material_dimensions,Eigen::RowMajor> JBlock;
        JBlock J=JBlock::Zero();
        for( unsigned int k=0; k<spatial_dimensions; k++ )
            for(unsigned int j=0; j<material_dimensions; j++)
                J(j,j+material_dimensions*k)=f[k][j];
        for( unsigned int k=0; k<spatial_dimensions; k++ )
        {
            J(material_dimensions,material_dimensions*k+1)=f[k][0];
            J(material_dimensions,material_dimensions*k)=f[k][1];
        }
        return J;
    }

    template<typename Real>
    static Eigen::Matrix<Real,3,6,Eigen::RowMajor> assembleJ(const defaulttype::Mat<3,2,Real>& f) // 3D->2D
    {
        static const unsigned int spatial_dimensions = 3;
        static const unsigned int material_dimensions = 2;
        static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2;
        typedef Eigen::Matrix<Real,strain_size,spatial_dimensions*material_dimensions,Eigen::RowMajor> JBlock;
        JBlock J=JBlock::Zero();
        for( unsigned int k=0; k<spatial_dimensions; k++ )
            for(unsigned int j=0; j<material_dimensions; j++)
                J(j,j+material_dimensions*k)=f[k][j];
        for( unsigned int k=0; k<spatial_dimensions; k++ )
        {
            J(material_dimensions,material_dimensions*k+1)=f[k][0];
            J(material_dimensions,material_dimensions*k)=f[k][1];
        }
        return J;
    }

    template<typename Real>
    static Eigen::Matrix<Real,1,3,Eigen::RowMajor> assembleJ(const defaulttype::Mat<3,1,Real>& f) // 3D->1D
    {
        static const unsigned int spatial_dimensions = 3;
        static const unsigned int material_dimensions = 1;
        static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2;
        typedef Eigen::Matrix<Real,strain_size,spatial_dimensions*material_dimensions,Eigen::RowMajor> JBlock;
        JBlock J=JBlock::Zero();
        for( unsigned int k=0; k<spatial_dimensions; k++ )
            for(unsigned int j=0; j<material_dimensions; j++)
                J(j,j+material_dimensions*k)=f[k][j];
        return J;
    }
};



template<class Real1, class Real2,  int Dim1, int Dim2>
inline Mat<Dim1, Dim2, Real2> covMN(const Vec<Dim1,Real1>& v1, const Vec<Dim2,Real2>& v2)
{
    Mat<Dim1, Dim2, Real2> res;
    for ( unsigned int i = 0; i < Dim1; ++i)
        for ( unsigned int j = 0; j < Dim2; ++j)
        {
            res[i][j] = (Real2)v1[i] * v2[j];
        }
    return res;
}

template<class Real,  int Dim>
inline MatSym<Dim, Real> covN(const Vec<Dim,Real>& v)
{
    MatSym<Dim, Real> res;
    for ( unsigned int i = 0; i < Dim; ++i)
        for ( unsigned int j = i; j < Dim; ++j)
        {
            res(i,j) = v[i] * v[j];
        }
    return res;
}



} // namespace defaulttype
} // namespace sofa



#endif
