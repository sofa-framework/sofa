/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef FLEXIBLE_BaseJacobian_H
#define FLEXIBLE_BaseJacobian_H

#include <sofa/defaulttype/Mat.h>
#include <Eigen/Core>
#include <Eigen/Dense>

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
    static const bool constantJ = false; ///< tells if the jacobian is constant (to avoid recomputations)

    // Called in Apply
    virtual void addapply( OutCoord& result, const InCoord& data )=0;
    // Called in ApplyJ
    virtual void addmult( OutDeriv& result,const InDeriv& data )=0;
    // Called in ApplyJT
    virtual void addMultTranspose( InDeriv& result, const OutDeriv& data )=0;
    // Called in getJ
    virtual MatBlock getJ()=0;

    // Geometric Stiffness = dJ^T.fc
    virtual KBlock getK(const OutDeriv& childForce)=0;
    // compute $ df += K dx $
    virtual void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const double& kfactor )=0;

protected:



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






} // namespace defaulttype
} // namespace sofa



#endif
