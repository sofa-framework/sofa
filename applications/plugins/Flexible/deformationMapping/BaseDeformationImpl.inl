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
#ifndef SOFA_COMPONENT_MAPPING_BaseDeformationImpl_INL
#define SOFA_COMPONENT_MAPPING_BaseDeformationImpl_INL

// NB: These implementations have been factored from BaseDeformationMapping.inl and BaseDeformationMultiMapping.inl
// so that LinearMapping.h and LinearMultiMapping.h can be included together.

namespace sofa
{

/** determinant for 1x1 matrix  to complement 2x2 and 3x3 implementations (used for visualization of  det F ) **/
namespace defaulttype { template<class real> inline real determinant(const Mat<1,1,real>& m) { return m(0,0);} }

namespace component
{
namespace mapping
{

template<int matdim,typename Real>
void drawEllipsoid(const defaulttype::Mat<3,matdim,Real> & F, const defaulttype::Vec<3,Real> &p, const float& scale)
{
#ifndef SOFA_NO_OPENGL
    glPushMatrix();

    GLdouble transformMatrix[16];
    for(size_t i=0; i<3; i++) for(size_t j=0; j<matdim; j++) transformMatrix[4*j+i] = (double)F(i,j)*scale;

    if(matdim==1)
    {
        for(size_t i=0; i<3; i++) for(size_t j=1; j<3; j++) transformMatrix[4*j+i] = 0;
    }
    else if(matdim==2)
    {
        defaulttype::Vec<3,Real> w=cross(F.transposed()[0],F.transposed()[1]); w.normalize();
        for(size_t i=0; i<3; i++)  transformMatrix[8+i]=(double)w[i]*scale*0.01; // arbitrarily small thickness
    }

    for(size_t i=0; i<3; i++)  transformMatrix[i+12]=p[i];
    for(size_t i=0; i<3; i++)  transformMatrix[4*i+3]=0; transformMatrix[15] = 1;
    glMultMatrixd(transformMatrix);

    GLUquadricObj* ellipsoid = gluNewQuadric();
    gluSphere(ellipsoid, 1.0, 10, 10);
    gluDeleteQuadric(ellipsoid);

    glPopMatrix();
#endif
}

/** inversion of rectangular deformation gradients (used in backward mapping) **/
template <int L,typename Real>
inline static void invert(defaulttype::Mat<L,L,Real> &Minv, const defaulttype::Mat<L,L,Real> &M)
{
    //    Eigen::Map<const Eigen::Matrix<Real,L,L,Eigen::RowMajor> >  eM(&M[0][0]);
    //    Eigen::Map<Eigen::Matrix<Real,L,L,Eigen::RowMajor> >  eMinv(&Minv[0][0]);
    //    eMinv=eM.inverse();
    Minv.invert(M);
}

template <int L,typename Real>
inline static void invert(defaulttype::Mat<1,L,Real> &Minv, const defaulttype::Mat<L,1,Real> &M)
{
    Real n2inv=0; for(size_t i=0; i<L; i++) n2inv+=M[i][0]*M[i][0];
    n2inv=1./n2inv;
    for(size_t i=0; i<L; i++)  Minv[0][i]=M[i][0]*n2inv;
}

template <typename Real>
inline static void invert(defaulttype::Mat<2,3,Real> &Minv, const defaulttype::Mat<3,2,Real> &M)
{
    defaulttype::Vec<3,Real> u=M.transposed()[0],v=M.transposed()[1],w=cross(u,v);
    w.normalize();
    defaulttype::Mat<3,3,Real> Mc; for(size_t i=0; i<3; i++) {Mc[i][0]=M[i][0]; Mc[i][1]=M[i][1]; Mc[i][2]=w[i];}
    defaulttype::Mat<3,3,Real> Mcinv; invert(Mcinv,Mc);
    for(size_t i=0; i<2; i++) for(size_t j=0; j<3; j++) Minv[i][j]=Mcinv[i][0];
}



template <typename Mat>
inline static Mat identity()
{
    Mat F;
    if(Mat::nbLines>=Mat::nbCols) for(int i=0; i<Mat::nbCols; i++) F[i][i]=1.0;
    else for(int i=0; i<Mat::nbLines; i++) F[i][i]=1.0;
    return F;
}

template <int C,int L,typename Real>
inline static void identity(defaulttype::Mat<C,L,Real> &F)
{
    F.clear();
    if(L>=C) for(size_t i=0; i<C; i++) F[i][i]=1.0;
    else for(size_t i=0; i<L; i++) F[i][i]=1.0;
}

template <int C1,int L1,int C2,int L2,typename Real>
inline static void copy(defaulttype::Mat<C1,L1,Real> &F1, const defaulttype::Mat<C2,L2,Real> &F2)
{
    F1.clear();
    for(size_t c=0; c<C1 && c<C2; c++)
        for(size_t l=0; l<L1 && l<L2; l++)
            F1[c][l]=F2[c][l];
}

} // namespace mapping
} // namespace component
} // namespace sofa

#endif
