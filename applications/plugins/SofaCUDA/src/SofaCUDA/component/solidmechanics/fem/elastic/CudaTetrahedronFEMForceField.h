/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>


namespace sofa::gpu::cuda
{

template<class DataTypes>
class CudaKernelsTetrahedronFEMForceField;

} // namespace sofa::gpu::cuda


namespace sofa::component::solidmechanics::fem::elastic
{

template <class TCoord, class TDeriv, class TReal>
class TetrahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef TetrahedronFEMForceField<DataTypes> Main;
    typedef TetrahedronFEMForceFieldInternalData<DataTypes> Data;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef typename Main::Element Element;
    typedef typename Main::VecElement VecElement;
    typedef type::Mat<6, 6, Real> MaterialStiffness;
    typedef type::Mat<12, 6, Real> StrainDisplacement;

    typedef gpu::cuda::CudaKernelsTetrahedronFEMForceField<DataTypes> Kernels;

    /** Static data associated with each element

    Only supports the LARGE computation method, allowing for several simplifications.

    For each element ABCD, the original algorithm computes the \f$3 \times 3\f$ rotation matrix R as follow:

    - the X axis is aligned to the AB edge
    - the Z axis is aligned to the cross-product between AB and AC
    - the Y axis is aligned to the cross-product between the previous two edges

    This matrix is used to compute the initial position of each element in the local rotated coordinate frame, with the origin on A, as follow:
    \f{eqnarray*}
        a &=& R^t \cdot (A-A) = \left( 0   \quad 0   \quad 0   \right) \\
        b &=& R^t \cdot (B-A) = \left( b_x \quad 0   \quad 0   \right) \\
        c &=& R^t \cdot (C-A) = \left( c_x \quad c_y \quad 0   \right) \\
        d &=& R^t \cdot (D-A) = \left( d_x \quad d_y \quad d_z \right) \\
    \f}

    The material stiffness is handled by a \f$6 \times 6\f$ matrix K computed as follow:

    \f{eqnarray*}
        K &=& \begin{pmatrix}
            \gamma+\mu^2 & \gamma & \gamma & 0 & 0 & 0 \\
            \gamma & \gamma+\mu^2 & \gamma & 0 & 0 & 0 \\
            \gamma & \gamma & \gamma+\mu^2 & 0 & 0 & 0 \\
            0 & 0 & 0 & \mu^2/2 & 0 & 0 \\
            0 & 0 & 0 & 0 & \mu^2/2 & 0 \\
            0 & 0 & 0 & 0 & 0 & \mu^2/2 \\
            \end{pmatrix} \\
        \gamma &=& \frac{youngModulus}{6 vol(ABCD)}\frac{poissonRatio}{(1 + poissonRatio)(1 - 2 poissonRatio)} \\
        \mu^2 &=& \frac{youngModulus}{6 vol(ABCD)}\frac{1}{1+poissonRatio}
    \f}

    The \f$12 \times 6\f$ Strain-displacement matrix is computed from the rotated initial positions abcd as follow:

    \f{eqnarray*}
        J &=& \begin{pmatrix}
            -p_x(bcd) & 0 & 0 & -p_y(bcd) & 0 & -p_z(bcd) \\
            0 & -p_y(bcd) & 0 & -p_x(bcd) & -p_z(bcd) & 0 \\
            0 & 0 & -p_z(bcd) & 0 & -p_y(bcd) & -p_x(bcd) \\
            p_x(cda) & 0 & 0 & p_y(cda) & 0 & p_z(cda) \\
            0 & p_y(cda) & 0 & p_x(cda) & p_z(cda) & 0 \\
            0 & 0 & p_z(cda) & 0 & p_y(cda) & p_x(cda) \\
            -p_x(dab) & 0 & 0 & -p_y(dab) & 0 & -p_z(dab) \\
            0 & -p_y(dab) & 0 & -p_x(dab) & -p_z(dab) & 0 \\
            0 & 0 & -p_z(dab) & 0 & -p_y(dab) & -p_x(abc) \\
            p_x(abc) & 0 & 0 & p_y(abc) & 0 & p_z(abc) \\
            0 & p_y(abc) & 0 & p_x(abc) & p_z(abc) & 0 \\
            0 & 0 & p_z(abc) & 0 & p_y(abc) & p_x(abc) \\
            \end{pmatrix} \\
        p(uvw) &=& u \times v + v \times w + w \times u
    \f}

    Given the zeros in abcd, we have:
    \f{eqnarray*}
        Jb &=& p(cda) = c \times d + d \times a + a \times c = \begin{pmatrix}c_x \\ c_y \\ 0\end{pmatrix} \times \begin{pmatrix}d_x \\ d_y \\ d_z\end{pmatrix} = \begin{pmatrix}c_y d_z \\ - c_x d_z \\ c_x d_y - c_y d_x\end{pmatrix} \\
        Jc &=& -p(dab) = - d \times a - a \times b - b \times d = - \begin{pmatrix}b_x \\ 0 \\ 0\end{pmatrix} \times \begin{pmatrix}d_x \\ d_y \\ d_z\end{pmatrix} = \begin{pmatrix}0 \\ b_x d_z \\ - b_x d_y\end{pmatrix} \\
        Jd &=& p(abc) = a \times b + b \times c + c \times a = \begin{pmatrix}b_x \\ 0 \\ 0\end{pmatrix} \times \begin{pmatrix}c_x \\ c_y \\ 0\end{pmatrix} = \begin{pmatrix}0 \\ 0 \\ b_x c_y\end{pmatrix}
    \f}

    Also, as the sum of applied forces must be zero, we know that:
    \f{eqnarray*}
        Ja+Jb+Jc+Jd &=& 0 \\
        -p(bcd)+p(cda)-p(dab)+b(abc) &=& - b \times c - c \times d - d \times b \\
                                     &+& c \times d + d \times a + a \times c \\
                                     &-& d \times a - a \times b - b \times d \\
                                     &+& a \times b + b \times c + c \times a \\
                                     &=& 0 \\
        Ja &=& -Jb-Jc-Jd
    \f}

    The forces will be computed using \f$F = R J K J^t R^t X\f$.
    We can apply a scaling factor to J if we divide K by its square: \f$F = R (1/b_x J) (b_x^2 K) (1/b_x J)^t R^t X\f$.

    This allows us to do all computations from the values \f$\left(b_x \quad c_x \quad c_y \quad d_x \quad d_y \quad d_z \quad \gamma b_x^2 \quad \mu^2 b_x^2 \quad Jb_x/b_x \quad Jb_y/b_x \quad Jb_z/b_x\right)\f$. Including the 4 vertex indices, this represents 15 values.

    */
    struct GPUElement
    {
        /// @name index of the 4 connected vertices
        /// @{
        //Vec<4,int> tetra;
        int ia[BSIZE],ib[BSIZE],ic[BSIZE],id[BSIZE];
        /// @}

        /// @name material stiffness matrix
        /// @{
        //Mat<6,6,Real> K;
        Real gamma_bx2[BSIZE], mu2_bx2[BSIZE];
        /// @}

        /// @name initial position of the vertices in the local (rotated) coordinate system
        /// @{
        //Vec3f initpos[4];
        Real bx[BSIZE],cx[BSIZE];
        Real cy[BSIZE],dx[BSIZE],dy[BSIZE],dz[BSIZE];
        /// @}

        /// strain-displacement matrix
        /// @{
        //Mat<12,6,Real> J;
        Real Jbx_bx[BSIZE],Jby_bx[BSIZE],Jbz_bx[BSIZE];
        /// @}
    };

    gpu::cuda::CudaVector<GPUElement> elems;

    /// Varying data associated with each element
    struct GPUElementState
    {
        /// transposed rotation matrix
        Real Rt[3][3][BSIZE];
        /// current internal stress
        //Vec<6,Real> S;
        /// unused value to align to 64 bytes
        //Real dummy;
    };

    /// Varying data associated with each element
    struct GPUElementForce
    {
        type::Vec<4, Real> fA, fB, fC, fD;
    };

    gpu::cuda::CudaVector<GPUElementState> initState;
    gpu::cuda::CudaVector<int> rotationIdx;
    gpu::cuda::CudaVector<GPUElementState> state;
    gpu::cuda::CudaVector<GPUElementForce> eforce;
    int nbElement; ///< number of elements
    int vertex0; ///< index of the first vertex connected to an element
    int nbVertex; ///< number of vertices to process to compute all elements
    int nbElementPerVertex; ///< max number of elements connected to a vertex
    /// Index of elements attached to each points (layout per bloc of NBLOC vertices, with first element of each vertex, then second element, etc)
    /// Note that each integer is actually equat the the index of the element * 4 + the index of this vertex inside the tetrahedron.
    int GATHER_PT;
    int GATHER_BSIZE;
    gpu::cuda::CudaVector<int> velems;
    TetrahedronFEMForceFieldInternalData() : nbElement(0), vertex0(0), nbVertex(0), nbElementPerVertex(0) {}
    void init(int nbe, int v0, int nbv, int nbelemperv)
    {
        elems.clear();
        state.clear();
        initState.clear();
        rotationIdx.clear();
        eforce.clear();
        velems.clear();
        nbElement = nbe;
        elems.resize((nbe+BSIZE-1)/BSIZE);
        state.resize((nbe+BSIZE-1)/BSIZE);
        eforce.resize(nbe);
        vertex0 = v0;
        nbVertex = nbv;
        nbElementPerVertex = nbelemperv;
        const int nbloc = (nbVertex+BSIZE-1)/BSIZE;
        velems.resize(nbloc*nbElementPerVertex*BSIZE);
        for (unsigned int i=0; i<velems.size(); i++)
            velems[i] = 0;
    }
    int size() const { return nbElement; }
    void setV(int vertex, int num, int index)
    {
        vertex -= vertex0;
        const int bloc = vertex/BSIZE;
        const int b_x  = vertex%BSIZE;
        velems[ bloc*BSIZE*nbElementPerVertex // start of the bloc
                + num*BSIZE                     // offset to the element
                + b_x                           // offset to the vertex
              ] = index+1;
    }

    void setE(int i, const Element& indices, const Coord& /*a*/, const Coord& b, const Coord& c, const Coord& d, const MaterialStiffness& K, const StrainDisplacement& /*J*/)
    {
        GPUElement& e = elems[i/BSIZE]; i = i%BSIZE;
        e.ia[i] = indices[0] - vertex0;
        e.ib[i] = indices[1] - vertex0;
        e.ic[i] = indices[2] - vertex0;
        e.id[i] = indices[3] - vertex0;
        e.bx[i] = b[0];
        e.cx[i] = c[0]; e.cy[i] = c[1];
        e.dx[i] = d[0]; e.dy[i] = d[1]; e.dz[i] = d[2];
        Real bx2 = e.bx[i] * e.bx[i];
        e.gamma_bx2[i] = K[0][1] * bx2;
        e.mu2_bx2[i] = 2*K[3][3] * bx2;
        e.Jbx_bx[i] = (e.cy[i] * e.dz[i]) / e.bx[i];
        e.Jby_bx[i] = (-e.cx[i] * e.dz[i]) / e.bx[i];
        e.Jbz_bx[i] = (e.cx[i]*e.dy[i] - e.cy[i]*e.dx[i]) / e.bx[i];
    }

    static void reinit(Main* m);
    static void addForce(Main* m, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/);
    static void addDForce (Main* m, VecDeriv& df, const VecDeriv& dx, SReal kFactor, SReal bFactor);
    static void addKToMatrix (Main* m, sofa::linearalgebra::BaseMatrix* mat, SReal kFactor, unsigned int& offset);
    static void getRotations(Main* m, VecReal& rotations);
    static void getRotations(Main* m, linearalgebra::BaseMatrix * rotations,int offset);

    VecReal vecTmpRotation;

    void initPtrData(Main* m)
    {
        m->_gatherPt.beginEdit()->setNames({"1","4","8"});
        m->_gatherPt.beginEdit()->setSelectedItem("8");
        m->_gatherPt.endEdit();

        m->_gatherBsize.beginEdit()->setNames({"32","64","128","256"});
        m->_gatherBsize.beginEdit()->setSelectedItem("256");
        m->_gatherBsize.endEdit();
    }
};

//
// TetrahedronFEMForceField
//

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaTetrahedronFEMForceField_DeclMethods(T) \
    template<> void TetrahedronFEMForceField< T >::reinit(); \
    template<> void TetrahedronFEMForceField< T >::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v); \
    template<> void TetrahedronFEMForceField< T >::getRotations(VecReal& vecR); \
    template<> void TetrahedronFEMForceField< T >::getRotations(linearalgebra::BaseMatrix * vecR,int offset); \
    template<> void TetrahedronFEMForceField< T >::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx); \
    template<> void TetrahedronFEMForceField< T >::addKToMatrix(sofa::linearalgebra::BaseMatrix* mat, SReal kFactor, unsigned int& offset); \


CudaTetrahedronFEMForceField_DeclMethods(gpu::cuda::CudaVec3fTypes);
CudaTetrahedronFEMForceField_DeclMethods(gpu::cuda::CudaVec3f1Types);

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaTetrahedronFEMForceField_DeclMethods(gpu::cuda::CudaVec3dTypes);
CudaTetrahedronFEMForceField_DeclMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaTetrahedronFEMForceField_DeclMethods

} // namespace sofa::component::solidmechanics::fem::elastic
