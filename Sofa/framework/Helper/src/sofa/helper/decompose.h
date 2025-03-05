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
#ifndef SOFA_HELPER_DECOMPOSE_H
#define SOFA_HELPER_DECOMPOSE_H
#include <sofa/helper/config.h>

#include <sofa/type/Mat.h>


namespace sofa::helper
{

template<class Real>
class Decompose
{

public:

    /** @name QR
      * @{
      */

    /** QR decomposition
      Compute an orthonormal right-handed 3x3 basis based on two vectors using Gram-Schmidt orthogonalization.
      The basis vectors are the columns of the matrix R. The matrix represents the rotation of the local frame with respect to the reference frame.
      The first basis vector is aligned to the first given vector, the second basis vector is in the plane of the two first given vectors, and the third basis vector is orthogonal to the two others.
      Undefined result if one of the vectors is null, or if the two vectors are parallel.
      */
    static void getRotation( type::Mat<3,3,Real>& r, type::Vec<3,Real>& edgex, type::Vec<3,Real>& edgey );

    /** QR decomposition
      Compute an orthonormal right-handed 3x3 basis based on a matrix using Gram-Schmidt orthogonalization.
      The basis vectors are the columns of the matrix R. The matrix represents the rotation of the local frame with respect to the reference frame.
      The first basis vector is aligned to the first given vector, the second basis vector is in the plane of the two first given vectors, and the third basis vector is orthogonal to the two others.
      Undefined result if one of the vectors is null, or if the two vectors are parallel.
      */
    static void QRDecomposition( const type::Mat<3,3,Real> &M, type::Mat<3,3,Real> &R );
    static void QRDecomposition( const type::Mat<3,2,Real> &M, type::Mat<3,2,Real> &R );
    static void QRDecomposition( const type::Mat<2,2,Real> &M, type::Mat<2,2,Real> &R );

    /** QR decomposition stable to null columns.
      * Result is still undefined if two columns are parallel.
      * In the clean case (not degenerated), there are only two additional 'if(x<e)' but no additional computations.
      * \returns true in a degenerated configuration
      */
    static bool QRDecomposition_stable( const type::Mat<3,3,Real> &M, type::Mat<3,3,Real> &R );
    static bool QRDecomposition_stable( const type::Mat<3,2,Real> &M, type::Mat<3,2,Real> &R );
    static bool QRDecomposition_stable( const type::Mat<2,2,Real> &M, type::Mat<2,2,Real> &R );

    /** QR decomposition (M=QR) rotation gradient dQ  (invR = R^-1)
      * Formula given in "Finite Random Matrix Theory, Jacobians of Matrix Transforms (without wedge products)", Alan Edelman, 2005, http://web.mit.edu/18.325/www/handouts/handout2.pdf
      * Note that dR is also easy to compute.
      */
    template<Size spatial_dimension, Size material_dimension>
    static void QRDecompositionGradient_dQ( const type::Mat<spatial_dimension,material_dimension,Real>&Q,
                                            const type::Mat<material_dimension,material_dimension,Real>&invR,
                                            const type::Mat<spatial_dimension,material_dimension,Real>& dM,
                                            type::Mat<spatial_dimension,material_dimension,Real>& dQ )
    {
        // dQ = Q ( lower(QT*dM*R−1) − lower(QT*dM*R−1)^T )
        // dR =   ( upper(QT*dM*R−1) + lower(QT*dM*R−1)^T ) R
        // lower -> strictly lower

        // tmp = QT*dM*R^−1
        type::Mat<material_dimension,material_dimension,Real> tmp = Q.multTranspose(dM * invR);

        // L = lower(tmp) - (lower(tmp))^T
        type::Mat<material_dimension,material_dimension,Real> L;

        for(Size i = 0; i < material_dimension; ++i)
        {
            for(Size j = 0; j < i; ++j) // strictly lower
                L[i][j] = tmp[i][j];
            for(Size j = i + 1; j < material_dimension; ++j) // strictly lower transposed
                L[i][j] = -tmp[j][i];
        }

        dQ = Q * L;
    }





    /** @}
      * @name Polar
      * @{
      */


    /** Polar Decomposition of 3x3 matrix,
     * M = QS.  See Nicholas Higham and Robert S. Schreiber,
     * Fast Polar Decomposition of An Arbitrary Matrix,
     * Technical Report 88-942, October 1988,
     * Department of Computer Science, Cornell University.
     */
    static Real polarDecomposition( const type::Mat<3,3,Real>& M, type::Mat<3,3,Real>& Q, type::Mat<3,3,Real>& S );

    /** The same as previous except we do not care about S
     */
    static Real polarDecomposition( const type::Mat<3,3,Real>& M, type::Mat<3,3,Real>& Q );

    /** Polar decomposition of a 2x2 matrix M = QS
     *  Analytic formulation given in
     *  "Matrix Animation and Polar Decomposition"
     *  Ken Shoemake, Computer Graphics Laboratory, University of Pennsylvania
     *  Tom Duff, AT&T Bell Laboratories, Murray Hill
     */
    static void polarDecomposition( const type::Mat<2,2,Real>& M, type::Mat<2,2,Real>& Q );

    /** Stable Polar Decomposition of 3x3 matrix based on a stable SVD using Q=UVt where M=UsV
      * \returns true iff the stabilization processed an inverted rotation or a degenerate case
     */
    static bool polarDecomposition_stable( const type::Mat<3,3,Real> &M, type::Mat<3,3,Real> &Q, type::Mat<3,3,Real> &S );
    static bool polarDecomposition_stable( const type::Mat<3,3,Real> &M, type::Mat<3,3,Real> &Q );
    static bool polarDecomposition_stable( const type::Mat<2,2,Real> &M, type::Mat<2,2,Real> &Q, type::Mat<2,2,Real> &S );
    static bool polarDecomposition_stable( const type::Mat<2,2,Real> &M, type::Mat<2,2,Real> &Q );

    /** Stable Polar Decomposition of 3x2 matrix based on a SVD using Q=UVt where M=UsV
     */
    static void polarDecomposition( const type::Mat<3,2,Real> &M, type::Mat<3,2,Real> &Q, type::Mat<2,2,Real> &S );


    /** Polar decomposition gradient, preliminary step: computation of invG = ((tr(S)*I-S)*Qt)^-1
     *  Inspired by Jernej Barbic, Yili Zhao, "Real-time Large-deformation Substructuring" SIGGRAPH 2011
     *  Note that second derivatives are also given in this paper
     *  Another way to compute the first derivatives are given in Yi-Chao Chen, Lewis Wheeler, "Derivatives of the stretch and rotation tensors", Journal of elasticity in 1993
     */
    static void polarDecompositionGradient_G( const type::Mat<3,3,Real>& Q, const type::Mat<3,3,Real>& S, type::Mat<3,3,Real>& invG );

    /** Polar decomposition rotation gradient, computes the rotation gradient dQ of a given polar decomposition
     *  First, invG needs to be computed with function polarDecompositionGradient_G
     */
    static void polarDecompositionGradient_dQ( const type::Mat<3,3,Real>& invG, const type::Mat<3,3,Real>& Q, const type::Mat<3,3,Real>& dM, type::Mat<3,3,Real>& dQ );
    static void polarDecompositionGradient_dQOverdM(const type::Mat<3,3,Real> &Q, const type::Mat<3,3,Real> &invG,  type::Mat<9,9,Real>& J);
    // another method based on the relation : M=QS -> dQ = (dM - Q dS)S^-1  ->  dQ = (dM - dSOverdM.dM)S^-1  -> dQ = JdM
    static void polarDecompositionGradient_dQOverdM(const type::Mat<3,3,Real> &Q, const type::Mat<3,3,Real> &Sinv, const type::Mat<9,9,Real>& dSOverdM, type::Mat<9,9,Real>& J);

    /** Polar decomposition rotation gradient, computes the strain gradient dS of a given polar decomposition
     *  qQ needs to be computed with function polarDecompositionGradient_dQ
     */
    static void polarDecompositionGradient_dS( const type::Mat<3,3,Real>& Q, const type::Mat<3,3,Real>& S, const type::Mat<3,3,Real>& dQ, const type::Mat<3,3,Real>& dM, type::Mat<3,3,Real>& dS );
    static void polarDecompositionGradient_dSOverdM(const type::Mat<3,3,Real> &Q, const type::Mat<3,3,Real> &M, const  type::Mat<3,3,Real>& invG,  type::Mat<9,9,Real>& J);
    // another method based on the relation :  M^TM = S.S -> M^TdM +dM^TM = dS.S + S.dS  -> J1.dM = J2.dS  -> J.dM = dS;  Requires the inversion of a 6x6 matrix..
    static void polarDecompositionGradient_dSOverdM(const type::Mat<3,3,Real> &M, const type::Mat<3,3,Real> &S,  type::Mat<9,9,Real>& J);

    /** Polar decomposition rotation gradient, computes the strain gradient dS of a given polar decomposition computed by a SVD such as M = U*Sdiag*V
      * Christopher Twigg, Zoran Kacic-Alesic, "Point Cloud Glue: Constraining simulations using the Procrustes transform", SCA'10
     */
    static bool polarDecomposition_stable_Gradient_dQ( const type::Mat<3,3,Real>& U, const type::Vec<3,Real>& Sdiag, const type::Mat<3,3,Real>& V, const type::Mat<3,3,Real>& dM, type::Mat<3,3,Real>& dQ );
    static bool polarDecomposition_stable_Gradient_dQOverdM( const type::Mat<3,3,Real> &U, const type::Vec<3,Real> &Sdiag, const type::Mat<3,3,Real> &V, type::Mat<9,9,Real>& dQOverdM );

    /** Polar decomposition rotation gradient, computes the strain gradient dS of a given polar decomposition computed by a SVD such as M = U*Sdiag*V
      * Christopher Twigg, Zoran Kacic-Alesic, "Point Cloud Glue: Constraining simulations using the Procrustes transform", SCA'10
     */
    static bool polarDecompositionGradient_dQ( const type::Mat<3,2,Real>& U, const type::Vec<2,Real>& Sdiag, const type::Mat<2,2,Real>& V, const type::Mat<3,2,Real>& dM, type::Mat<3,2,Real>& dQ );
    static bool polarDecompositionGradient_dQOverdM( const type::Mat<3,2,Real>& U, const type::Vec<2,Real>& Sdiag, const type::Mat<2,2,Real>& V, type::Mat<6,6,Real>& dQOverdM );


    /** @}
      * @name Eigen Decomposition
      * @{
      */

    /** Non-iterative & faster Eigensystem decomposition: eigenvalues @param diag and eigenvectors (columns of @param V) of the 3x3 Real Matrix @param A
      * Derived from Wild Magic Library
      */
    static void eigenDecomposition( const type::Mat<3,3,Real> &A, type::Mat<3,3,Real> &V, type::Vec<3,Real> &diag );

    /// Non-iterative Eigensystem decomposition: eigenvalues @param diag and eigenvectors (columns of @param V) of the 2x2 Real Matrix @param A
    /// @warning this method is instable in specific configurations TODO
    static void eigenDecomposition( const type::Mat<2,2,Real> &A, type::Mat<2,2,Real> &V, type::Vec<2,Real> &diag );


    /** Eigensystem decomposition: eigenvalues @param diag and eigenvectors (columns of @param V) of the 3x3 Real Matrix @param M
      * Derived from Wild Magic Library
      */
    static void eigenDecomposition_iterative( const type::Mat<3,3,Real> &M, type::Mat<3,3,Real> &V, type::Vec<3,Real> &diag );

    /** Eigensystem decomposition: eigenvalues @param diag and eigenvectors (columns of @param V) of the 2x2 Real Matrix @param M
      * Derived from Wild Magic Library
      */
    static void eigenDecomposition_iterative( const type::Mat<2,2,Real> &M, type::Mat<2,2,Real> &V, type::Vec<2,Real> &diag );

    /** @} */


    /** @}
      * @name SVD
      * @{
      */

    /** SVD F = U*F_diagonal*V based on the Eigensystem decomposition of FtF
      * all eigenvalues are positive
      * Warning U & V are not guarantee to be rotations (they can be reflexions), eigenvalues are not sorted
      */
    static void SVD( const type::Mat<3,3,Real> &F, type::Mat<3,3,Real> &U, type::Vec<3,Real> &S, type::Mat<3,3,Real> &V );

    /** SVD based on the Eigensystem decomposition of FtF with robustness against inversion and degenerate configurations
      * \returns true iff the stabilization processed an inverted rotation or a degenerate case
      * U & V are rotations
      * Warning eigenvalues are not guaranteed to be positive, eigenvalues are not sorted
      */
    static bool SVD_stable( const type::Mat<3,3,Real> &F, type::Mat<3,3,Real> &U, type::Vec<3,Real> &S, type::Mat<3,3,Real> &V );
    static bool SVD_stable( const type::Mat<2,2,Real> &F, type::Mat<2,2,Real> &U, type::Vec<2,Real> &S, type::Mat<2,2,Real> &V );

    /** SVD F = U*F_diagonal*V based on the Eigensystem decomposition of FtF
      * all eigenvalues are positive
      * Warning U & V are not guarantee to be rotations (they can be reflexions), eigenvalues are not sorted
      */
    static void SVD( const type::Mat<3,2,Real> &F, type::Mat<3,2,Real> &U, type::Vec<2,Real> &S, type::Mat<2,2,Real> &V );

    /** SVD based on the Eigensystem decomposition of FtF with robustness against inversion and degenerate configurations
      * \returns true in a degenerate case
      * U & V are rotations
      * Warning eigenvalues are not guaranteed to be positive, eigenvalues are not sorted
      */
    static bool SVD_stable( const type::Mat<3,2,Real> &F, type::Mat<3,2,Real> &U, type::Vec<2,Real> &S, type::Mat<2,2,Real> &V );


    /** SVD rotation gradients, computes the rotation gradients dU & dV
      * T. Papadopoulo, M.I.A. Lourakis, "Estimating the Jacobian of the Singular Value Decomposition: Theory and Applications", European Conference on Computer Vision, 2000
     */
    static bool SVDGradient_dUdV( const type::Mat<3,3,Real> &U, const type::Vec<3,Real> &S, const type::Mat<3,3,Real> &V, const type::Mat<3,3,Real>& dM, type::Mat<3,3,Real>& dU, type::Mat<3,3,Real>& dV );
    static bool SVDGradient_dUdVOverdM( const type::Mat<3,3,Real> &U, const type::Vec<3,Real> &S, const type::Mat<3,3,Real> &V, type::Mat<9,9,Real>& dUOverdM, type::Mat<9,9,Real>& dVOverdM );

    /** SVD rotation gradients, computes the rotation gradients dU & dV
      * T. Papadopoulo, M.I.A. Lourakis, "Estimating the Jacobian of the Singular Value Decomposition: Theory and Applications", European Conference on Computer Vision, 2000
     */
    static bool SVDGradient_dUdV( const type::Mat<3,2,Real> &U, const type::Vec<2,Real> &S, const type::Mat<2,2,Real> &V, const type::Mat<3,2,Real>& dM, type::Mat<3,2,Real>& dU, type::Mat<2,2,Real>& dV );
    static bool SVDGradient_dUdVOverdM( const type::Mat<3,2,Real> &U, const type::Vec<2,Real> &S, const type::Mat<2,2,Real> &V, type::Mat<6,6,Real>& dUOverdM, type::Mat<4,6,Real>& dVOverdM );


    /** @}
      * @name Diagonalization
      * @{
      */

    /// Diagonalization of a symmetric 3x3 matrix
    /// A = Q.w.Q^{-1} with w the eigenvalues and Q the eigenvectors
    static int symmetricDiagonalization( const type::Mat<3,3,Real> &A, type::Mat<3,3,Real> &Q, type::Vec<3,Real> &w );


    /// project a symmetric 3x3 matrix to a PSD (symmetric, positive semi-definite)
    static void PSDProjection( type::Mat<3,3,Real> &A );

    /// project a symmetric 2x2 matrix to a PSD (symmetric, positive semi-definite)
    static void PSDProjection( type::Mat<2,2,Real> &A );
    static void PSDProjection( Real& A00, Real& A01, Real& A10, Real& A11 );

    // does nothing, for template compatibility
    static void PSDProjection( type::Mat<1,1,Real> & ) {}



    /// project a symmetric 3x3 matrix to a NSD (symmetric, negative semi-definite)
    static void NSDProjection( type::Mat<3,3,Real> &A );

    /// project a symmetric 2x2 matrix to a NSD (symmetric, negative semi-definite)
    static void NSDProjection( type::Mat<2,2,Real> &A );
    static void NSDProjection( Real& A00, Real& A01, Real& A10, Real& A11 );

    // does nothing, for template compatibility
    static void NSDProjection( type::Mat<1,1,Real> & ) {}


    /** @} */

    /// threshold for zero comparison (1e-6 for float and 1e-8 for double)
    static Real zeroTolerance();



private:



//    /** @internal useful for polarDecomposition
//      * Set MadjT to transpose of inverse of M times determinant of M
//      */
//    static void adjoint_transpose(const type::Mat<3,3,Real>& M, type::Mat<3,3,Real>& MadjT);

//    /** @internal useful for polarDecomposition
//      * Compute the infinity norm of M
//      */
//    static Real norm_inf(const type::Mat<3,3,Real>& M);

//    /** @internal useful for polarDecomposition
//      * Compute the 1 norm of M
//      */
//    static Real norm_one(const type::Mat<3,3,Real>& M);

//    /** @internal useful for polarDecomposition
//      * Return index of column of M containing maximum abs entry, or -1 if M=0
//      */
//    static int find_max_col(const type::Mat<3,3,Real>& M);

//    /** @internal useful for polarDecomposition
//      * Setup u for Household reflection to zero all v components but first
//      */
//    static void make_reflector(const type::Vec<3,Real>& v, type::Vec<3,Real>& u);

//    /** @internal useful for polarDecomposition
//      * Apply Householder reflection represented by u to column vectors of M
//      */
//    static void reflect_cols(type::Mat<3,3,Real>& M, const type::Vec<3,Real>& u);
//    /** @internal useful for polarDecomposition
//      * Apply Householder reflection represented by u to row vectors of M
//      */
//    static void reflect_rows(type::Mat<3,3,Real>& M, const type::Vec<3,Real>& u);

//    /** @internal useful for polarDecomposition
//      * Find orthogonal factor Q of rank 1 (or less) M
//      */
//    static void do_rank1(type::Mat<3,3,Real>& M, type::Mat<3,3,Real>& Q);

//    /** @internal useful for polarDecomposition
//      * Find orthogonal factor Q of rank 2 (or less) M using adjoint transpose
//      */
//    static void do_rank2(type::Mat<3,3,Real>& M, type::Mat<3,3,Real>& MadjT, type::Mat<3,3,Real>& Q);

    /** @internal useful for polarDecompositionGradient
      * \returns M such as Mu = cross( v, u ), note that M is antisymmetric
      */
    static type::Mat<3,3,Real> skewMat( const type::Vec<3,Real>& v );

    /** @internal useful for polarDecompositionGradient
      * Returns the "skew part" v of a matrix such as skewMat(v) = 0.5*(M-Mt)
      */
    static type::Vec<3,Real> skewVec( const type::Mat<3,3,Real>& M );


    /// @internal useful for eigenDecomposition
    static void ComputeRoots( const type::Mat<3,3,Real>& A, double root[3] );

    /// @internal useful for eigenDecomposition
    static bool PositiveRank( type::Mat<3,3,Real>& M, Real& maxEntry, type::Vec<3,Real>& maxRow );

    /// @internal useful for eigenDecomposition
    // Input vec0 must be a unit-length vector.  The output vectors
    // {vec0,vec1} are unit length and mutually perpendicular, and
    // {vec0,vec1,vec2} is an orthonormal basis.
    static void GenerateComplementBasis( type::Vec<3,Real>& vec0, type::Vec<3,Real>& vec1, const type::Vec<3,Real>& vec2 );

    /// @internal useful for eigenDecomposition
    static void ComputeVectors( const type::Mat<3,3,Real>& A, type::Vec<3,Real>& U2, int i0, int i1, int i2, type::Mat<3,3,Real> &V, type::Vec<3,Real> &diag );


    /** @internal useful for iterative eigenDecomposition
      * QL algorithm with implicit shifting, applies to tridiagonal matrices
      * Derived from numerical recipes
      */
    template <sofa::Size iSize>
    static void QLAlgorithm( type::Vec<iSize,Real> &diag, type::Vec<iSize,Real> &subDiag, type::Mat<iSize,iSize,Real> &V );

}; // class Decompose

template<>
SOFA_HELPER_API inline float Decompose<float>::zeroTolerance()
{
    return 1e-6f;
}

template<>
SOFA_HELPER_API inline double Decompose<double>::zeroTolerance()
{
    return 1e-8;
}

#if !defined(SOFA_BUILD_HELPER)
extern template class SOFA_HELPER_API Decompose<double>;
extern template class SOFA_HELPER_API Decompose<float>;
#endif

} // namespace sofa::helper


#endif // SOFA_HELPER_DECOMPOSE_H
