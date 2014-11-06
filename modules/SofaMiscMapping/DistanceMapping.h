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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_DistanceMapping_H
#define SOFA_COMPONENT_MAPPING_DistanceMapping_H

#include <sofa/SofaMisc.h>
#include <sofa/core/Mapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace mapping
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class DistanceMappingInternalData
{
public:
};


/** Maps point positions to distances, in meters.
  Type TOut corresponds to a scalar value.
  The pairs are given in an EdgeSetTopologyContainer in the same node.
  If the rest lengths are not defined, they are set using the initial values.
  If computeDistance is set to true, the rest lengths are set to 0.

  (Changed class name on Feb. 4, 2014, previous name was ExtensionMapping)

    In: parent point positions

    Out: distance between point pairs, minus a rest distance.

@author Francois Faure
  */
template <class TIn, class TOut>
class DistanceMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DistanceMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef linearsolver::EigenSparseMatrix<TIn,TOut>   SparseMatrixEigen;
    typedef linearsolver::EigenSparseMatrix<TIn,TIn>    SparseKMatrixEigen;
    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };
    typedef defaulttype::Mat<Out::deriv_total_size, In::deriv_total_size,Real>  Block;
    typedef topology::EdgeSetTopologyContainer::SeqEdges SeqEdges;


    Data< bool >		   f_computeDistance;	///< computeDistance = true ---> restDistance = 0
    Data< vector< Real > > f_restLengths;		///< rest length of each link
    Data< Real >           d_showObjectScale;   ///< drawing size
    Data< defaulttype::Vec4f > d_color;         ///< drawing color

    virtual void init();

    virtual void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in);

    virtual void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in);

    virtual void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in);

    virtual void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in);

    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce );

    virtual const sofa::defaulttype::BaseMatrix* getJ();
    virtual const vector<sofa::defaulttype::BaseMatrix*>* getJs();
    virtual const vector<defaulttype::BaseMatrix*>* getKs();

    virtual void draw(const core::visual::VisualParams* vparams);

protected:
    DistanceMapping();
    virtual ~DistanceMapping();

    topology::EdgeSetTopologyContainer* edgeContainer;  ///< where the edges are defined
    SparseMatrixEigen jacobian;                         ///< Jacobian of the mapping
    SparseMatrixEigen geometricStiffness;               ///< Stiffness due to the non-linearity of the mapping
    vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector
    SparseKMatrixEigen K;                               ///< Assembled geometric stiffness matrix
    vector<defaulttype::BaseMatrix*> stiffnessBaseMatrices;      ///< Vector of geometric stiffness matrices, for the Compliant plugin API
    vector<InDeriv> directions;                         ///< Unit vectors in the directions of the lines
    vector< Real > invlengths;                          ///< inverse of current distances. Null represents the infinity (null distance)

    /// r=b-a only for position (eventual rotation, affine transform... remains null)
    void computeCoordPositionDifference( InDeriv& r, const InCoord& a, const InCoord& b );
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_DistanceMapping_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API DistanceMapping< defaulttype::Vec3dTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_MISC_MAPPING_API DistanceMapping< defaulttype::Rigid3dTypes, defaulttype::Vec1dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API DistanceMapping< defaulttype::Vec3fTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_MISC_MAPPING_API DistanceMapping< defaulttype::Rigid3fTypes, defaulttype::Vec1fTypes >;
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
