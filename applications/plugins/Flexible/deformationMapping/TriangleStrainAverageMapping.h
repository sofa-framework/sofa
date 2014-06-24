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
#ifndef SOFA_COMPONENT_MAPPING_TriangleStrainAverageMapping_H
#define SOFA_COMPONENT_MAPPING_TriangleStrainAverageMapping_H

#include <sofa/core/Mapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <sofa/core/State.inl>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include "../initFlexible.h"
#include "../types/DeformationGradientTypes.h"
#include "../shapeFunction/BaseShapeFunction.h"


namespace sofa
{
using helper::vector;

namespace component
{

namespace mapping
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class TriangleStrainAverageMappingInternalData
{
public:
};


/** Averages triangle strains to the nodes of the triangles.
  Input: strains in triangles. Output: strains in nodes, as averages of the strains of the adjacent triangles.

@author Francois Faure
  */
template <class TIn, class TOut>
class TriangleStrainAverageMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(TriangleStrainAverageMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

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
    typedef linearsolver::EigenSparseMatrix<TIn,TOut>    SparseMatrixEigen;
    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };
    typedef defaulttype::Mat<In::deriv_total_size, In::deriv_total_size,Real>  InBlock;
    typedef defaulttype::Mat<Out::deriv_total_size, In::deriv_total_size,Real>  Block;
    typedef topology::TriangleSetTopologyContainer::SeqTriangles SeqTriangles;



    Data< vector<unsigned> > f_triangleIndices;  ///< For each node, indices of the adjacent triangles
    Data< vector<unsigned> > f_endIndices;   ///< For each node, index of the end of the list of triangle indices, in f_indices.
    Data< vector<Real> > f_weights;      ///< For each node, weight of the triangles in the average

    virtual void init();

    virtual void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in);

    virtual void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in);

    virtual void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in);

    virtual void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in);


    virtual const sofa::defaulttype::BaseMatrix* getJ();
    virtual const vector<sofa::defaulttype::BaseMatrix*>* getJs();


protected:
    TriangleStrainAverageMapping();
    virtual ~TriangleStrainAverageMapping();

    topology::TriangleSetTopologyContainer::SPtr triangleContainer;  ///< where the edges are defined
    SparseMatrixEigen jacobian;                         ///< Jacobian of the mapping
    vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector

    /// Compute the product, used in apply and applyJ
    virtual void mult(Data<OutVecCoord>& out, const Data<InVecCoord>& in);

    vector<Real> diagMat; ///< diagonal matrix used to scale up node values based on the area they represent
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_TriangleStrainAverageMapping_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API TriangleStrainAverageMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::F321dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API TriangleStrainAverageMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::F321fTypes >;
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
