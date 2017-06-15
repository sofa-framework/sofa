/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_IDENTITYMULTIMAPPING_H
#define SOFA_COMPONENT_MAPPING_IDENTITYMULTIMAPPING_H
#include "config.h"

#include <sofa/core/MultiMapping.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

namespace sofa
{

namespace component
{

namespace mapping
{

/// concatanate several entire mechanical states together
template <class TIn, class TOut>
class IdentityMultiMapping : public core::MultiMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(IdentityMultiMapping, TIn, TOut), SOFA_TEMPLATE2(core::MultiMapping, TIn, TOut));

    typedef core::MultiMapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename InCoord::value_type Real;
    typedef typename OutCoord::value_type OutReal;
    typedef typename helper::vector <const InVecCoord*> vecConstInVecCoord;
    typedef typename helper::vector<OutVecCoord*> vecOutVecCoord;

    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;


    virtual void init();

    virtual void apply(const core::MechanicalParams* mparams, const helper::vector<OutDataVecCoord*>& dataVecOutPos, const helper::vector<const InDataVecCoord*>& dataVecInPos);
    virtual void applyJ(const core::MechanicalParams* mparams, const helper::vector<OutDataVecDeriv*>& dataVecOutVel, const helper::vector<const InDataVecDeriv*>& dataVecInVel);
    virtual void applyJT(const core::MechanicalParams* mparams, const helper::vector<InDataVecDeriv*>& dataVecOutForce, const helper::vector<const OutDataVecDeriv*>& dataVecInForce);
    virtual void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/) {}
    virtual void applyJT( const core::ConstraintParams* cparams, const helper::vector< InDataMatrixDeriv* >& dataMatOutConst, const helper::vector< const OutDataMatrixDeriv* >& dataMatInConst );

    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs();

    virtual void updateForceMask();


protected :

    IdentityMultiMapping();
    virtual ~IdentityMultiMapping();

    helper::vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector

    typedef linearsolver::EigenSparseMatrix<TIn,TOut> EigenMatrix;
//    size_t previousMaskHash; ///< storing previous dof maskTo to check if it changed from last time step to updateJ in consequence


};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_IDENTITYMULTIMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API IdentityMultiMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_MISC_MAPPING_API IdentityMultiMapping< defaulttype::Rigid3dTypes, defaulttype::Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API IdentityMultiMapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_API IdentityMultiMapping< defaulttype::Rigid3fTypes, defaulttype::Rigid3fTypes >;
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_IDENTITYMULTIMAPPING_H
