/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_H
#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_H
#include "config.h"

#include <sofa/core/MultiMapping.h>
#include <sofa/core/behavior/BaseMass.h>

#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace mapping
{


template <class TIn, class TOut>
class CenterOfMassMultiMapping : public core::MultiMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CenterOfMassMultiMapping, TIn, TOut), SOFA_TEMPLATE2(core::MultiMapping, TIn, TOut));

    typedef core::MultiMapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef In InDataTypes;
    typedef typename In::Coord    InCoord;
    typedef typename In::Deriv    InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;

    typedef Out OutDataTypes;
    typedef typename Out::Coord   OutCoord;
    typedef typename Out::Deriv   OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename OutCoord::value_type Real;

    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;


    typedef typename helper::vector<OutVecCoord*> vecOutVecCoord;
    typedef typename helper::vector<const InVecCoord*> vecConstInVecCoord;

    virtual void apply(const core::MechanicalParams* mparams, const helper::vector<OutDataVecCoord*>& dataVecOutPos, const helper::vector<const InDataVecCoord*>& dataVecInPos) override;
    //virtual void apply(const helper::vector<OutVecCoord*>& outPos, const vecConstInVecCoord& inPos);

    virtual void applyJ(const core::MechanicalParams* mparams, const helper::vector<OutDataVecDeriv*>& dataVecOutVel, const helper::vector<const InDataVecDeriv*>& dataVecInVel) override;
    //virtual void applyJ(const helper::vector<OutVecDeriv*>& outDeriv, const helper::vector<const  InVecDeriv*>& inDeriv);

    virtual void applyJT(const core::MechanicalParams* mparams, const helper::vector<InDataVecDeriv*>& dataVecOutForce, const helper::vector<const OutDataVecDeriv*>& dataVecInForce) override;
    //virtual void applyJT(const helper::vector< InVecDeriv*>& outDeriv, const helper::vector<const OutVecDeriv*>& inDeriv);


    //virtual void apply(const vecOutVecCoord& outPos, const vecConstInVecCoord& inPos );
    //virtual void applyJ(const helper::vector< OutVecDeriv*>& outDeriv, const helper::vector<const InVecDeriv*>& inDeriv);
    //virtual void applyJT( const helper::vector<InVecDeriv*>& outDeriv , const helper::vector<const OutVecDeriv*>& inDeriv );

    virtual void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/) override {}

    virtual void init() override;
    void draw(const core::visual::VisualParams* vparams) override;

protected:

    CenterOfMassMultiMapping()
        : Inherit()
    {
    }

    virtual ~CenterOfMassMultiMapping() {}

    helper::vector<const core::behavior::BaseMass*> inputBaseMass;
    InVecCoord inputWeightedCOM;
    InVecDeriv inputWeightedForce;
    helper::vector<double> inputTotalMass;
    double invTotalMass;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< defaulttype::Rigid3dTypes, defaulttype::Rigid3dTypes >;
extern template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< defaulttype::Rigid3dTypes, defaulttype::Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< defaulttype::Rigid3fTypes, defaulttype::Rigid3fTypes >;
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_H
