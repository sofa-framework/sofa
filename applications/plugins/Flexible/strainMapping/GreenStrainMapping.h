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
#ifndef SOFA_COMPONENT_MAPPING_GreenStrainMAPPING_H
#define SOFA_COMPONENT_MAPPING_GreenStrainMAPPING_H

#include <Flexible/config.h>
#include "../strainMapping/BaseStrainMapping.h"
#include "../strainMapping/GreenStrainJacobianBlock.h"

namespace sofa
{
namespace component
{
namespace mapping
{


/** Deformation Gradient to Green Lagrangian Strain mapping
*/

template <class TIn, class TOut>
class GreenStrainMapping : public BaseStrainMappingT<defaulttype::GreenStrainJacobianBlock<TIn,TOut> >
{
public:
    typedef defaulttype::GreenStrainJacobianBlock<TIn,TOut> BlockType;
    typedef BaseStrainMappingT<BlockType > Inherit;

    SOFA_CLASS(SOFA_TEMPLATE2(GreenStrainMapping,TIn,TOut), SOFA_TEMPLATE(BaseStrainMappingT,BlockType ));

    Data<bool> f_geometricStiffness; ///< should geometricStiffness be considered?

protected:
    GreenStrainMapping (core::State<TIn>* from = NULL, core::State<TOut>* to= NULL)
        : Inherit ( from, to )
         , f_geometricStiffness( initData( &f_geometricStiffness, true, "geometricStiffness", "Should geometricStiffness be considered?" ) )
    {
    }

    virtual ~GreenStrainMapping()     { }

    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId childForce)
    {
        if(!f_geometricStiffness.getValue()) return;
        else Inherit::applyDJT(mparams,parentDfId,childForce);
    }
};


} // namespace mapping
} // namespace component
} // namespace sofa

#endif
