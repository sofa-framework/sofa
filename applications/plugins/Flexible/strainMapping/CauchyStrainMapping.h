/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_CauchyStrainMAPPING_H
#define SOFA_COMPONENT_MAPPING_CauchyStrainMAPPING_H

#include <Flexible/config.h>
#include "../strainMapping/BaseStrainMapping.h"
#include "../strainMapping/CauchyStrainJacobianBlock.h"

namespace sofa
{
namespace component
{
namespace mapping
{

/** Deformation Gradient to linear/Cauchy Strain mapping
*/

template <class TIn, class TOut>
class CauchyStrainMapping : public BaseStrainMappingT<defaulttype::CauchyStrainJacobianBlock<TIn,TOut> >
{
public:
    typedef defaulttype::CauchyStrainJacobianBlock<TIn,TOut> BlockType;
    typedef BaseStrainMappingT<BlockType > Inherit;

    SOFA_CLASS(SOFA_TEMPLATE2(CauchyStrainMapping,TIn,TOut), SOFA_TEMPLATE(BaseStrainMappingT,BlockType ));

protected:
    CauchyStrainMapping (core::State<TIn>* from = NULL, core::State<TOut>* to= NULL)
        : Inherit ( from, to )
    {
    }

    virtual ~CauchyStrainMapping()     { }

    virtual void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*parentDfId*/, core::ConstMultiVecDerivId /*childForce*/)    {}
};


} // namespace mapping
} // namespace component
} // namespace sofa

#endif
