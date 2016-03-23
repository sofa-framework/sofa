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
#ifndef SOFA_COMPONENT_MAPPING_InvariantMAPPING_H
#define SOFA_COMPONENT_MAPPING_InvariantMAPPING_H

#include <Flexible/config.h>
#include "../strainMapping/BaseStrainMapping.h"
#include "../strainMapping/InvariantJacobianBlock.inl"

namespace sofa
{
namespace component
{
namespace mapping
{


/** Map deformation gradients to invariants of the right Cauchy Green deformation tensor: I1, I2 and J
s*/

template <class TIn, class TOut>
class InvariantMapping : public BaseStrainMappingT<defaulttype::InvariantJacobianBlock<TIn,TOut> >
{
public:
    typedef defaulttype::InvariantJacobianBlock<TIn,TOut> BlockType;
    typedef BaseStrainMappingT<BlockType > Inherit;

    SOFA_CLASS(SOFA_TEMPLATE2(InvariantMapping,TIn,TOut), SOFA_TEMPLATE(BaseStrainMappingT,BlockType ));

protected:
    InvariantMapping (core::State<TIn>* from = NULL, core::State<TOut>* to= NULL)
        : Inherit ( from, to )
    {
    }

    virtual ~InvariantMapping()     { }

};


} // namespace mapping
} // namespace component
} // namespace sofa

#endif
