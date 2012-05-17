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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_CorotationalStrainMAPPING_H
#define SOFA_COMPONENT_MAPPING_CorotationalStrainMAPPING_H

#include "../initFlexible.h"
#include "../strainMapping/BaseStrainMapping.h"
#include "../strainMapping/CorotationalStrainJacobianBlock.inl"

namespace sofa
{
namespace component
{
namespace mapping
{

using helper::vector;

/** Deformation Gradient to Corotational Lagrangian Strain mapping
*/

template <class TIn, class TOut>
class SOFA_Flexible_API CorotationalStrainMapping : public BaseStrainMapping<defaulttype::CorotationalStrainJacobianBlock<TIn,TOut> >
{
public:
    typedef defaulttype::CorotationalStrainJacobianBlock<TIn,TOut> BlockType;
    typedef BaseStrainMapping<BlockType > Inherit;

    SOFA_CLASS(SOFA_TEMPLATE2(CorotationalStrainMapping,TIn,TOut), SOFA_TEMPLATE(BaseStrainMapping,BlockType ));

    /** @name  Corotational methods */
    //@{
    Data<std::string> f_method;
    sofa::defaulttype::RotationDecompositionMethod decompositionMethod;
    //@}

    virtual void reinit()
    {
        if (f_method.getValue() == "small") decompositionMethod = sofa::defaulttype::SMALL;
        else if (f_method.getValue() == "large" || f_method.getValue() == "qr") decompositionMethod = sofa::defaulttype::QR;
        else if (f_method.getValue() == "svd") decompositionMethod = sofa::defaulttype::SVD;
        else  decompositionMethod = sofa::defaulttype::POLAR;

        for(unsigned int i=0; i<this->jacobian.size(); i++) this->jacobian[i].decompositionMethod=decompositionMethod;
        Inherit::reinit();
    }


protected:
    CorotationalStrainMapping (core::State<TIn>* from = NULL, core::State<TOut>* to= NULL)
        : Inherit ( from, to )
        , f_method(initData(&f_method,std::string("polar"),"method", "\"qr\", \"polar\", \"svd\" or \"small\" displacements"))
    {
    }

    virtual ~CorotationalStrainMapping()     { }

};


} // namespace mapping
} // namespace component
} // namespace sofa

#endif
