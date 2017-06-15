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
#define SOFA_ImageGaussPointSAMPLER_CPP

#include <Flexible/config.h>
#include "../quadrature/ImageGaussPointSampler.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ImageGaussPointSampler)

// Register in the Factory
int ImageGaussPointSamplerClass = core::RegisterObject("Samples an object represented by an image with gauss points")

        .add<ImageGaussPointSampler<Image<SReal>,ImageUC> >(true)
        .add<ImageGaussPointSampler<Image<SReal>,ImageB> >()
        ;

template class SOFA_Flexible_API ImageGaussPointSampler<Image<SReal>,ImageUC>;
template class SOFA_Flexible_API ImageGaussPointSampler<Image<SReal>,ImageB>;

} // namespace engine
} // namespace component
} // namespace sofa

