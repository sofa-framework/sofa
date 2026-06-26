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

#ifndef IMAGE_IMAGETYPES_MULTITHREAD_H
#define IMAGE_IMAGETYPES_MULTITHREAD_H

#include <image/ImageTypes.h>
#include <image_multithread/config.h>
#include <MultiThreading/DataExchange.h>

namespace sofa::core
{

extern template class SOFA_IMAGE_MULTITHREAD_API DataExchange<sofa::defaulttype::ImageB>;
extern template class SOFA_IMAGE_MULTITHREAD_API DataExchange<sofa::defaulttype::ImageD>;
extern template class SOFA_IMAGE_MULTITHREAD_API DataExchange<sofa::defaulttype::ImageUC>;
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
extern template class SOFA_IMAGE_MULTITHREAD_API DataExchange<sofa::defaulttype::ImageI>;
extern template class SOFA_IMAGE_MULTITHREAD_API DataExchange<sofa::defaulttype::ImageUI>;
extern template class SOFA_IMAGE_MULTITHREAD_API DataExchange<sofa::defaulttype::ImageS>;
extern template class SOFA_IMAGE_MULTITHREAD_API DataExchange<sofa::defaulttype::ImageUS>;
extern template class SOFA_IMAGE_MULTITHREAD_API DataExchange<sofa::defaulttype::ImageL>;
extern template class SOFA_IMAGE_MULTITHREAD_API DataExchange<sofa::defaulttype::ImageUL>;
extern template class SOFA_IMAGE_MULTITHREAD_API DataExchange<sofa::defaulttype::ImageF>;
extern template class SOFA_IMAGE_MULTITHREAD_API DataExchange<sofa::defaulttype::ImageC>;
#endif

}
#endif
