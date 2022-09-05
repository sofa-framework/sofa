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
#pragma once
#include <sofa/gui/common/MouseOperations.h>

SOFA_DEPRECATED_HEADER("v22.06", "v23.06", "sofa/gui/common/MouseOperations.h")

namespace sofa::gui
{
    using MousePosition = sofa::gui::common::MousePosition;
    using Operation = sofa::gui::common::Operation;
    using AttachOperation = sofa::gui::common::AttachOperation;
    using ConstraintAttachOperation = sofa::gui::common::ConstraintAttachOperation;
    using FixOperation = sofa::gui::common::FixOperation;
    using AddFrameOperation = sofa::gui::common::AddFrameOperation;
    using AddRecordedCameraOperation = sofa::gui::common::AddRecordedCameraOperation;
    using StartNavigationOperation = sofa::gui::common::StartNavigationOperation;
    using InciseOperation = sofa::gui::common::InciseOperation;
    using TopologyOperation = sofa::gui::common::TopologyOperation;
    using AddSutureOperation = sofa::gui::common::AddSutureOperation;

} // namespace sofa::gui
