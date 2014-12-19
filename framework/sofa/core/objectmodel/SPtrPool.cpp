
/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2014 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/SPtrPool.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/core/ExecParams.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class PoolEnabledArray : public helper::fixed_array<bool, SOFA_DATA_MAX_ASPECTS>
{
public:
	PoolEnabledArray(bool init) { assign(init); }
};

static PoolEnabledArray poolEnabled(false);

void PoolSettings::enable(bool enabled)
{
	poolEnabled[ExecParams::currentAspect()] = enabled;
}

bool PoolSettings::isEnabled()
{
	return poolEnabled[ExecParams::currentAspect()];
}

} // namespace objectmodel

} // namespace core

} // namespace sofa