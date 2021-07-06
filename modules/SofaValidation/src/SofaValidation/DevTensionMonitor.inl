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
#include <SofaValidation/DevTensionMonitor.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::misc
{

template <class DataTypes>
void DevTensionMonitor<DataTypes>::init()
{
}

template <class DataTypes>
void DevTensionMonitor<DataTypes>::eval()
{
    const VecCoord & xPos = mstate->read(core::ConstVecCoordId::position())->getValue();

    if (f_indices.getValue().empty())
    {
        /*msg_info() << "measuring metrics...";
        msg_info() << "first point position " << xPos[0].getCenter();
        msg_info() << "first point orientation " << xPos[0].getOrientation();

        msg_info() << "last point position " << xPos[xPos.size()-1].getCenter();
        msg_info() << "last point orientation " << xPos[xPos.size()-1].getOrientation();*/

        //Compute tension
        // ....
        type::Vec1d tension = type::Vec1d(xPos[0].getOrientation()[0]);
        std::pair<sofa::type::Vec1d, Real> temp;
        temp.first = tension;
        temp.second = timestamp;

        data.push_back(temp);
    }
}

} // namespace sofa::component::misc