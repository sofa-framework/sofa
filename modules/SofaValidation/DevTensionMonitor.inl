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
#include <SofaValidation/DevTensionMonitor.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace misc
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
        /*sout << "measuring metrics..." << sendl;
        sout << "first point position " << xPos[0].getCenter() << sendl;
        sout << "first point orientation " << xPos[0].getOrientation() << sendl;

        sout << "last point position " << xPos[xPos.size()-1].getCenter() << sendl;
        sout << "last point orientation " << xPos[xPos.size()-1].getOrientation() << sendl;*/

        //Compute tension
        // ....
        defaulttype::Vec1d tension = defaulttype::Vec1d(xPos[0].getOrientation()[0]);
        std::pair<sofa::defaulttype::Vec1d, Real> temp;
        temp.first = tension;
        temp.second = timestamp;

        data.push_back(temp);
    }
}



} // namespace misc

} // namespace component

} // namespace sofa
