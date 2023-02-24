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
#include <sofa/core/visual/Data[DisplayFlags].h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/helper/DiffLib.h>

void showErrorUnkownFlag(sofa::core::visual::DisplayFlags& value, sofa::core::objectmodel::Data<sofa::core::visual::DisplayFlags>& data, std::string unknownFlag)
{
    if (data.getOwner())
    {
        const sofa::type::vector<std::string> allFlagNames = value.getAllFlagsLabels();

        std::stringstream tmp;
        tmp << "Unknown flag '" << unknownFlag << "'" << ". The closest existing ones:" << msgendl;
        for(auto& [name, score] : sofa::helper::getClosestMatch(unknownFlag, allFlagNames, 2, 0.6))
        {
            tmp << "\t" << "- " << name << " ("+ std::to_string((int)(100*score))+"% match)" << msgendl;
        }
        tmp << "Complete list is: " << allFlagNames;

        msg_error(data.getOwner()) << tmp.str();
    }
}

void showErrorLetterCase(sofa::core::visual::DisplayFlags& value,
                         sofa::core::objectmodel::Data<sofa::core::visual::DisplayFlags>& data,
                         std::string incorrectLetterCaseFlag, std::string flagWithCorrectLetterCase)
{
    SOFA_UNUSED(value);

    if (data.getOwner())
    {
        msg_error(data.getOwner()) << "Letter case of flag '" + incorrectLetterCaseFlag
            + "' is not correct, please use '" + flagWithCorrectLetterCase + "' instead";
    }
}

template <>
std::istream& sofa::core::objectmodel::Data<sofa::core::visual::DisplayFlags>::readValue(std::istream& in)
{
    auto& value = *beginEdit();

    value.read(in,
        [this, &value](std::string unkownFlag)
        {
            showErrorUnkownFlag(value, *this, unkownFlag);
        },
        [this, &value](std::string incorrectLetterCaseFlag, std::string flagWithCorrectLetterCase)
        {
            showErrorLetterCase(value, *this, incorrectLetterCaseFlag, flagWithCorrectLetterCase);
        });

    endEdit();

    return in;
}
