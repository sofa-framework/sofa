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

#include <sofa/helper/config.h>
#include <sofa/type/vector.h>

#include <string>

#include <sofa/helper/StringUtils.h>

namespace sofa::helper
{

struct SOFA_HELPER_API LinkParser
{
    static constexpr char prefix { '@' };
    static constexpr char separator { '/' };


    explicit LinkParser(std::string linkString);

    //pre-condition: cleaned
    [[nodiscard]] bool hasPrefix() const;

    //pre-condition: cleaned
    [[nodiscard]] bool isAbsolute() const;

    LinkParser& cleanLink();

    void validate();

    [[nodiscard]] std::string getLink() const;

    template <class T>
    bool submitErrors(T* t)
    {
        for (const auto& error : m_errors)
        {
            msg_error(t) << error;
        }
        return m_errors.empty();
    }

    //pre-condition: cleaned
    [[nodiscard]] sofa::type::vector<std::string> split() const;

    [[nodiscard]] std::vector<std::string> getErrors() const;

    template<class InputIt>
    std::string join(InputIt first, InputIt last, bool isAbsolute, bool withPrefix = true)
    {
        std::string result = sofa::helper::join(first, last, separator);
        if (isAbsolute)
        {
            result = separator + result;
        }
        if (withPrefix)
        {
            result = prefix + result;
        }
        return result;
    }


private:

    void addError(std::string s);

    std::string m_initialLinkString;
    std::string m_linkString;


    sofa::type::vector< std::string > m_errors;
};

}
