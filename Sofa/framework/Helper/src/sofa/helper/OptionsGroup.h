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

#include <string>
#include <iostream>
#include <cstdarg>
#include <sstream>

#include <sofa/type/vector.h>
#include <sofa/helper/config.h>

namespace sofa::helper
{

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
/**
 * \brief OptionsGroup is a kind of data for a radio button. It has a list of text
 * representing a list of choices, and a integer number indicating the choice
 * selected.
 *
 */
class SOFA_HELPER_API OptionsGroup
{
public :

    using size_type = type::vector<std::string>::size_type;

    /// @name Constructors
    /// @{
    /// Default constructor
    OptionsGroup();

    ///generic constructor taking other string container like list<string>, set<string>, vector<string>
    template <std::ranges::range T>
    explicit OptionsGroup(const T& list);

    template <class T> OptionsGroup(const std::initializer_list<T>& list);

    ///Copy constructor
    OptionsGroup(const OptionsGroup& m_radiotrick);
    /// @}


    /// @name setting operators
    /// @{

    ///Set the number of items
    void setNbItems( size_type nbofRadioButton );

    ///Set the name of the id-th item
    void setItemName( unsigned int id_item, const std::string& name );

    ///Get the vector of names available
    type::vector<std::string> getItemNames();

    template <class T>
    void setNames(const std::initializer_list<T>& list);

    ///Setting the activated item by its id
    OptionsGroup& setSelectedItem(unsigned int id_item);

    ///Setting the activated item by its value (string)
    OptionsGroup& setSelectedItem(const std::string& );

    ///Setting the activated item by a input-stream.
    ///the istream is converted to string.
    ///If the reading string is in options list, its value is set activated,
    ///else push a warning.
    void readFromStream(std::istream& stream);

    /// @}

    /// @name getting information operators
    /// @{
    [[nodiscard]] unsigned int       getSelectedId()                      const;
    [[nodiscard]] const std::string& getSelectedItem()                    const;
    [[nodiscard]] const std::string& operator[](const unsigned int i)     const {return textItems[i];}
    [[nodiscard]] size_t             size()                               const {return textItems.size();}
    void          writeToStream(std::ostream& stream)  const;
    OptionsGroup& operator=(const OptionsGroup& m_radiotrick) = default;
    /// @}

protected:

    type::vector<std::string>   textItems    ;
    unsigned int                selectedItem ;

    template <std::ranges::range T> void buildFromContainer(const T& list);

public:

    ///return the id_item of the string if found in string list button
    ///             -1    if not found
    [[nodiscard]] int isInOptionsList(const std::string & tempostring) const;

};


inline std::ostream & operator <<(std::ostream& on, const OptionsGroup& m_trick)
{
    m_trick.writeToStream(on);
    return on;
}

inline std::istream & operator >>(std::istream& in, OptionsGroup& m_trick)
{
    m_trick.readFromStream(in);
    return in;
}

template <std::ranges::range T>
OptionsGroup::OptionsGroup(const T& list)
{
    buildFromContainer(list);
    selectedItem=0;
}

template <class T>
OptionsGroup::OptionsGroup(const std::initializer_list<T>& list)
{
    buildFromContainer(list);
    selectedItem=0;
}

template <class T>
void OptionsGroup::setNames(const std::initializer_list<T>& list)
{
    textItems.clear();
    buildFromContainer(list);
    selectedItem=0;
}

template <std::ranges::range T>
void OptionsGroup::buildFromContainer(const T& list)
{
    textItems.reserve(list.size());
    for (const auto& item : list)
    {
        std::ostringstream oss;
        oss << item;
        textItems.push_back( oss.str() );
    }
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
} // namespace sofa::helper
