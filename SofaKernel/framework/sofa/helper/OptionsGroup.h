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
#ifndef SOFA_HELPER_OPTIONSGROUP_H
#define SOFA_HELPER_OPTIONSGROUP_H

#include <string>
#include <iostream>
#include <cstdarg>
#include <sstream>

#include <sofa/helper/vector.h>
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
/**
 * \brief OptionsGroup is a kind of data for a radio button. It has a list of text
 * representing a list of choices, and a interger number indicating the choice
 * selected.
 *
 */

class SOFA_HELPER_API OptionsGroup
{
public :

    /// @name Constructors
    /// @{
    /// Default constructor
    OptionsGroup();

    ///Constructor by given the number of argument following by the variable arguments
    ///Example OptionsGroup m_options(4,"button0","button1","button2","button3");
    OptionsGroup(int nbofRadioButton,...);

    ///generic constructor taking other string container like list<string>, set<string>, vector<string>
    template <class T> OptionsGroup(const T& list);

    ///Copy constructor
    OptionsGroup(const OptionsGroup& m_radiotrick);
    /// @}


    /// @name setting operators
    /// @{

    ///Set the number of items
    void setNbItems( unsigned int nbofRadioButton );

    ///Set the name of the id-th item
    void setItemName( unsigned int id_item, const std::string& name );

    ///Reinitializing options by a pre-constructed optionsgroup objected
    ///Example m_options.setNames(4,"button0","button1","button2","button3");
    void setNames(int nbofRadioButton,...);

    ///Setting the activated item by its id
    void setSelectedItem(unsigned int id_item);

    ///Setting the activated item by its value (string)
    void setSelectedItem(const std::string& );

    ///Setting the activated item by a input-stream.
    ///the istream is converted to string.
    ///If the reading string is in options list, its value is setted activated,
    ///else push a warning.
    void readFromStream(std::istream& stream);

    /// @}

    /// @name getting informations operators
    /// @{
    unsigned int       getSelectedId()                      const;
    const std::string& getSelectedItem()                    const;
    const std::string& operator[](const unsigned int i)     const {return textItems[i];}
    size_t             size()                               const {return textItems.size();}
    void               writeToStream(std::ostream& stream)  const;
    OptionsGroup&      operator=(const OptionsGroup& m_radiotrick);
    /// @}

protected:

    helper::vector<std::string> textItems    ;
    unsigned int                selectedItem ;

public:

    ///return the id_item of the string if found in string list button
    ///             -1    if not found
    int isInOptionsList(const std::string & m_string) const;

};


inline std::ostream & operator <<(std::ostream& on, const OptionsGroup& m_trick)
{m_trick.writeToStream(on); return on;}

inline std::istream & operator >>(std::istream& in, OptionsGroup& m_trick)
{m_trick.readFromStream(in); return in;}


template <class T>
inline OptionsGroup::OptionsGroup(const T& list)
{
    for (typename T::const_iterator it=list.begin(); it!=list.end(); ++it)
    {
        std::ostringstream oss;
        oss << (*it);
        textItems.push_back( oss.str() );
    }
    selectedItem=0;
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


} // namespace helper

} // namespace sofa

#endif /* SOFA_HELPER_OPTIONSGROUP_H */
