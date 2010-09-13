/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
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
 * \namespace sofa::helper
 * \brief OptionsGroup is a kind of data for a radio button. It has a list of text
 * representing a list of choices, and a interger number indicating the choice
 * selected.
 *
 */

class SOFA_HELPER_API OptionsGroup //: public std::vector<std::string>
{
public :
//	typedef
    helper::vector<std::string> textItems;

    OptionsGroup();

    ///Example OptionsGroup(4,"button0","button1","button2","button3");
    OptionsGroup(int nbofRadioButton,...);

    template <class T>
    OptionsGroup(const T &list)
    {
        for (typename T::const_iterator it=list.begin(); it!=list.end(); ++it)
        {
            std::ostringstream oss;
            oss << (*it);
            textItems.push_back( oss.str() );
        }
        selectedItem=0;
    }

    ///Copy
    OptionsGroup(const OptionsGroup & m_radiotrick);

    OptionsGroup & operator=(const OptionsGroup & m_radiotrick);

    ///Example OptionsGroup::setNames(4,"button0","button1","button2","button3");
    void setNames(int nbofRadioButton,...);
    void setSelectedItem(unsigned int id_item);
    void setSelectedItem(const std::string &);
    unsigned int getSelectedId() const;
    const std::string &getSelectedItem() const;
    std::string & operator[](unsigned int i) {return textItems[i];}
    unsigned int size() const {return textItems.size();}


    ///An other way to do the setSelectedItem() using a string for input
    ///If the reading string is in string list, set the selected item to this
    ///else push a warning.
    void readFromStream(std::istream & stream);
    void writeToStream(std::ostream & stream) const;


protected:

    unsigned int selectedItem;

    ///return the id_item of the string if found in string list button
    ///             -1    if not found
    int isInButtonList(const std::string & m_string) const;

};


inline std::ostream & operator <<(std::ostream & on, const OptionsGroup & m_trick)
{m_trick.writeToStream(on); return on;}

inline std::istream & operator >>(std::istream & in, OptionsGroup & m_trick)
{m_trick.readFromStream(in); return in;}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


} // namespace helper

} // namespace sofa

#endif /* SOFA_HELPER_OPTIONSGROUP_H */
