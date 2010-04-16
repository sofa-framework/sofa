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
#ifndef SOFA_HELPER_TRICKS_H
#define SOFA_HELPER_TRICKS_H

#include <sofa/helper/helper.h>

#include <sofa/helper/vector.h>
#include <string>
#include <iostream>
#include <cstdarg>
#include <sstream>



namespace sofa
{

namespace helper
{

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
/**
 * \namespace sofa::helper
 * \brief RadioTrick is a kind of data for a radio button. It has a list of text
 * representing a list of choices, and a interger number indicating the choice
 * selected.
 *
 */

class RadioTrick : public sofa::helper::vector<std::string>
{
public :
    typedef sofa::helper::vector<std::string> textItems;

    RadioTrick();

    ///Example RadioTrick m_radiobutton(4,"button0","button1","button2","button3");
    RadioTrick(int nbofRadioButton,...);

    ///Copy
    RadioTrick(const RadioTrick & m_radiotrick);
    RadioTrick & operator=(const RadioTrick & m_radiotrick);

    void setSelectedItem(unsigned int id_item);
    void setSelectedItem(const std::string &);
    unsigned int getSelectedId() const;
    std::string getSelectedItem() const;



    ///An other way to do the setSelectedItem() using a string for input
    ///If the reading string is in string list, set the selected item to this
    ///else push a warning.
    void readFromStream(std::istream & stream);

    void TestRadioTrick();

protected:

    unsigned int selectedItem;

    ///return the id_item of the string if found in string list button
    ///             -1    if not found
    int isInButtonList(const std::string & m_string);

};


inline std::istream & operator >>(std::istream & in, RadioTrick & m_trick)
{m_trick.readFromStream(in); return in;}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

} // namespace helper

} // namespace sofa

#endif

