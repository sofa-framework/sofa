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
template< typename T >
class RadioTrick : public sofa::helper::vector<T>
{
public :
    typedef sofa::helper::vector<std::string> textItems;

    RadioTrick() : textItems()
    {
        selectedItem=0;
    }
///////////////////////////////////////
    RadioTrick(int nbofRadioButton,...)
    {
        textItems::resize(nbofRadioButton);
        va_list vl;
        va_start(vl,nbofRadioButton);
        for(unsigned int i=0; i<textItems::size(); i++)
        {
            const char * tempochar=va_arg(vl,char *);
            std::string  tempostring(tempochar);
            textItems::operator[](i)=tempostring;
        }
        va_end(vl);
        selectedItem=0;
    }
///////////////////////////////////////
    RadioTrick(const RadioTrick & m_radiotrick) : textItems(m_radiotrick)
    {
        selectedItem=m_radiotrick.getSelectedId();
    }
///////////////////////////////////////
    /*RadioTrick & RadioTrick::operator=(const RadioTrick & m_radiotrick)
    {
    	RadioTrick m_newRadioTrick(m_radiotrick);
    	return m_newRadioTrick;
    }*/
///////////////////////////////////////

///////////////////////////////////////
    void setSelectedItem(unsigned int id_item)
    {
        if (id_item<textItems::size())
            selectedItem=id_item;
        std::cout<<"=============================checked number changed :"<<id_item<<std::endl;
    }
///////////////////////////////////////
    void setSelectedItem(const std::string & m_string)
    {
        int id_stringinButtonList = isInButtonList(m_string);
        if (id_stringinButtonList == -1)
        {
            std::cout<<"WARNING(RadioTrick) : \""<< m_string <<"\" is not a parameter in button list :\" "<<(*this)<<"\""<< std::endl;
        }
        else
        {
            setSelectedItem(id_stringinButtonList);
        }
    }
///////////////////////////////////////
    unsigned int getSelectedId() const
    {
        return selectedItem;
    }
///////////////////////////////////////
    std::string  getSelectedItem() const
    {
        std::string checkedString;
        checkedString = textItems::operator[](selectedItem);
        return checkedString;
    }
///////////////////////////////////////
    void readFromStream(std::istream & stream)
    {
        std::string tempostring;
        stream >> tempostring;
        int id_stringinButtonList = isInButtonList(tempostring);
        if (id_stringinButtonList == -1)
        {
            std::cout<<"WARNING(RadioTrick) : \""<< tempostring <<"\" is not a parameter in button list :\" "<<(*this)<<"\""<< std::endl;
        }
        else
        {
            setSelectedItem(id_stringinButtonList);
        }
    }
///////////////////////////////////////
    void TestRadioTrick()
    {
        sofa::helper::RadioTrick<std::string> m_radiotrick(3,"hello1","hello2","hello3");
        std::cout<<"Radio button :"<<m_radiotrick<<"    selectedId :"<<m_radiotrick.getSelectedId()<<"   getSelectedItem() :"<<m_radiotrick.getSelectedItem()<<std::endl;
        std::cin>>m_radiotrick;
        std::cout<<"Radio button :"<<m_radiotrick<<"    selectedId :"<<m_radiotrick.getSelectedId()<<"   getSelectedItem() :"<<m_radiotrick.getSelectedItem()<<std::endl;
    }
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

protected:

    unsigned int selectedItem;

    ///return the id_item of the string if found in string list button
    ///             -1    if not found
    int isInButtonList(const std::string & tempostring)
    {
        for(unsigned int i=0; i<textItems::size(); i++)
        {
            if (textItems::operator[](i)==tempostring) return i;
        }
        return -1;
    }

};

template< typename T>
inline std::istream & operator >>(std::istream & in, RadioTrick<T> & m_trick)
{m_trick.readFromStream(in); return in;}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
} // namespace helper

} // namespace sofa

#endif

