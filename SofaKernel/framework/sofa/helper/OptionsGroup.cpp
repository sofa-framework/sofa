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
#include <sofa/helper/OptionsGroup.h>
#include <cstdlib>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace helper
{

class OptionsGroup;
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
OptionsGroup::OptionsGroup() : textItems()
{
    selectedItem=0;
}
///////////////////////////////////////
OptionsGroup::OptionsGroup(int nbofRadioButton,...)
{
    textItems.resize(nbofRadioButton);
    va_list vl;
    va_start(vl,nbofRadioButton);
    for(unsigned int i=0; i<textItems.size(); i++)
    {
        const char * tempochar=va_arg(vl,char *);
        assert( strcmp( tempochar, "") );
        std::string  tempostring(tempochar);
        textItems[i]=tempostring;
    }
    va_end(vl);
    selectedItem=0;
}
///////////////////////////////////////
OptionsGroup::OptionsGroup(const OptionsGroup & m_radiotrick) : textItems(m_radiotrick.textItems)
{
    selectedItem=m_radiotrick.getSelectedId();
}
///////////////////////////////////////
OptionsGroup & OptionsGroup::operator=(const OptionsGroup & m_radiotrick)
{
    textItems.resize(m_radiotrick.textItems.size());
    for(unsigned int i=0; i<textItems.size(); i++)
    {
        textItems[i]=m_radiotrick.textItems[i];
    }
    selectedItem=m_radiotrick.selectedItem;
    return *this;
}
///////////////////////////////////////
void OptionsGroup::setNbItems( unsigned int nbofRadioButton )
{
    textItems.resize( nbofRadioButton );
    selectedItem = 0;
}
///////////////////////////////////////
void OptionsGroup::setItemName( unsigned int id_item, const std::string& name )
{
    textItems[id_item] = name;
}
///////////////////////////////////////
void OptionsGroup::setNames(int nbofRadioButton,...)
{
    textItems.resize(nbofRadioButton);
    va_list vl;
    va_start(vl,nbofRadioButton);
    for(unsigned int i=0; i<textItems.size(); i++)
    {
        const char * tempochar=va_arg(vl,char *);
        std::string  tempostring(tempochar);
        textItems[i]=tempostring;
    }
    va_end(vl);
    selectedItem=0;
}
///////////////////////////////////////
int OptionsGroup::isInOptionsList(const std::string & tempostring) const
{
    for(unsigned int i=0; i<textItems.size(); i++)
    {
        if (textItems[i]==tempostring) return i;
    }
    return -1;
}
///////////////////////////////////////
void OptionsGroup::setSelectedItem(unsigned int id_item)
{
    if (id_item<textItems.size())
        selectedItem=id_item;
}
///////////////////////////////////////
void OptionsGroup::setSelectedItem(const std::string & m_string)
{
    int id_stringinButtonList = isInOptionsList(m_string);
    if (id_stringinButtonList == -1)
    {
        msg_warning("OptionsGroup") << "\""<< m_string <<"\" is not a parameter in button list :\" "<<(*this)<<"\"";
    }
    else
    {
        setSelectedItem(id_stringinButtonList);
    }
}
///////////////////////////////////////
unsigned int OptionsGroup::getSelectedId() const
{
    return selectedItem;
}
///////////////////////////////////////
const std::string& OptionsGroup::getSelectedItem() const
{
	if (textItems.empty())
	{
		static std::string empty;
		return empty;
	}
    return textItems[selectedItem];
}
///////////////////////////////////////
void OptionsGroup::readFromStream(std::istream & stream)
{
    std::string tempostring;
    std::getline(stream,tempostring);
    int id_stringinButtonList = isInOptionsList(tempostring);
    if (id_stringinButtonList == -1)
    {
        int idx=atoi(tempostring.c_str());
        if (idx >=0 && idx < (int)size()) 
			setSelectedItem(idx);
        else
            msg_warning("OptionsGroup") << "\""<< tempostring <<"\" is not a parameter in button list :\" "<<(*this)<<"\"";
    }
    else
    {
        setSelectedItem(id_stringinButtonList);
    }
}
///////////////////////////////////////
void OptionsGroup::writeToStream(std::ostream & stream) const
{
    stream << getSelectedItem();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace helper

} // namespace sofa
