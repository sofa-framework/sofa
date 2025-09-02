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
#include <sofa/helper/OptionsGroup.h>
#include <cstdlib>
#include <sofa/helper/logging/Messaging.h>

namespace sofa::helper
{

class OptionsGroup;
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
OptionsGroup::OptionsGroup() : textItems()
{
    selectedItem=0;
}
///////////////////////////////////////
OptionsGroup::OptionsGroup(const OptionsGroup & m_radiotrick) : textItems(m_radiotrick.textItems)
{
    selectedItem = m_radiotrick.getSelectedId();
}
///////////////////////////////////////
void OptionsGroup::setNbItems(const size_type nbofRadioButton )
{
    textItems.resize( nbofRadioButton );
    selectedItem = 0;
}
///////////////////////////////////////
type::vector<std::string> OptionsGroup::getItemNames()
{
    return textItems;
}
///////////////////////////////////////
void OptionsGroup::setItemName(const unsigned int id_item, const std::string& name )
{
    if(id_item < textItems.size())
    {
        textItems[id_item] = name;
    }
}
///////////////////////////////////////
int OptionsGroup::isInOptionsList(const std::string & tempostring) const
{
    for(std::size_t i=0; i<textItems.size(); i++)
    {
        if (textItems[i]==tempostring) return i;
    }
    return -1;
}
///////////////////////////////////////
OptionsGroup& OptionsGroup::setSelectedItem(unsigned int id_item)
{
    if (id_item < textItems.size())
        selectedItem = id_item;
    return *this;
}
///////////////////////////////////////
OptionsGroup& OptionsGroup::setSelectedItem(const std::string & m_string)
{
    const int id_stringinButtonList = isInOptionsList(m_string);
    if (id_stringinButtonList == -1)
    {
        msg_warning("OptionsGroup") << "\""<< m_string <<"\" is not a parameter in button list :\" "<<(*this)<<"\"";
    }
    else
    {
        setSelectedItem(id_stringinButtonList);
    }
    return *this;
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
    const int id_stringinButtonList = isInOptionsList(tempostring);
    if (id_stringinButtonList == -1)
    {
        const int idx=atoi(tempostring.c_str());
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
} // namespace sofa::helper
