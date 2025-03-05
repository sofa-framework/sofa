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
#include <sofa/core/visual/FlagTreeItem.h>
#include <sofa/helper/logging/Messaging.h>


namespace sofa::core::visual
{
FlagTreeItem::FlagTreeItem(const std::string& showName, const std::string& hideName, FlagTreeItem* parent):
    m_showName({showName}),
    m_hideName({hideName}),
    m_state(tristate::neutral_value),
    m_parent(parent)
{
    if( m_parent ) m_parent->m_child.push_back(this);
}


void FlagTreeItem::addAliasShow(const std::string& newAlias)
{
    this->addAlias(this->m_showName, newAlias);
}


void FlagTreeItem::addAliasHide(const std::string& newAlias)
{
    this->addAlias(this->m_hideName, newAlias);
}


void FlagTreeItem::addAlias(sofa::type::vector<std::string>& name, const std::string& newAlias)
{
    name.push_back(newAlias);
    return;
}

void FlagTreeItem::getLabels(sofa::type::vector<std::string>& labels) const
{
    for (const auto& names : {m_showName, m_hideName})
    {
        for (const auto& label : names)
        {
            labels.push_back(label);
        }
    }

    for (const auto* child : m_child)
    {
        if (child)
        {
            child->getLabels(labels);
        }
    }
}


void FlagTreeItem::setValue(const tristate &state)
{
    this->m_state = state;
    propagateStateDown(this);
    propagateStateUp(this);
}

void FlagTreeItem::propagateStateDown(FlagTreeItem* origin)
{
    ChildIterator iter;
    for( iter = origin->m_child.begin(); iter != origin->m_child.end(); ++iter)
    {
        (*iter)->m_state = origin->m_state;
        propagateStateDown(*iter);
    }
}

void FlagTreeItem::propagateStateUp(FlagTreeItem* origin)
{
    FlagTreeItem* parent = origin->m_parent;
    if(!parent) return;

    tristate flag = origin->m_state;
    for( unsigned int i = 0 ; i < parent->m_child.size(); ++i)
    {
        const FlagTreeItem* current = parent->m_child[i];
        flag = fusion_tristate(current->m_state,flag);
    }

    parent->m_state=flag;
    propagateStateUp(parent);
}

std::ostream& FlagTreeItem::write(std::ostream &os) const
{
    std::string s;
    write_recursive(this,s);
    s.erase(s.find_last_not_of(" \n\r\t")+1);
    os << s;
    return os;
}

FlagTreeItem::READ_FLAG FlagTreeItem::readFlag(std::map<std::string, bool, FlagTreeItem::ci_comparison>& parseMap, std::string flag)
{
    const auto iter = parseMap.find(flag);
    if(iter != parseMap.end() )
    {
        if(iter->first != flag)
        {
            return READ_FLAG::INCORRECT_LETTER_CASE;
        }

        return READ_FLAG::KNOWN_FLAG;
    }

    return READ_FLAG::UNKNOWN_FLAG;
}

std::istream& FlagTreeItem::read(std::istream &in)
{
    return read(in, [](std::string){}, [](std::string, std::string){});
}

std::istream& FlagTreeItem::read(std::istream& in,
                                 const std::function<void(std::string)>& unknownFlagFunction,
                                 const std::function<void(std::string, std::string)>& incorrectLetterCaseFunction)
{
    std::map<std::string, bool,  ci_comparison> parse_map;
    create_parse_map(this,parse_map);
    std::string token;
    while(in >> token)
    {
        switch (readFlag(parse_map, token))
        {
            case READ_FLAG::KNOWN_FLAG:
                parse_map[token] = true;
                break;
            case READ_FLAG::INCORRECT_LETTER_CASE:
            {
                const auto iter = parse_map.find(token);
                iter->second = true;
                incorrectLetterCaseFunction(token, iter->first);
                break;
            }
            case READ_FLAG::UNKNOWN_FLAG:
                unknownFlagFunction(token);
                break;
        }
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }

    read_recursive(this,parse_map);
    return in;
}


/*static*/
void FlagTreeItem::create_parse_map(FlagTreeItem *root, std::map<std::string, bool, ci_comparison> &map)
{
    const size_t sizeShow = root->m_showName.size();
    const size_t sizeHide = root->m_hideName.size();
    for(size_t i=0; i<sizeShow; i++)
    {
        map[root->m_showName[i]] = false;
    }
    for(size_t i=0; i<sizeHide; ++i)
    {
        map[root->m_hideName[i]] = false;
    }

    ChildIterator iter;
    for( iter = root->m_child.begin(); iter != root->m_child.end(); ++iter)
    {
        create_parse_map(*iter,map);
    }
}

void FlagTreeItem::read_recursive(FlagTreeItem *root, const std::map<std::string, bool, ci_comparison> &parse_map)
{
    ChildIterator iter;
    std::map<std::string,bool>::const_iterator iter_show;
    std::map<std::string,bool>::const_iterator iter_hide;
    for( iter = root->m_child.begin(); iter != root->m_child.end(); ++iter)
    {
        const size_t sizeShow = (*iter)->m_showName.size();
        const size_t sizeHide = (*iter)->m_hideName.size();

        bool found = false;

        for(size_t i=0; i<sizeHide; i++)
        {
            iter_hide = parse_map.find((*iter)->m_hideName[i]);
            if( iter_hide != parse_map.end() )
            {
                const bool hide = iter_hide->second;
                if(hide)
                {
                    if(i != 0)
                    {
                        msg_warning("DisplayFlags") << "FlagTreeItem '" << (*iter)->m_hideName[i] << "' is deprecated, please use '"<<(*iter)->m_hideName[0]<<"' instead";
                    }

                    (*iter)->setValue(tristate::false_value);
                    found = true;
                }
            }
        }
        for(size_t i=0; i<sizeShow; i++)
        {
            iter_show = parse_map.find((*iter)->m_showName[i]);
            if( iter_show != parse_map.end() )
            {
                const bool show  = iter_show->second;
                if(show)
                {
                    if(i != 0)
                    {
                        msg_warning("DisplayFlags") << "FlagTreeItem '" << (*iter)->m_showName[i] << "' is deprecated, please use '"<<(*iter)->m_showName[0]<<"' instead";
                    }

                    (*iter)->setValue(tristate::true_value);
                    found = true;
                }
            }
        }

        if(!found)
        {
            (*iter)->m_state = tristate::neutral_value;
            read_recursive(*iter,parse_map);
        }
    }
}

void FlagTreeItem::write_recursive(const FlagTreeItem* root, std::string& str )
{
    ChildConstIterator iter;
    for( iter = root->m_child.begin(); iter != root->m_child.end(); ++iter )
    {
        switch( (*iter)->m_state.state )
        {
            case tristate::true_value:
                str.append((*iter)->m_showName[0]);
            str.append(" ");
            break;
            case tristate::false_value:
                str.append((*iter)->m_hideName[0]);
            str.append(" ");
            break;
            case tristate::neutral_value:
                write_recursive(*iter,str);
        }
    }
}

std::ostream& operator<< ( std::ostream& os, const FlagTreeItem& root )
{
    return root.write(os);
}
std::istream& operator>> ( std::istream& in, FlagTreeItem& root )
{
    return root.read(in);
}
}
