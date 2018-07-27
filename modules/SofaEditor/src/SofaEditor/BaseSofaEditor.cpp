/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
/*****************************************************************************
 * Contributors:
 *     - damien.marchal@univ-lille1.fr
 ****************************************************************************/

#include <algorithm>

#include <SofaEditor/BaseSofaEditor.h>
using sofaeditor::SofaEditorState;

namespace sofaeditor
{

SofaEditorState::SofaEditorState(const std::string& name)
{
    editorname = name ;
}

SofaEditorState::~SofaEditorState()
{
}

void SofaEditorState::setSelectionFromPath( const std::vector<std::string>& paths )
{
    m_currentSelection = paths;
}

const std::vector<std::string>& SofaEditorState::getSelection() const
{
    return m_currentSelection;
}

const SofaEditor::ID SofaEditor::InvalidID = std::numeric_limits<ID>::max() ;

SofaEditor::ID SofaEditor::getIdFromEditorName(const std::string& s)
{
   auto res = std::find_if(s_editorsstate.begin(), s_editorsstate.end(),
                           [&s](const SofaEditorState* item) {
                            if(item==nullptr)
                                return false;
                            return item->editorname == s;
                        });
   if(res == s_editorsstate.end())
        return SofaEditor::InvalidID;
   return static_cast<ID>( res - s_editorsstate.begin() );
}

SofaEditor::ID SofaEditor::createId(const SofaEditorState* s)
{
    s_editorsstate.push_back(s);
    return s_editorsstate.size() - 1;
}

bool SofaEditor::attachState(ID editorId, const SofaEditorState* s)
{
    if(editorId < s_editorsstate.size())
    {
        s_editorsstate[editorId] = s;
        return true;
    }
    return false;
}

const SofaEditorState* SofaEditor::getState(ID editorId)
{
    if(editorId < s_editorsstate.size())
        return s_editorsstate[editorId];
    return nullptr;
}

std::vector<const SofaEditorState*> SofaEditor::s_editorsstate ;

} /// namespace sofaditor


