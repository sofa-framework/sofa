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
#ifndef SOFAEDITOR_BASESOFAEDITOR_H

#include <vector>
#include <string>
#include <limits>

#include <SofaEditor/config.h>
#include <memory>

namespace sofaeditor
{
    /// The SofaEditorStatic is holding information related to the editors
    /// The informations can then be communicated to sofa component or controllers.
    ///
    /// Thread safetyness:
    ///     The object is not thread safe, given the presence of getSelection it seems
    ///     hard to make it so and preserving the API. A different class design may be needed.
    ///
    class SOFAEDITOR_API SofaEditorState
    {
    public:
        /// The name that identify the editor.
        std::string editorname ;

        /// Create a new editor state with a given 'name' so multiple editor can coexist
        /// by having different names;
        SofaEditorState(const std::string& name="");
        ~SofaEditorState();

        /// Set the current selection from a vector of string contains the absolute paths to the
        /// selected objects.
        void setSelectionFromPath(const std::vector<std::string>& paths );

        /// get the current selection as a vector of absolute paths.
        const std::vector<std::string>& getSelection() const ;
    private:
        std::vector<std::string> m_currentSelection;
    };

    /// A static API that allows to share editors properties. The general idea is that each editors
    /// can attach its properties to an ID that is stored in this object.
    ///
    /// Thread safetyness:
    ///     The class is not (yet) thread safe. To make it such is relatively easy, each static
    ///     function should be protected by a mutex.
    ///
    /// Example of use:
    /// In the editor:
    ///     SofaEditorState* s = new SofaEditorState("main");
    ///     SofaEditor::ID id = SofaEditor:createId(s);
    ///     s->setSelectionFromPath({"/root/child1/object1","root/child2/object2"})
    ///
    /// Somewhere else:
    ///     SofaEditor::ID id = SofaEditor::getIdFromEditorName("main");
    ///     SofaEditorState* state = SofaEditor::getState(id);
    ///     state->getSelection();
    class SOFAEDITOR_API SofaEditor
    {
    public:
        typedef size_t ID;

        /// Indicates an invalid ID.
        const static ID InvalidID;

        /// Returns the first editor matching the give editor's name
        /// returns InvalidId is none is found.
        static ID getIdFromEditorName(const std::string& s);

        static ID createIdAndAttachState(std::shared_ptr<SofaEditorState>& s);
        static bool attachState(ID editorId, std::shared_ptr<SofaEditorState>& s);
        static std::shared_ptr<SofaEditorState> getState(ID editorId=0);

    private:
        static std::vector<std::shared_ptr<SofaEditorState>> s_editorsstate ;
    };
}

#endif // SOFAEDITOR_BASESOFAEDITOR_H
