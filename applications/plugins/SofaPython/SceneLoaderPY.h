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
#ifndef SCENELOADERPY_H
#define SCENELOADERPY_H

#include <SofaPython/config.h>
#include <sofa/simulation/SceneLoaderFactory.h>


#include <sofa/simulation/Visitor.h>
#include <string>
#include <map>

extern "C" {
    struct PyMethodDef;
}

namespace sofa
{

namespace simulation
{

// The scene loader/exporter for python scene files
class SOFA_SOFAPYTHON_API SceneLoaderPY : public SceneLoader
{
public:
    /// Pre-loading check
    virtual bool canLoadFileExtension(const char *extension);
    /// Pre-saving check
    virtual bool canWriteFileExtension(const char *extension);

    /// load the file
    virtual Node::SPtr load(const char *filename);

    // max: added out parameter to get the root *before* createScene is called
    void loadSceneWithArguments(const char *filename, const std::vector<std::string>& arguments=std::vector<std::string>(0), Node::SPtr* root_out = 0);
    bool loadTestWithArguments(const char *filename, const std::vector<std::string>& arguments=std::vector<std::string>(0));

    /// write the file
    virtual void write(Node* node, const char *filename);

    /// get the file type description
    virtual std::string getFileTypeDesc();

    /// get the list of file extensions
    virtual void getExtensionList(ExtensionList* list);

    /// add a header that will be injected before the loading of the scene
    static void setHeader(const std::string& header);

private:
    static std::string                          OurHeader;

};

/// Export the scene graph in Python format
void SOFA_SOFAPYTHON_API exportPython( Node* node, const char* fileName=NULL );

/// Visitor that exports all nodes/components in python
class SOFA_SOFAPYTHON_API PythonExporterVisitor : public Visitor
{

protected:

    std::ostream& m_out; ///< the output stream

    std::map<core::objectmodel::BaseNode*, std::string > m_mapNodeVariable; ///< gives a python variable name per node
    unsigned m_variableIndex; ///< unique index per node to garanty a unique variablename

public:
    PythonExporterVisitor(std::ostream& out) : Visitor(sofa::core::ExecParams::defaultInstance()), m_out(out), m_variableIndex(0) {}

    template<class T> void processObject( T obj, const std::string& nodeVariable );

    virtual Result processNodeTopDown(Node* node) override ;
    virtual void processNodeBottomUp(Node* node) override ;

    virtual const char* getClassName() const override { return "PythonExporterVisitor"; }
};




} // namespace simulation

} // namespace sofa



#endif // SCENELOADERPY_H
