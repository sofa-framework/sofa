/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_VIEWER_FILESRECENTLYOPENEDMANAGER_H
#define SOFA_GUI_VIEWER_FILESRECENTLYOPENEDMANAGER_H

#include <sofa/helper/vector.h>

#include <string>

#include "SofaGUI.h"

namespace sofa
{
namespace gui
{
class SOFA_SOFAGUI_API FilesRecentlyOpenedManager
{
public:
    FilesRecentlyOpenedManager(const std::string &configFile);
    virtual ~FilesRecentlyOpenedManager() {};

    virtual void openFile(const std::string &file);
    virtual std::string getFilename(unsigned int idx) const
    {
        if (idx < files.size()) return files[idx];
        else return std::string();
    }

    unsigned int getMaxNumFiles() const {return max_num_files;};

    const std::string &getPath() const {return path;}
    void setPath(const std::string &p);

    const sofa::helper::vector< std::string >& getFiles() const {return files;}
    void setFiles(const helper::vector< std::string > &f) {files=f; writeFiles();}

protected:
    void writeFiles() const;

    const unsigned int max_num_files;
    sofa::helper::vector< std::string > files;
    std::string path;
};


}
}

#endif
