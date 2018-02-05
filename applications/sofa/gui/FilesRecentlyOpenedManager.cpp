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

#include <sofa/gui/FilesRecentlyOpenedManager.h>

#include <sofa/helper/system/FileSystem.h>
#include <fstream>
#include <algorithm>

using sofa::helper::system::FileSystem;

namespace sofa
{
namespace gui
{

FilesRecentlyOpenedManager::FilesRecentlyOpenedManager(const std::string &configFile):
    max_num_files(10)
{
    path = configFile;
    setPath(configFile);
}

void FilesRecentlyOpenedManager::setPath(const std::string &path)
{
    // File does not exist? Create an empty one.
    if (!FileSystem::exists(path))
    {
        std::ofstream ofile(path.c_str());
        ofile << "";
        ofile.close();
    }

    files.clear();

    std::ifstream filesStream(path.c_str());
    std::string filePath;
    while (std::getline(filesStream, filePath))
        files.push_back(filePath);
    filesStream.close();
};

void FilesRecentlyOpenedManager::writeFiles() const
{
    std::ofstream out(path.c_str(),std::ios::out);
    for (unsigned int i=0; i<files.size(); ++i) out << files[i] << "\n";
    out.close();
}

void FilesRecentlyOpenedManager::openFile(const std::string &path)
{
    // Verify the existence of the file
    if (path.empty() || !FileSystem::exists(path))
        return;

    // Remove previous occurence of the file, if any
    helper::vector<std::string>::iterator fileFound = std::find(files.begin(), files.end(), path);
    if (fileFound != files.end())
        files.erase(fileFound);

    // Add the current file to the list
    helper::vector<std::string>::iterator front=files.begin();
    files.insert(front, path);

    // Only keep a given number of files
    if (files.size() > max_num_files)
        files.resize(max_num_files);

    writeFiles();
}

}
}

