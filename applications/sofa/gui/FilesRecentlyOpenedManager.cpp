/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/gui/FilesRecentlyOpenedManager.h>

#include <sofa/helper/system/FileRepository.h>
#include <fstream>
#include <algorithm>


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

void FilesRecentlyOpenedManager::setPath(const std::string &configFile)
{
    path=configFile;
    if ( !sofa::helper::system::DataRepository.findFile ( path ) )
    {
        path = sofa::helper::system::DataRepository.getFirstPath() + "/" + configFile;

        std::ofstream ofile(path.c_str());
        ofile << "";
        ofile.close();
    }
    else path = sofa::helper::system::DataRepository.getFile ( configFile );


    //Open the file containing the list of previously opened files
    files.clear();

    std::ifstream filesStream(path.c_str());
    std::string filename;
    while (std::getline(filesStream, filename)) files.push_back(sofa::helper::system::DataRepository.getFile(filename));
    filesStream.close();
};

void FilesRecentlyOpenedManager::writeFiles() const
{
    std::ofstream out(path.c_str(),std::ios::out);
    for (unsigned int i=0; i<files.size(); ++i) out << files[i] << "\n";
    out.close();
}

void FilesRecentlyOpenedManager::openFile(const std::string &file)
{
    //Verify the existence of the file
    std::string fileLoaded(file);
    if (file.empty() || !sofa::helper::system::DataRepository.findFile(fileLoaded))
        return;

    fileLoaded=sofa::helper::system::DataRepository.getFile(file);

    //Reformat for Windows
#ifdef WIN32
    //Remove all occurences of '\\' in the path
    std::replace(fileLoaded.begin(), fileLoaded.end(), '\\', '/');
#endif

    //Remove previous occurence of the file
    helper::vector<std::string>::iterator fileFound=std::find(files.begin(),files.end(), fileLoaded);
    if (fileFound != files.end()) files.erase(fileFound);
    //Add the current file to the list
    helper::vector<std::string>::iterator front=files.begin();
    files.insert(front, fileLoaded);
    //Only keep a given number of files
    if (files.size() > max_num_files) files.resize(max_num_files);

    writeFiles();
}

}
}

