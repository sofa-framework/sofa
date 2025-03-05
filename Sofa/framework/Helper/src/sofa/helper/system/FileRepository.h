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
#ifndef SOFA_HELPER_SYSTEM_FILEREPOSITORY_H
#define SOFA_HELPER_SYSTEM_FILEREPOSITORY_H

#include <sofa/helper/config.h>

#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <list>


namespace sofa::helper::system
{

/// Helper class to find files in a list of directories.
///
/// Each file is searched as follow:
///
/// 1: Using the specified filename in current directory, or in the specified directory.
/// If the filename does not start with "/", "./", or "../" :
/// 2: In the directory path specified using addFirstPath method.
/// 3: In the directory path specified using an environment variable (default to SOFA_DATA_PATH).
/// 4: In the default directories relative to the main executable (default to ../share).
/// 5: In the directory path specified using addLastPath method.
///
/// For file name starting with '/', './' or '../' only the first step is used.
///
/// A path is considered as a concatenation of directories separated by : on linux / mac and ; on windows
// A small utility class to temporarly set the current directory to the same as a specified file
class SOFA_HELPER_API FileRepository
{
public:
    typedef std::map< std::string, std::list<std::string> > fileKeysMap;

    /**
     * Initialize the set of paths using the environment variable SOFA_DATA_PATH as default.
     */
    FileRepository() : FileRepository("SOFA_DATA_PATH", nullptr, {}) {};

    /**
     * Initialize the set of paths using the environment variable specified by the parameter envVar.
     */
    FileRepository(const char* envVar) : FileRepository(envVar, nullptr, {}) {};

    /**
     * Initialize the set of paths using the environment variable specified by the parameter envVar and the relative path
     * specified by the parameter relativePath.
     */
    FileRepository(const char* envVar,  const char* relativePath) : FileRepository(envVar, relativePath, {}) {};

    /**
     * Initialize the set of paths using the environment variable specified by the parameter envVar and the relative paths
     * specified by the parameter paths.
     */
    FileRepository(const char* envVar,  const std::vector<std::string> & paths) : FileRepository(envVar, paths, {}) {};

    /**
     * Initialize the set of paths using the environment variable specified by the parameter envVar, the relative path
     * specified by the parameter relativePath and the ini files and respective keys specified by the parameter iniFilesAndKeys.
     */
    FileRepository(const char* envVar, const char* relativePath, const fileKeysMap& iniFilesAndKeys)
    : FileRepository(envVar, {relativePath?std::string(relativePath):""}, iniFilesAndKeys) {}

    /**
     * Initialize the set of paths using the environment variable specified by the parameter envVar, the relative paths
     * specified by the parameter paths and the ini files and respective keys specified by the parameter iniFilesAndKeys.
     */
    FileRepository(const char* envVar, const std::vector<std::string> & paths, const fileKeysMap& iniFilesAndKeys);

    ~FileRepository();

    /// Adds a path to the front of the set of paths.
    void addFirstPath(const std::string& path);

    /// Replaces every occurrences of "//" by "/"
    /// @deprecated Use FileSystem::cleanPath instead.
    static std::string cleanPath(const std::string& path);

    /// Adds a path to the back of the set of paths.
    void addLastPath(const std::string& path);

    /// Remove a path of the set of paths.
    void removePath(const std::string& path);

    /// Remove all known paths.
    void clear();

    /// Get the first path into the set of paths
    std::string getFirstPath();

    /// Returns a string such as refPath + string = path if path contains refPath.
    /// Otherwise returns path.
    static std::string relativeToPath(std::string path, std::string refPath);

    const std::vector< std::string > &getPaths() const {return vpath;}

    const std::string getPathsJoined();

    const std::string& getDirectAccessProtocolPrefix() const { return directAccessProtocolPrefix; }
    void setDirectAccessProtocolPrefix(const std::string& protocolPrefix) { directAccessProtocolPrefix = protocolPrefix; }

    /// Find file using the stored set of paths.
    /// @param basedir override current directory (optional)
    /// @param filename requested file as input, resolved file path as output
    /// @return true if the file was found in one of the directories, false otherwise
    bool findFile(std::string& filename, const std::string& basedir="", std::ostream* errlog=&std::cerr);

    /// Alias for findFile, but returning the resolved file as the result.
    /// Less informative for errors, but sometimes easier to use
    std::string getFile(std::string filename, const std::string& basedir="", std::ostream* errlog=&std::cerr)
    {
        findFile(filename, basedir, errlog);
        return filename;
    }

    /// Find file using the stored set of paths.
    /// @param basefile override current directory by using the parent directory of the given file
    /// @param filename requested file as input, resolved file path as output
    /// @return true if the file was found in one of the directories, false otherwise
    bool findFileFromFile(std::string& filename, const std::string& basefile, std::ostream* errlog=&std::cerr);

    /// Print the list of path to std::cout
    void print();


    /// OS-dependant character separating entries in list of paths.
    static char entrySeparator()
    {
#ifdef WIN32
        return ';';
#else
        return ':';
#endif
    }

    /// Display all current sofa search paths
    friend std::ostream& operator << (std::ostream& _flux, FileRepository _fr)
    {
        _flux<< "FileRepository vpath :"<<std::endl;
        for(std::vector<std::string>::iterator it = _fr.vpath.begin(); it!=_fr.vpath.end(); it++)
            _flux<<(*it)<<std::endl;

        return _flux;
    }

    void displayPaths() const {std::cout<<(*this)<<std::endl;}

    const std::string getTempPath() const;

protected:

    /// A protocol like http: or file: which will bypass the file search if found in the filename of the findFile* functions that directly returns the path as if the function succeeded
    /// Use case: add the prefix ram: as the direct protocol, this way the FileRepository will not try to look for the file on the hard disk and will directly return
    /// then the inherited FileAccess singleton enhanced with the capacity to find ram file will deliver a correct stream to this in-ram virtual file
    std::string directAccessProtocolPrefix;

    /// Vector of paths.
    std::vector<std::string> vpath;

    /// Search file in a given path.
    static bool findFileIn(std::string& filename, const std::string& path);
};

extern SOFA_HELPER_API FileRepository DataRepository; ///< Default repository
extern SOFA_HELPER_API FileRepository PluginRepository; ///< Default repository

} // namespace sofa::helper::system


#endif
