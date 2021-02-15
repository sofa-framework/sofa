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
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>

#include <sys/types.h>
#include <sys/stat.h>
#if defined(WIN32)
#include <windows.h>
#include <direct.h>
#else
#include <unistd.h>
#endif

#include <boost/filesystem.hpp>

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/Utils.h>
#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem;

#ifdef WIN32
#define ON_WIN32 true
#else
#define ON_WIN32 false
#endif // WIN32

namespace sofa
{

namespace helper
{

namespace system
{
// replacing every occurences of "//"  by "/"
std::string cleanPath( const std::string& path )
{
    std::string p = path;
    size_t pos = p.find("//");
    size_t len = p.length();
    while( pos != std::string::npos )
    {
        if ( pos == (len-1))
            p.replace( pos, 2, "");
        else
            p.replace(pos,2,"/");
        pos = p.find("//");
    }
    return p;
}

// Initialize PluginRepository and DataRepository
#ifdef WIN32
FileRepository PluginRepository(
    "SOFA_PLUGIN_PATH",
    {
        Utils::getSofaPathTo("plugins"),
        Utils::getSofaPathTo("bin"),
        Utils::getExecutableDirectory(),
    }
);
#else
FileRepository PluginRepository(
    "SOFA_PLUGIN_PATH",
    {
        Utils::getSofaPathTo("plugins"),
        Utils::getSofaPathTo("lib"),
    }
);
#endif
FileRepository DataRepository(
    "SOFA_DATA_PATH",
    {
        Utils::getSofaPathTo("share/sofa"),
        Utils::getSofaPathTo("share/sofa/examples")
    },
    {
        { Utils::getSofaPathTo("etc/sofa.ini"), {"SHARE_DIR", "EXAMPLES_DIR"} }
    }
);

FileRepository::FileRepository(const char* envVar, const std::vector<std::string> & paths, const fileKeysMap& iniFilesAndKeys) {
    if (envVar != nullptr && envVar[0]!='\0')
    {
        const char* envpath = getenv(envVar);
        if (envpath != nullptr && envpath[0]!='\0')
            addFirstPath(envpath);
    }

    for (const auto & path : paths) {
        if (!path.empty()) {
            size_t p0 = 0;
            while (p0 < path.size()) {
                size_t p1 = path.find(entrySeparator(), p0);
                if (p1 == std::string::npos) p1 = path.size();
                if (p1 > p0 + 1) {
                    std::string p = path.substr(p0, p1 - p0);
                    addLastPath(SetDirectory::GetRelativeFromProcess(p.c_str()));
                }
                p0 = p1 + 1;
            }
        }
    }
    if ( !iniFilesAndKeys.empty() )
    {
        for ( const auto &iniFileAndKeys : iniFilesAndKeys )
        {
            const std::string& file = iniFileAndKeys.first;
            const std::list<std::string>& keys = iniFileAndKeys.second;

            std::map<std::string, std::string> iniFileLines = Utils::readBasicIniFile(file);
            for ( const auto &iniFileLine : iniFileLines )
            {
                const std::string& lineKey = iniFileLine.first;
                const std::string& lineDir = iniFileLine.second;

                if ( std::find(keys.begin(), keys.end(), lineKey) == keys.end() )
                {
                    // The key on this line is not one of the keys searched in this file
                    continue;
                }

                const std::string& absoluteDir = SetDirectory::GetRelativeFromProcess(lineDir.c_str());
                if ( FileSystem::exists(absoluteDir) && FileSystem::isDirectory(absoluteDir) )
                {
                    addFirstPath(absoluteDir);
                }
            }
        }
    }
}

FileRepository::~FileRepository()
{
}

std::string FileRepository::cleanPath(const std::string& path)
{
    msg_deprecated("FileRepository::cleanPath") << "Use FileSystem::cleanPath instead.";
    return FileSystem::cleanPath(path);
}

void FileRepository::addFirstPath(const std::string& p)
{
    // replacing every occurences of "//" by "/"
    std::string path = FileSystem::cleanPath(p);

    std::vector<std::string> entries;
    size_t p0 = 0;
    while ( p0 < path.size() )
    {
        size_t p1 = path.find(entrySeparator(),p0);
        if (p1 == std::string::npos) p1 = path.size();
        if (p1>p0+1)
        {
            entries.push_back(path.substr(p0,p1-p0));
        }
        p0 = p1+1;
    }
    vpath.insert(vpath.begin(), entries.begin(), entries.end());
}

void FileRepository::addLastPath(const std::string& p)
{
    // replacing every occurences of "//" by "/"
    std::string path = FileSystem::cleanPath(p);

    std::vector<std::string> entries;
    size_t p0 = 0;
    while ( p0 < path.size() )
    {
        size_t p1 = path.find(entrySeparator(),p0);
        if (p1 == std::string::npos) p1 = path.size();
        if (p1>p0+1)
        {
            entries.push_back(path.substr(p0,p1-p0));
        }
        p0 = p1+1;
    }
    vpath.insert(vpath.end(), entries.begin(), entries.end());
}

void FileRepository::removePath(const std::string& path)
{
    std::vector<std::string> entries;
    size_t p0 = 0;
    while ( p0 < path.size() )
    {
        size_t p1 = path.find(entrySeparator(),p0);
        if (p1 == std::string::npos) p1 = path.size();
        if (p1>p0+1)
        {
            entries.push_back(path.substr(p0,p1-p0));
        }
        p0 = p1+1;
    }

    for(std::vector<std::string>::iterator it=entries.begin();
        it!=entries.end(); ++it)
    {
        vpath.erase( find(vpath.begin(), vpath.end(), *it) );
    }
}

void FileRepository::clear()
{
    vpath.clear();
}

std::string FileRepository::getFirstPath()
{
    if (vpath.size() > 0)
        return vpath.front();
    else return "";
}

bool FileRepository::findFileIn(std::string& filename, const std::string& path)
{
    if (filename.empty()) return false; // no filename
    std::string newfname = SetDirectory::GetRelativeFromDir(filename.c_str(), path.c_str());
    std::locale locale;
    try
    {
        locale = std::locale("en_US.UTF-8");
    }
    catch (std::exception e)
    {
        locale = std::locale("");
    }
    boost::filesystem::path::imbue(locale);
    boost::filesystem::path p(newfname);
    if (boost::filesystem::exists(p))
    {
        // File found
        filename = newfname;
        return true;
    }
    return false;
}

bool FileRepository::findFile(std::string& filename, const std::string& basedir, std::ostream* errlog)
{
    if (filename.empty()) return false; // no filename
    if (!directAccessProtocolPrefix.empty() && filename.substr(0, directAccessProtocolPrefix.size()) == directAccessProtocolPrefix)
        return true;

    std::string currentDir = SetDirectory::GetCurrentDir();
    if (!basedir.empty())
    {
        currentDir = SetDirectory::GetRelativeFromDir(basedir.c_str(),currentDir.c_str());
    }
    if (findFileIn(filename, currentDir)) return true;

    if (SetDirectory::IsAbsolute(filename)) return false; // absolute file path
    if (filename.substr(0,2)=="./" || filename.substr(0,3)=="../")
    {
        // update filename with current dir
        filename = SetDirectory::GetRelativeFromDir(filename.c_str(), currentDir.c_str());
        return false; // local file path
    }
    for (std::vector<std::string>::const_iterator it = vpath.begin(); it != vpath.end(); ++it)
        if (findFileIn(filename, *it)) return true;
    if (errlog)
    {
        // hack to use logging rather than directly writing in std::cerr/std::cout
        // @todo do something cleaner

        std::stringstream tmplog;
        tmplog << "File "<<filename<<" NOT FOUND in "<<basedir;
        for (std::vector<std::string>::const_iterator it = vpath.begin(); it != vpath.end(); ++it)
            tmplog << ':'<<*it;
        if( errlog==&std::cerr || errlog==&std::cout)
            msg_error("FileRepository") << tmplog.str();
        else
            (*errlog)<<tmplog.str()<<std::endl;
    }
    return false;
}

bool FileRepository::findFileFromFile(std::string& filename, const std::string& basefile, std::ostream* errlog)
{
    return findFile(filename, SetDirectory::GetParentDir(basefile.c_str()), errlog);
}

void FileRepository::print()
{
    for (std::vector<std::string>::const_iterator it = vpath.begin(); it != vpath.end(); ++it)
        std::cout << *it << std::endl;
}

const std::string FileRepository::getPathsJoined()
{
    std::ostringstream imploded;
    std::string delim = ":";
#ifdef WIN32
    delim = ";";
#endif
    std::copy(vpath.begin(), vpath.end(), std::ostream_iterator<std::string>(imploded, delim.c_str()));
    std::string implodedStr = imploded.str();
    implodedStr = implodedStr.substr(0, implodedStr.size()-1); // remove trailing separator
    return implodedStr;
}

/*static*/
std::string FileRepository::relativeToPath(std::string path, std::string refPath, bool doLowerCaseOnWin32)
{
    /// This condition replace the #ifdef in the code.
    /// The advantage is that the code is compiled and is
    /// removed by the optimization pass.
    if( ! ON_WIN32 )
    {
        /// Case sensitive OS.
        std::string::size_type loc = path.find( refPath, 0 );
        if (loc==0)
            path = path.substr(refPath.size()+1);

        return path;
    }

    /// WIN32 is a pain here because of mixed case formatting with randomly
    /// picked slash and backslash to separate dirs.
    ///TODO(dmarchal 2017-05-01): remove the deprecated part in one year.
    std::string tmppath;
    std::replace(path.begin(),path.end(),'\\' , '/' );
    std::replace(refPath.begin(),refPath.end(),'\\' , '/' );

    std::transform(refPath.begin(), refPath.end(), refPath.begin(), ::tolower );

    if( doLowerCaseOnWin32 )
    {
        dmsg_deprecated("FileRepository") << "Using relativePath(doLowerCaseOnWin32=true) is a deprecated behavior since 2017-05-01. "
                                             "This behavior will be removed in 2018-05-01 and you need to update your code to use new "
                                             "API";
        /// The complete path is lowered... and copied into the tmppath;
        std::transform(path.begin(), path.end(), path.begin(), ::tolower );
        tmppath = path ;
    }else{
        tmppath = path ;
        std::transform(tmppath.begin(), tmppath.end(), tmppath.begin(), ::tolower );
    }

    std::string::size_type loc = tmppath.find( refPath, 0 );
    if (loc==0)
        path = path.substr(refPath.size()+1);

    return path;
}

const std::string FileRepository::getTempPath() const
{
    // TODO: replace std::filesystem when all compilers support it. (and not std::experimental::filesystem)
    return boost::filesystem::temp_directory_path().string();
}

} // namespace system

} // namespace helper

} // namespace sofa

