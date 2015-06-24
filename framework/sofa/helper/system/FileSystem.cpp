#include <sofa/helper/system/FileSystem.h>

#include <sofa/helper/Utils.h>

#include <fstream>
#include <iostream>
#ifdef WIN32
# include <windows.h>
# include <strsafe.h>
# include "Shlwapi.h"           // for PathFileExists()
#elif defined(_XBOX)
# include <xtl.h>
#else
# include <dirent.h>
# include <sys/stat.h>
# include <sys/types.h>
# include <errno.h>
# include <string.h>            // for strerror()
#endif

namespace sofa
{
namespace helper
{
namespace system
{


#if defined(WIN32)
// Helper: call FindFirstFile, taking care of wstring to string conversion.
static HANDLE helper_FindFirstFile(std::string path, WIN32_FIND_DATA *ffd)
{
    TCHAR szDir[MAX_PATH];
    HANDLE hFind = INVALID_HANDLE_VALUE;

    // Prepare string for use with FindFile functions.  First, copy the
    // string to a buffer, then append '\*' to the directory name.
    StringCchCopy(szDir, MAX_PATH, Utils::widenString(path).c_str());
    StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

    // Find the first file in the directory.
    hFind = FindFirstFile(szDir, ffd);

    return hFind;
}
#elif defined (_XBOX)
static HANDLE helper_FindFirstFile(std::string path, WIN32_FIND_DATA *ffd)
{
    char szDir[MAX_PATH];
    HANDLE hFind = INVALID_HANDLE_VALUE;

    // Prepare string for use with FindFile functions.  First, copy the
    // string to a buffer, then append '\*' to the directory name.
	strcpy_s(szDir, MAX_PATH, path.c_str());
	strcat_s(szDir, MAX_PATH, "\\*");

    // Find the first file in the directory.
    hFind = FindFirstFile(szDir, ffd);

    return hFind;
}
#endif


bool FileSystem::listDirectory(const std::string& directoryPath,
                               std::vector<std::string>& outputFilenames)
{
#if defined(WIN32) || defined (_XBOX)
    // Find the first file in the directory.
    WIN32_FIND_DATA ffd;
    HANDLE hFind = helper_FindFirstFile(directoryPath, &ffd);
    if (hFind == INVALID_HANDLE_VALUE) {
        std::cerr << "FileSystem::listDirectory(\"" << directoryPath << "\"): "
                  << Utils::GetLastError() << std::endl;
        return false;
    }

    // Iterate over files and push them in the output vector
    do {
# if defined (_XBOX)
		std::string filename = ffd.cFileName;
# else
		std::string filename = Utils::narrowString(ffd.cFileName);
# endif
        if (filename != "." && filename != "..")
			outputFilenames.push_back(filename);
    } while (FindNextFile(hFind, &ffd) != 0);

    // Check for errors
    bool errorOccured = ::GetLastError() != ERROR_NO_MORE_FILES;
    if (errorOccured)
        std::cerr << "FileSystem::listDirectory(\"" << directoryPath << "\"): "
                  << Utils::GetLastError() << std::endl;

    FindClose(hFind);
    return errorOccured;
#else
    DIR *dp = opendir(directoryPath.c_str());
    if (dp == NULL) {
        std::cerr << "FileSystem::listDirectory(\"" << directoryPath << "\"): "
                  << strerror(errno) << std::endl;
        return true;
    }

    struct dirent *ep;
    while ( (ep = readdir(dp)) ) {
        const std::string filename(ep->d_name);
        if (filename != "." && filename != "..")
            outputFilenames.push_back(std::string(ep->d_name));
    }
    closedir(dp);
    return false;
#endif
}


bool FileSystem::exists(const std::string& path)
{
#if defined(WIN32)
	::SetLastError(0);
    if (PathFileExists(Utils::widenString(path).c_str()) != 0)
        return true;
    else
    {
        DWORD errorCode = ::GetLastError();
        if (errorCode != 2) // not No such file error
            std::cerr << "FileSystem::exists(\"" << path << "\"): "
                << Utils::GetLastError() << std::endl;
         return false;
    }

#elif defined (_XBOX)
	DWORD fileAttrib = GetFileAttributes(path.c_str());
    return fileAttrib != -1;
#else
    struct stat st_buf;
    if (stat(path.c_str(), &st_buf) == 0)
        return true;
    else
        if (errno == ENOENT)    // No such file or directory
            return false;
        else {
            std::cerr << "FileSystem::exists(\"" << path << "\"): "
                      << strerror(errno) << std::endl;
            return false;
        }
#endif
}


bool FileSystem::isDirectory(const std::string& path)
{
#if defined(WIN32)
    DWORD fileAttrib = GetFileAttributes(Utils::widenString(path).c_str());
    if (fileAttrib == INVALID_FILE_ATTRIBUTES) {
        std::cerr << "FileSystem::isDirectory(\"" << path << "\"): "
                  << Utils::GetLastError() << std::endl;
        return false;
    }
    else
        return (fileAttrib & FILE_ATTRIBUTE_DIRECTORY) != 0;
#elif defined (_XBOX)
    DWORD fileAttrib = GetFileAttributes(path.c_str());
    if (fileAttrib == -1) {
        std::cerr << "FileSystem::isDirectory(\"" << path << "\"): "
                  << Utils::GetLastError() << std::endl;
        return false;
    }
    else
        return (fileAttrib & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat st_buf;
    if (stat(path.c_str(), &st_buf) != 0) {
        std::cerr << "FileSystem::isDirectory(\"" << path << "\"): "
                  << strerror(errno) << std::endl;
        return false;
    }
    else
        return S_ISDIR(st_buf.st_mode);
#endif
}

bool FileSystem::listDirectory(const std::string& directoryPath,
                               std::vector<std::string>& outputFilenames,
                               const std::string& extension)
{
    // List directory
    std::vector<std::string> files;
    if (listDirectory(directoryPath, files))
        return true;

    // Filter files
    for (std::size_t i=0 ; i!=files.size() ; i++) {
        const std::string& filename = files[i];
        if (filename.size() >= extension.size())
            if (filename.compare(filename.size()-extension.size(),
                                 std::string::npos, extension) == 0)
                outputFilenames.push_back(filename);
    }
    return false;
}

static bool pathHasDrive(const std::string& path) {
    return path.length() >=3
        && ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z'))
        && path[1] == ':';
}

static std::string pathWithoutDrive(const std::string& path) {
    return path.substr(2, std::string::npos);
}

static std::string pathDrive(const std::string& path) {
    return path.substr(0, 2);
}

bool FileSystem::isAbsolute(const std::string& path)
{
    return !path.empty()
        && (pathHasDrive(path)
            || path[0] == '/');
}

std::string FileSystem::convertBackSlashesToSlashes(const std::string& path)
{
    std::string str = path;
    size_t backSlashPos = str.find('\\');
    while(backSlashPos != std::string::npos)
    {
        str[backSlashPos] = '/';
        backSlashPos = str.find("\\");
    }
    return str;
}

std::string FileSystem::removeExtraSlashes(const std::string& path)
{
    std::string str = path;
    size_t pos = str.find("//");
    while(pos != std::string::npos) {
        str.replace(pos, 2, "/");
        pos = str.find("//");
    }
    return str;
}

std::string FileSystem::cleanPath(const std::string& path)
{
    return removeExtraSlashes(convertBackSlashesToSlashes(path));
}

static std::string computeParentDirectory(const std::string& path)
{
    if (path == "")
        return ".";
    else if (path == "/")
        return "/";
    else if (path[path.length()-1] == '/')
        return computeParentDirectory(path.substr(0, path.length() - 1));
    else {
        size_t last_slash = path.find_last_of('/');
        if (last_slash == std::string::npos)
            return ".";
        else if (last_slash == 0)
            return "/";
        else if (last_slash == path.length())
            return "";
        else
            return path.substr(0, last_slash);
    }
}

std::string FileSystem::getParentDirectory(const std::string& path)
{
    if (pathHasDrive(path))     // check for Windows drive
        return pathDrive(path) + computeParentDirectory(pathWithoutDrive(path));
    else
        return computeParentDirectory(path);
}

std::string FileSystem::stripDirectory(const std::string& path)
{
    if (pathHasDrive(path))     // check for Windows drive
        return stripDirectory(pathWithoutDrive(path));
    else
    {
        size_t last_slash = path.find_last_of("/");
        if (last_slash == std::string::npos)    // No slash
            return path;
        else if (last_slash == path.size() - 1) // Trailing slash
            if (path.size() == 1)
                return "/";
            else
                return stripDirectory(path.substr(0, path.size() - 1));
        else
            return path.substr(last_slash + 1, std::string::npos);
        return "";
    }
}


} // namespace system
} // namespace helper
} // namespace sofa

