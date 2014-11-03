#include <sofa/helper/system/FileSystem.h>

#include <sofa/helper/system/Utils.h>

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
namespace FileSystem
{


#if defined(WIN32)
// Helper: call FindFirstFile, taking care of wstring to string conversion.
static HANDLE helper_FindFirstFile(std::string path, WIN32_FIND_DATA *ffd)
{
    TCHAR szDir[MAX_PATH];
    HANDLE hFind = INVALID_HANDLE_VALUE;

    // Prepare string for use with FindFile functions.  First, copy the
    // string to a buffer, then append '\*' to the directory name.
    StringCchCopy(szDir, MAX_PATH, Utils::s2ws(path).c_str());
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


bool listDirectory(const std::string& directoryPath,
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
		std::string filename = Utils::ws2s(std::wstring(ffd.cFileName));
#endif
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


bool exists(const std::string& path)
{
#if defined(WIN32)
	::SetLastError(0);
    bool pathExists = PathFileExists(Utils::s2ws(path).c_str()) != 0;
    DWORD errorCode = ::GetLastError();
	if (errorCode != 0
		&& errorCode != ERROR_FILE_NOT_FOUND
		&& errorCode != ERROR_PATH_NOT_FOUND) {
        std::cerr << "FileSystem::exists(\"" << path << "\"): "
                  << Utils::GetLastError() << std::endl;
    }
    return  pathExists;
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


bool isDirectory(const std::string& path)
{
#if defined(WIN32)
    DWORD fileAttrib = GetFileAttributes(Utils::s2ws(path).c_str());
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

bool listDirectory(const std::string& directoryPath,
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

} // namespace FileSystem
} // namespace system
} // namespace helper
} // namespace sofa

