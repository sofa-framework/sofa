#ifndef SOFA_HELPER_SYSTEM_SETDIRECTORY_H
#define SOFA_HELPER_SYSTEM_SETDIRECTORY_H

#include <string>

namespace sofa
{

namespace helper
{

namespace system
{

// A small utility class to temporarly set the current directory to the same as a specified file
class SetDirectory
{
public:
    char previousDir[1024];
    std::string directory;

    SetDirectory(const char* filename);

    ~SetDirectory();

    /// Get the parent directory of a given file, i.e. if given "a/b/c", return "a/b".
    static std::string GetParentDir(const char* filename);

    /// Get the file relative to another file path, i.e. if given "../e" and "a/b/c", return "a/e".
    static std::string GetRelativeFile(const char* filename, const char* basename);
};

} // namespace system

} // namespace helper

} // namespace sofa

#endif
