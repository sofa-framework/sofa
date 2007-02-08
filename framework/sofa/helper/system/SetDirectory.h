#ifndef SOFA_HELPER_SYSTEM_SETDIRECTORY_H
#define SOFA_HELPER_SYSTEM_SETDIRECTORY_H

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
    char* directory;

    SetDirectory(const char* filename);

    ~SetDirectory();
};

} // namespace system

} // namespace helper

} // namespace sofa

#endif
