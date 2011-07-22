#ifndef SOFA_HELPER_SYSTEM_DYNAMICLIBRARY_H
#define SOFA_HELPER_SYSTEM_DYNAMICLIBRARY_H


#include <iostream>

namespace sofa
{
namespace helper
{
namespace system
{

class DynamicLibrary
{
public:

    static DynamicLibrary * load(const std::string & path,
            std::ostream* errlog=&std::cerr);
    ~DynamicLibrary();

    void * getSymbol(const std::string & name,
            std::ostream* errlog=&std::cerr);
    static const char* getExtension();
    static const char* getSuffix();
private:
    DynamicLibrary();

    DynamicLibrary(const std::string& name, void * handle );
    DynamicLibrary(const DynamicLibrary &);

private:
    void * m_handle;
    std::string m_name;
};


}

}

}

#endif // SOFA_HELPER_SYSTEM_DYNAMICLIBRARY_H
