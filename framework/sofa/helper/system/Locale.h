#ifndef HELPER_SYSTEM_LOCALE_H
#define HELPER_SYSTEM_LOCALE_H

#include <sofa/helper/helper.h>

#include <clocale>
#include <string>

namespace sofa {

namespace helper {

namespace system {


/// RAII class to modify the locale temporarily.
class SOFA_HELPER_API Locale
{
private:
    int m_category;
    std::string m_oldValue;
public:
    Locale(int category, std::string locale);
    ~Locale();

    static std::string getCategoryName(int category);
};


} // namespace system

} // namespace helper

} // sofa

#endif
