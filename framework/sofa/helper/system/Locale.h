#ifndef HELPER_SYSTEM_LOCALE_H
#define HELPER_SYSTEM_LOCALE_H

#include <sofa/helper/helper.h>

#include <clocale>
#include <string>

namespace sofa {

namespace helper {

namespace system {


class SOFA_HELPER_API Locale
{
public:
    static std::string getCategoryName(int category);
};


/// RAII class to modify the locale temporarily.
class SOFA_HELPER_API TemporaryLocale
{
private:
    int m_category;
    std::string m_oldValue;
public:
    TemporaryLocale(int category, std::string locale);
    ~TemporaryLocale();
};


} // namespace system

} // namespace helper

} // sofa

#endif
