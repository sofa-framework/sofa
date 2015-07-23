#include "Locale.h"

#include <sofa/helper/Logger.h>

#include <clocale>

namespace sofa {

namespace helper {

namespace system {


Locale::Locale(int category, std::string locale):
    m_category(category), m_oldValue(std::string(setlocale(category, NULL)))
{
    char *value = setlocale(category, locale.c_str());
    if (value == NULL)
        Logger::getMainLogger().log(Logger::Error, "Failed to set " + getCategoryName(category) + " to " + locale);
}

Locale::~Locale()
{
    setlocale(m_category, m_oldValue.c_str());
}

std::string Locale::getCategoryName(int category)
{
    switch(category)
    {
    case LC_ALL:
        return "LC_ALL";
    case LC_COLLATE:
        return "LC_COLLATE";
    case LC_CTYPE:
        return "LC_CTYPE";
    case LC_MESSAGES:
        return "LC_MESSAGES";
    case LC_MONETARY:
        return "LC_MONETARY";
    case LC_NUMERIC:
        return "LC_NUMERIC";
    case LC_TIME:
        return "LC_TIME";
    default:
        return "UNKNOWN";
    }
}


} // namespace system

} // namespace helper

} // sofa
