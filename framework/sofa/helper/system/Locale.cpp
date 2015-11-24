#include "Locale.h"
#include <sofa/helper/logging/Messaging.h>
#include <clocale>

namespace sofa {

namespace helper {

namespace system {


TemporaryLocale::TemporaryLocale(int category, std::string locale):
    m_category(category), m_oldValue(std::string(setlocale(category, NULL)))
{
    char *value = setlocale(category, locale.c_str());
    if (value == NULL)
        msg_error("TemporaryLocale") << "Failed to set " << Locale::getCategoryName(category) << " to " << locale;
}

TemporaryLocale::~TemporaryLocale()
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
#if WIN32
#if (_MSC_VER < 1800)	// visual studio >= 2013 does not recognize LC_MESSAGES
    case LC_MESSAGES:
        return "LC_MESSAGES";
#endif
#else
	case LC_MESSAGES:
		return "LC_MESSAGES";
#endif
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
