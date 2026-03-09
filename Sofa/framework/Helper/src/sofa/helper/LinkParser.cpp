#include <sofa/helper/LinkParser.h>
#include <sofa/helper/StringUtils.h>

namespace sofa::helper
{
sofa::helper::LinkParser::LinkParser(std::string linkString)
    : m_initialLinkString(std::move(linkString)), m_linkString(m_initialLinkString)
{
}
bool LinkParser::hasPrefix() const { return !m_linkString.empty() && m_linkString[0] == prefix; }

bool LinkParser::isAbsolute() const
{
    if (m_linkString.size() > 1)
    {
        return m_linkString[1] == separator;
    }
    return false;
}

LinkParser& LinkParser::cleanLink()
{
    m_linkString = sofa::helper::removeLeadingCharacter(m_linkString, ' ');

    //replace backslash by slash
    sofa::helper::replaceAll(m_linkString, "\\", std::string(1, separator));

    //replace double slash by single slash
    while (m_linkString.find("//") != std::string::npos)
    {
        sofa::helper::replaceAll(m_linkString, "//", std::string(1, separator));
    }

    auto decomposition = this->split();

    if (decomposition.empty())
        return *this;

    // a "." references itself, so it can be removed
    auto it = std::find(decomposition.begin(), decomposition.end(), ".");
    while (it != decomposition.end())
    {
        decomposition.erase(it);
        it = std::find(decomposition.begin(), decomposition.end(), ".");
    }

    m_linkString = this->join(decomposition.begin(), decomposition.end(), isAbsolute(), hasPrefix());

    return *this;
}
void LinkParser::validate()
{
    cleanLink();

    if (!hasPrefix())
    {
        addError("Parsing link '" + m_linkString + "' failed: missing prefix '@'");
    }

    const auto decomposition = this->split();

    for (const auto& element : decomposition)
    {
        if (!element.empty())
        {
            if (element.front() == '[')
            {
                if (element.back() != ']')
                {
                    addError("The element '" + element + "' from link '" + m_initialLinkString +
                        "' is not valid: it starts with '[' but does not end with ']'");
                }
                else
                {
                    addError("The element '" + element + "' from link '" + m_initialLinkString +
                        "' is not valid: index references are not supported");
                }
            }
        }
    }
}

std::string LinkParser::getLink() const
{
    return m_linkString;
}

sofa::type::vector<std::string> LinkParser::split() const
{
    if (m_linkString.empty())
        return {};

    std::string withoutPrefix = m_linkString;
    if (m_linkString[0] == prefix)
    {
        withoutPrefix = m_linkString.substr(1);
    }

    auto decomposition = sofa::helper::split(withoutPrefix, separator);
    std::erase_if(decomposition, [](const std::string& path) { return path.empty(); });

    return decomposition;
}
std::vector<std::string> LinkParser::getErrors() const
{
    std::vector<std::string> errors;
    for (const auto& error : m_errors)
    {
        errors.push_back(error);
    }
    return errors;
}

void LinkParser::addError(std::string s)
{
    m_errors.push_back(s);
}

}  // namespace sofa::helper
