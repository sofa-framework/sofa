#ifndef PROJECTS_GENERATEDOC_H
#define PROJECTS_GENERATEDOC_H

#include <string>

namespace projects
{

bool generateFactoryHTMLDoc(const std::string& filename);

bool generateFactoryPHPDoc(const std::string& filename, const std::string& url);

}

#endif
