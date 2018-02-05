/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "generateDoc.h"

#include <fstream>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/CategoryLibrary.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/Colors.h>

using namespace sofa::core;

namespace projects
{

bool generateFactoryHTMLDoc(const std::string& filename)
{
    //ObjectFactory::getInstance()->dump();
    std::ofstream ofile(filename.c_str());
    ofile << "<html><body>\n";
    ObjectFactory::getInstance()->dumpHTML(ofile);
    ofile << "</body></html>\n";

    return true;
}

static std::string xmlencode(const std::string& str)
{
    std::string res;
    for (unsigned int i=0; i<str.length(); ++i)
    {
        switch(str[i])
        {
        case '<': res += "&lt;"; break;
        case '>': res += "&gt;"; break;
        case '&': res += "&amp;"; break;
        case '"': res += "&quot;"; break;
        case '\'': res += "&apos;"; break;
        default:  res += str[i];
        }
    }
    return res;
}

//bool generateClassPHPDoc(const std::string& filename, ObjectFactory::ClassEntry::SPtr entry)
//{
//    return true;
//}

static const char* baseClasses[] =
{
//"BaseNode",
//"BaseObject",
    "BehaviorModel",
    "CollisionModel",
    "VisualModel",
    "Mapping",
    "MechanicalState",
    "MechanicalMapping",
    "AnimationLoop",
    "AnimationLoop",
    "OdeSolver",
    "Mass",
    "ForceField",
    "InteractionForceField",
    "Constraint",
//"CollisionPipeline",
    "Topology",
    "ContextObject",
    NULL
};

#if 0
class DummyObjectDescription : public objectmodel::BaseObjectDescription
{
protected:
    std::string name;
    AttributeMap attributes;
    objectmodel::BaseObjectDescription* parent;
public:
    DummyObjectDescription()
        : name ("dummy")
    {
        parent = this;
    }

    /// Get the associated object
    virtual objectmodel::Base* getObject() { return NULL; }

    /// Get the node instance name
    virtual const std::string& getName() const { return name; }

    /// Get the parent node
    virtual objectmodel::BaseObjectDescription* getParent() const { return parent; }

    /// Get all attribute data, read-only
    virtual const AttributeMap& getAttributeMap() const { return attributes; }

    /// Find a node given its name
    virtual objectmodel::BaseObjectDescription* find(const char* /*nodeName*/, bool /*absolute*/=false) { return this; }

    virtual std::string getFullName() { return name; }

    /// Find an object given its name
    virtual objectmodel::Base* findObject(const char* /*nodeName*/) { return NULL; }

};
#endif

bool generateFactoryPHPDoc(const std::string& filename, const std::string& url)
{
    //ObjectFactory::getInstance()->dump();
    std::ofstream out(filename.c_str());

    //out << "<html><body>\n";

    std::vector<std::string> templates;
    std::set<std::string> templateSet;

    ObjectFactory::ClassEntry& mobject = ObjectFactory::getInstance()->getEntry("MechanicalObject");
    for (ObjectFactory::CreatorMap::iterator itc = mobject.creatorMap.begin(), itcend = mobject.creatorMap.end(); itc != itcend; ++itc)
      {
	if (itc->first.length() < 10) // only put short templates
	  {
	    templates.push_back(itc->first);
	    templateSet.insert(itc->first);
	  }
      }

    std::vector<ObjectFactory::ClassEntry::SPtr> classes;
    ObjectFactory::getInstance()->getAllEntries(classes);

    // re-order classes depending on namespaces

    std::map<std::string, std::vector<ObjectFactory::ClassEntry::SPtr> > sortedClasses;
    for (std::vector<ObjectFactory::ClassEntry::SPtr>::iterator it = classes.begin(), itend = classes.end(); it != itend; ++it)
    {
        ObjectFactory::ClassEntry::SPtr entry = *it;
        std::string nameSpace = objectmodel::BaseClass::decodeNamespaceName(entry->creatorMap.begin()->second->type());
        sortedClasses[nameSpace].push_back(entry);
    }

    out << "<?php $show=$_REQUEST['show']; ?>\n";

    out << "<?php $desc=$_REQUEST['desc']; ?>\n";

    out << "<?php $base=0; \n";
    for (const char** c = baseClasses; *c; ++c)
        out << " if ($show == '" << xmlencode(*c) << "') $base=1;\n";
    out << " if (!$show) $base=1; ?>\n";

    out << "<h1><?php if (!$base) echo 'Component : '.$show; else if ($show) echo 'Registered Components : '.$show; else echo 'Registered Components' ?></h1>\n";

    out << "<?php if ($base) { ?>";

    out << "<p> <b>Filter by Base Class :</b> ";

    out << "<?php if ($show) { echo '<a href=\""<<url<<"'.($desc?'?desc='.$desc:'').'\">'; ?>";
    out << "ALL";
    out << "<?php } if ($show) echo '</a>'; ?>";
    for (const char** c = baseClasses; *c; ++c)
    {
        out << " <span class=\"class-base-name\" style=\"background-color: " << sofa::simulation::Colors::getColor(*c) << ";\">";
        out << "<?php if ($show != '"<<xmlencode(*c)<<"') echo '<a href=\""<<url<<"?show="<<xmlencode(*c)<<"'.($desc?'&desc='.$desc:'').'\">'; ?>";
        out << *c;
        out << "<?php if ($show != '"<<xmlencode(*c)<<"') echo '</a>'; ?>";
        out << "</span>";
    }
    out << "</p>\n";
    out << "<p><?php if ($desc) echo '<a href=\""<<url<<"'.($show? ('?show='.$show) : '').'\">Hide Descriptions<a>'; else echo '<a href=\""<<url<<"'.($show? ('?show='.$show).'&desc=1' : '?desc=1').'\">Show Descriptions<a>'; ?></p>";
    out << "<?php } ?>";

    out << "<table class=\"sofa-classes\" border=\"0\" cellspacing=\"0\" cellpadding=\"1\">\n";

    int nbcol = 2 + templates.size() + 2;

    // header
    out << "<tr class=\"sofa-header\">";
    out << "<td class=\"sofa-base-header\">Base&nbsp;Class</td>";
    out << "<td class=\"sofa-name-header\">Class&nbsp;Name</td>";
    for (std::vector<std::string>::iterator itt = templates.begin(), ittend = templates.end(); itt != ittend; ++itt)
        out << "<td class=\"sofa-template-header\">"<<xmlencode(*itt)<<"</td>";
    out << "<td class=\"sofa-template-header\">Other</td>";
    out << "<td class=\"sofa-namespace-header\">Namespace</td>";
    out << "</tr>\n";

    for (std::map< std::string, std::vector<ObjectFactory::ClassEntry::SPtr> >::iterator it1 = sortedClasses.begin(), it1end = sortedClasses.end(); it1 != it1end; ++it1)
        for (std::vector<ObjectFactory::ClassEntry::SPtr>::iterator it = it1->second.begin(), itend = it1->second.end(); it != itend; ++it)
        {
            ObjectFactory::ClassEntry::SPtr entry = *it;

            out << "<?php if (!$show || $show=='"<<xmlencode(entry->className)<<"'";

	    std::vector<std::string> categories;
	    if (! entry->creatorMap.empty())
	        CategoryLibrary::getCategories(entry->creatorMap.begin()->second->getClass(), categories);

	    for (std::vector<std::string>::iterator it = categories.begin(), itend = categories.end(); it != itend; ++it)
  	        if (categories.size()==1 || *it != "VisualModel") // VisualModel matters only if it is the only base class
		    out << " || $show == '"<<xmlencode(*it)<<"'";

            out << ") { ?>\n";


            out << "<tr class=\"sofa-class\">";
            out << "<td class=\"sofa-base\" valign=\"top\">";

	    for (std::vector<std::string>::iterator it = categories.begin(), itend = categories.end(); it != itend; ++it)
	        out << "<span class=\"class-base\" style=\"background-color: " << sofa::simulation::Colors::getColor((*it).c_str()) << ";\" alt=\"" << xmlencode(*it) << "\" title=\"" << xmlencode(*it) << "\">&nbsp;</span>";

            out << "</td>";

            out << "<td class=\"sofa-name\" valign=\"top\"><?php if ($base) echo '<a href=\""<<url<<"?show="<<xmlencode(entry->className)<<"'.($desc?'&desc='.$desc:'').'\" class=\"class-name\">'; else echo '<span class=\"class-name\">'; ?>" << xmlencode(entry->className) <<"<?php if ($base) echo '</a>'; else echo '</span>'; ?>\n";
            if (!entry->aliases.empty())
            {
                for (std::set<std::string>::iterator it = entry->aliases.begin(), itend = entry->aliases.end(); it != itend; ++it)
                    out << "<br/><span class=\"class-alias\">" << xmlencode(*it) <<"</span>\n";
            }
            out << "</td>";

            for (std::vector<std::string>::iterator itt = templates.begin(), ittend = templates.end(); itt != ittend; ++itt)
            {
                out << "<td class=\"sofa-template\" align=\"center\" valign=\"top\">";
                if (entry->defaultTemplate == *itt)
//		out << "<img src=\"images/classes/default-template.png\" alt=\""<<xmlencode(*itt)<<" (default)\" title=\""<<xmlencode(*itt)<<" (default)\" />";
                    out << "<span class=\"class-template-default\" alt=\""<<xmlencode(*itt)<<" (default)\" title=\""<<xmlencode(*itt)<<" (default)\">*</span>";
                else if (entry->creatorMap.find(*itt)!=entry->creatorMap.end())
//		out << "<img src=\"images/classes/template.png\" alt=\""<<xmlencode(*itt)<<"\" title=\""<<xmlencode(*itt)<<"\" />";
                    out << "<span class=\"class-template\" alt=\""<<xmlencode(*itt)<<"\" title=\""<<xmlencode(*itt)<<"\">*</span>";
                else
                    out << "<span class=\"class-no-template\">&nbsp;<span>";
                out << "</td>";
            }
            std::string others;
            for (ObjectFactory::CreatorMap::iterator itc = entry->creatorMap.begin(), itcend = entry->creatorMap.end(); itc != itcend; ++itc)
            {
                if (!itc->first.empty() && templateSet.find(itc->first)==templateSet.end())
                {
                    if (!others.empty())
                        others += ", ";
                    others += xmlencode(itc->first);
                    if (itc->first == entry->defaultTemplate)
                        others += " (default)";
                }
            }
            out << "<td class=\"sofa-template\" align=\"center\" valign=\"top\">";
            if (!others.empty())
//	    out << "<img src=\"images/classes/template.png\" alt=\"...\" title=\""<<others<<"\" />";
                out << "<span class=\"class-template-others\" alt=\""<<others<<"\" title=\""<<others<<"\">*</span>";
            else
                out << "<span class=\"class-no-template\">&nbsp;<span>";
            out << "</td>";
            std::string nameSpace = it1->first; //objectmodel::BaseClass::decodeNamespaceName(entry->creatorMap.begin()->second->type());
            out << "<td class=\"sofa-namespace\" valign=\"top\">";
            out << "<span class=\"class-namespace\">"<< xmlencode(nameSpace) <<"</span>";
            out << "</td>";
            out << "</tr>\n";
            if ((!entry->description.empty() && entry->description.substr(0,4)!="TODO") || !entry->authors.empty() || !entry->license.empty())
                out << "<?php if ($desc || !$base) { ?>";
            else
                out << "<?php if (!$base) { ?>";
            out << "<tr class=\"sofa-description\">";
            out << "<td>";
            out << "</td>";
            out << "<td class=\"sofa-description\" colspan=\""<< nbcol-1 <<"\" >";
            if (!entry->description.empty() && entry->description.substr(0,4)!="TODO")
                out << "<div class=\"class-description\">"<<entry->description<<"</div>";
            if (!entry->authors.empty())
                out << "<div class=\"class-authors\">Authors: "<<entry->authors<<"</div>";
            if (!entry->license.empty())
                out << "<div class=\"class-license\">License: "<<entry->license<<"</div>";
            out << "<?php if (!$base) { echo '<a href=\"api?class="<<xmlencode(nameSpace)<<"::"<<xmlencode(entry->className)<<"\">Doxygen class documentation</a>'; } ?>";
            out << "</td>";
            out << "</tr>\n";
            out << "<?php if (!$base) { ?>";
            // Write Fields
            out << "<tr class=\"sofa-fields-header\">";
            out << "<td></td>";
            out << "<td class=\"sofa-field-name-header\">Field Name</td>";
            out << "<td class=\"sofa-field-type-header\" colspan=\""<<templates.size() + 1<<"\">Type</td>";
            out << "<td class=\"sofa-field-value-header\">Value</td>";
            out << "</tr>\n";

            // Take the first creator for now
            try
            {
                std::cerr << "Trying to instantiate "<<entry->className<<std::endl;
                ObjectFactory::Creator::SPtr creator = entry->creatorMap.begin()->second;
                //DummyObjectDescription arg;
                //objectmodel::Context ctx;
                objectmodel::BaseObject::SPtr object = creator->createInstance(NULL, NULL); //&ctx, &arg);
                if (object == NULL)
                {
                    std::cerr << "ERROR: Failed to instantiate "<<entry->className<<std::endl;
                }
                else
                {
                    const objectmodel::Base::VecData& fields = object->getDataFields();
                    for (objectmodel::Base::VecData::const_iterator itf = fields.begin(), itfend = fields.end(); itf != itfend; ++itf)
                    {
                        objectmodel::BaseData* f = *itf;
                        out << "<tr class=\"sofa-field\">";
                        out << "<td></td>";
                        out << "<td class=\"sofa-field-name\">";
                        out << "<span class=\"field-name\">"<<xmlencode(f->getName())<<"</span>";
                        out << "</td>";
                        out << "<td class=\"sofa-field-type\" colspan=\""<<templates.size() + 1<<"\">";
                        out << "<span class=\"field-type\">"<<xmlencode(f->getValueTypeString())<<"</span>";
                        out << "</td>";
                        out << "<td class=\"sofa-field-value\">";
                        out << "<span class=\"field-value\">"<<xmlencode(f->getValueString())<<"</span>";
                        out << "</td>";
                        out << "</tr>\n";
                        if (f->getHelp() && *f->getHelp() && strncmp(f->getHelp(),"TODO",4))
                        {
                            out << "<tr class=\"sofa-field-description\">";
                            out << "<td></td>";
                            out << "<td class=\"sofa-field-description\" colspan=\""<<nbcol - 1<<"\">";
                            out << "<div class=\"field-description\">"<<f->getHelp()<<"</div>";
                            out << "</td>";
                            out << "</tr>\n";
                        }
                    }
                    // object is now automatically deleted when the smart pointer is destructed
                    //delete object;
                }
            }
            catch(...)
            {
                std::cerr << "ERROR: Exception raised while trying to instantiate "<<entry->className<<std::endl;
            }
            out << "</tr>\n";
            out << "<?php } ?>\n";
            out << "<?php } ?>\n";
            out << "<?php } ?>\n";

        }
    out << "</table>\n";

    out << "<?php if ($base) { ?>";

    out << "<p> <b>Filter by Base Class :</b> ";

    out << "<?php if ($show) { echo '<a href=\""<<url<<"'.($desc?'?desc='.$desc:'').'\">'; ?>";
    out << "ALL";
    out << "<?php } if ($show) echo '</a>'; ?>";
    for (const char** c = baseClasses; *c; ++c)
    {
        out << " <span class=\"class-base-name\" style=\"background-color: " << sofa::simulation::Colors::getColor(*c) << ";\">";
        out << "<?php if ($show != '"<<xmlencode(*c)<<"') echo '<a href=\""<<url<<"?show="<<xmlencode(*c)<<"'.($desc?'&desc='.$desc:'').'\">'; ?>";
        out << *c;
        out << "<?php if ($show != '"<<xmlencode(*c)<<"') echo '</a>'; ?>";
        out << "</span>";
    }
    out << "</p>\n";
    out << "<?php } ?>";

    //out << "</body></html>\n";

    return true;
}

}
