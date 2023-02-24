/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <string>
#include <typeinfo>
#include <cstdlib>
#include <sofa/simulation/common/xml/XML.h>
#include <sofa/helper/logging/Message.h>
#include <sofa/helper/system/Locale.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/core/ObjectFactory.h>
#include <cstring>
#include <tinyxml.h>

/* For loading the scene */


namespace sofa::simulation::xml
{

using std::cout;
using std::endl;

#define is(n1, n2) (! xmlStrcmp((const xmlChar*)n1,(const xmlChar*)n2))
#define getProp(n) ( xmlGetProp(cur, (const xmlChar*)n) )

void recReplaceAttribute(BaseElement* node, const char* attr, const char* value, const char* nodename=nullptr)
{
    if (nodename)
    {
        if (node->getName() == nodename)
        {
            node->addReplaceAttribute(attr,value);
        }
    }
    else
    {
        node->addReplaceAttribute(attr,value);
    }
    BaseElement::child_iterator<> it = node->begin();
    BaseElement::child_iterator<> end = node->end();
    while (it != end)
    {
        recReplaceAttribute( it, attr, value, nodename );
        ++it;
    }
}


BaseElement* includeNode  (TiXmlNode* root,const char *basefilename);
BaseElement* attributeNode(TiXmlNode* root,const char *basefilename);
void recursiveMergeNode(BaseElement* destNode, BaseElement* srcNode);

int numDefault=0;

BaseElement* createNode(TiXmlNode* root, const char *basefilename, bool isRoot = false)
{
    //if (!xmlStrcmp(root->name,(const xmlChar*)"text")) return nullptr;

    // TinyXml API changed in 2.6.0, ELEMENT was replaced with TINYXML_ELEMENT
    // As the version number is not available as a macro, the most portable was is to
    // replace these constants with checks of the return value of ToElement()
    // (which is already done here). -- Jeremie A. 02/07/2011
    // if (root->Type() != TiXmlNode::ELEMENT) return nullptr;
    TiXmlElement* element = root->ToElement();
    if (!element)
        return nullptr;

    if (!element->Value() || !element->Value()[0])
    {
        msg_error_withfile("XMLParser", basefilename, element->Row()) << "Invalid element : " << *element ;
        return nullptr;
    }

    // handle special 'preprocessor' tags

    if (std::string(element->Value())=="include")
    {
        return includeNode(root, basefilename);
    }

    std::string classType,name, type;

    classType = element->Value();

    const char* pname = element->Attribute("name");

    if (pname != nullptr)
    {
        name = pname;
    }
    else
    {
        name = "";
        ++numDefault;
    }

    const char* ptype = element->Attribute("type");

    if (ptype != nullptr)
    {
        type = ptype;
    }
    else
    {
        type = "default";
    }

    if (!BaseElement::NodeFactory::HasKey(classType) && type == "default")
    {
        type=classType;
        classType="Object";
    }
    if (classType == "Object" && !sofa::core::ObjectFactory::HasCreator(type))
    {
        // look if we have a replacement XML for this type
        std::string filename = "Objects/";
        filename += type;
        filename += ".xml";

        if (sofa::helper::system::DataRepository.findFileFromFile(filename, basefilename, nullptr))
        {
            // we found a replacement xml
            element->SetAttribute("href",filename.c_str());
            element->RemoveAttribute("type");
            return includeNode(root, basefilename);
        }
    }

    BaseElement* node = BaseElement::Create(classType,name,type);
    if (node==nullptr)
    {
        msg_info_withfile("XMLParser", basefilename, element->Row()) << "Node "<<element->Value()<<" name "<<name<<" type "<<type<<" creation failed.\n";
        return nullptr;
    }

    if (isRoot)
        node->setBaseFile( basefilename );

    node->setSrcFile(basefilename);
    node->setSrcLine(element->Row()) ;

     // List attributes
    for (TiXmlAttribute* attr=element->FirstAttribute(); attr ; attr = attr->Next())
    {
        if (attr->Value()==nullptr) continue;
        if (!(strcmp(attr->Name(), "name"))) continue;
        if (!(strcmp(attr->Name(), "type"))) continue;
        node->setAttribute(attr->Name(), std::string(attr->Value()));
    }

    for (TiXmlNode* child = root->FirstChild() ; child != nullptr; child = child->NextSibling())
    {
        BaseElement* childnode = createNode(child, basefilename);
        if (childnode != nullptr)
        {
            //  if the current node is an included node, with the special name Group, we only add the objects.
            switch(childnode->getIncludeNodeType())
            {
            case INCLUDE_NODE_CHILD:
            {
                if (!node->addChild(childnode))
                {
                    msg_info_withfile("XMLParser", basefilename, element->Row()) << "Node "<<childnode->getClass()<<" name "<<childnode->getName()<<" type "<<childnode->getType()
                            <<" cannot be a child of node "<<node->getClass()<<" name "<<node->getName()<<" type "<<node->getType() ;
                    delete childnode;
                }
                break;
            }
            case INCLUDE_NODE_GROUP:
            {
                BaseElement::child_iterator<> it(childnode->begin());
                for(; it!=childnode->end(); ++it) {node->addChild(it); childnode->removeChild(it);}
                //delete childnode;
                break;
            }
            case INCLUDE_NODE_MERGE:
            {
                recursiveMergeNode(node, childnode);
                //delete childnode;
                break;
            }
            }
        }
    }
    return node;
}

BaseElement* processXMLLoading(const char *filename, const TiXmlDocument &doc, bool fromMem)
{
    const TiXmlElement* hRoot = doc.RootElement();

    if (hRoot == nullptr)
    {
        msg_info("XMLParser") << " Empty document: " << filename ;
        return nullptr;
    }

    std::string basefilename;
    if(fromMem)
        basefilename = filename ;
    else
        basefilename = sofa::helper::system::SetDirectory::GetRelativeFromDir(filename,sofa::helper::system::SetDirectory::GetCurrentDir().c_str());
    BaseElement* graph = createNode((TiXmlElement*)hRoot, basefilename.c_str(), true);

    if (graph == nullptr)
    {
        msg_error("XMLParser") << "XML Graph creation failed." ;
        return nullptr;
    }

    return graph;
}

BaseElement* loadFromMemory(const char* filename, const char* data)
{
    TiXmlDocument doc; // the resulting document tree

    //xmlSubstituteEntitiesDefault(1);

    doc.Parse(data);
    if (doc.Error())
    {
        msg_error("XMLParser") << "Failed to open " << filename << "\n" << doc.ErrorDesc() << " at line " << doc.ErrorRow() << " row " << doc.ErrorCol();
        return nullptr;
    }
    return processXMLLoading(filename, doc, true);
}

BaseElement* loadFromFile(const char *filename)
{
    // Temporarily set the numeric formatting locale to ensure that
    // floating-point values are interpreted correctly by tinyXML. (I.e. the
    // decimal separator is a dot '.').
    helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

    // this initialize the library and check potential ABI mismatches
    // between the version it was compiled for and the actual shared
    // library used.
    TiXmlDocument* doc = new TiXmlDocument; // the resulting document tree

    // xmlSubstituteEntitiesDefault(1);

    if (!(doc->LoadFile(filename)))
    {
        msg_error("XMLParser") << "Failed to open " << filename << "\n" << doc->ErrorDesc() << " at line " << doc->ErrorRow() << " row " << doc->ErrorCol() ;
        delete doc;
        return nullptr;
    }

    BaseElement* r = processXMLLoading(filename, *doc);
    //dmsg_error("XML") << "clear doc";
    doc->Clear();
    //dmsg_error("XML") << "delete doc";
    delete doc;
    //dmsg_error("XML") << "<loadFromFile";
    return r;
}


BaseElement* includeNode(TiXmlNode* root,const char *basefilename)
{
    TiXmlElement* element = root->ToElement();
    if (!element) return nullptr;

    std::string filename;
    const char *pfilename = element->Attribute("href");
    if (pfilename)
    {
        filename = (const char*)pfilename;
    }
    if (filename.empty())
    {
        msg_error("XMLParser") << "Xml include tag requires non empty filename or href attribute." ;
        return nullptr;
    }
    sofa::helper::system::DataRepository.findFileFromFile(filename, basefilename);
    TiXmlDocument doc; // the resulting document tree
    if (!doc.LoadFile(filename.c_str()))
    {
        msg_error("XMLParser") << "Failed to parse " << filename << "\n";
        return nullptr;
    }
    TiXmlElement* newroot = doc.RootElement();

    if (newroot == nullptr)
    {
        msg_error("XMLParser") << "ERROR: empty document in " << filename << "\n";
        //xmlFreeDoc(doc);
        return nullptr;
    }
    BaseElement* result = createNode(newroot, filename.c_str(), true);
    if (result)
    {
        if (result->getName() == "Group") result->setIncludeNodeType(INCLUDE_NODE_GROUP);
        if (result->getName() == "_Group_") result->setIncludeNodeType(INCLUDE_NODE_GROUP);
        if (result->getName() == "_Merge_") result->setIncludeNodeType(INCLUDE_NODE_MERGE);
        // Copy attributes
        for (TiXmlAttribute* attr=element->FirstAttribute(); attr != nullptr ; attr = attr->Next())
        {
            if (attr->Value()==nullptr) continue;
            if (!(strcmp(attr->Name(), "href"))) continue;
            if (!(strcmp(attr->Name(), "name")))
            {
                result->setName(attr->Value());
                if (result->getName() == "Group") result->setIncludeNodeType(INCLUDE_NODE_GROUP);
                if (result->getName() == "_Group_") result->setIncludeNodeType(INCLUDE_NODE_GROUP);
                if (result->getName() == "_Merge_") result->setIncludeNodeType(INCLUDE_NODE_MERGE);
            }
            else
            {
                const char* attrname = attr->Name();

                const char* value = attr->Value();
                if (const char* sep = strstr(attrname,"__"))
                {
                    // replace attribute in nodes with a given name
                    std::string nodename(attrname, sep);
                    recReplaceAttribute(result, sep+2, value, nodename.c_str());
                }
                else
                {
                    // replace attribute in all nodes already containing it
                    recReplaceAttribute(result, attrname, value);
                }
            }
        }
    }
    //xmlFreeDoc(doc);
    return result;
}

void recursiveMergeNode(BaseElement* destNode, BaseElement* srcNode)
{
    // copy all attributes
    std::vector<std::string> attrs;
    srcNode->getAttributeList(attrs);
    for (std::vector<std::string>::const_iterator it = attrs.begin(); it != attrs.end(); ++it)
    {
        std::string aname = *it;
        if (aname == "name") continue;
        const char* aval = srcNode->getAttribute(aname);
        if (!aval) continue;
        destNode->setAttribute(aname, std::string(aval));
    }
    BaseElement::child_iterator<> itS(srcNode->begin());
    for(; itS!=srcNode->end(); ++itS)
    {
        BaseElement* srcElem = itS;
        BaseElement* destElem = nullptr;
        if (!srcElem->getName().empty())
        {
            BaseElement::child_iterator<> itD(destNode->begin());
            for(; itD!=destNode->end(); ++itD)
            {
                BaseElement* e = itD;
                if (e->getType() == srcElem->getType() && e->getName() == srcElem->getName())
                {
                    destElem = e;
                    break;
                }
            }
        }
        if (destElem)
            recursiveMergeNode(destElem, srcElem); // match found, merge recursively
        else
        {
            // no match found, move the whole sub-tree
            destNode->addChild(srcElem);
            srcNode->removeChild(srcElem);
        }
    }
}


} // namespace sofa::simulation::xml
