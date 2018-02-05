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
#define SOFA_GUI_QT_LINKWIDGET_CPP

#include "LinkWidget.h"
#include "ModifyObject.h"
#include <sofa/helper/Factory.inl>

#include <QToolTip>

#define SIZE_TEXT     60

namespace sofa
{

using namespace core::objectmodel;

namespace gui
{

namespace qt
{

LinkWidget *LinkWidget::CreateLinkWidget(const LinkWidget::CreatorArgument &/*dwarg*/)
{
    return NULL; // TODO
}

/*QDisplayLinkInfoWidget definitions */

QDisplayLinkInfoWidget::QDisplayLinkInfoWidget(QWidget* parent, const std::string& helper,
        core::objectmodel::BaseLink* l, bool /*modifiable*/)
    : QWidget(parent), link(l), numLines_(1)
{
    QHBoxLayout* layout = new QHBoxLayout(this);

    layout->setContentsMargins(0,0,0,0);

    std::string final_str;
    formatHelperString(helper,final_str);
    
	const core::objectmodel::BaseClass* ownerClass=link->getOwnerClass();
    std::string ownerClassName;
	if (ownerClass) ownerClassName = ownerClass->className;
	
	/*
#ifndef SOFA_GUI_QT_NO_DATA_HELP
    QLabel* helper_label = new QLabel(this);
    helper_label->setText(QString(final_str.c_str()));
    helper_label->setMinimumWidth(20);
    layout->addWidget(helper_label);
    if (!ownerClassName.empty()) QToolTip::add(helper_label, ("Link from "+ownerClassName).c_str());
#else
    numLines_ = 0;
    if (!final_str.empty() || !ownerClassName.empty())
    {
        if (!final_str.empty()) final_str += '\n';
        final_str += "Link from ";
        final_str += ownerClassName;
        QToolTip::add(parent, final_str.c_str());
    }
#endif
	*/
}

void QDisplayLinkInfoWidget::formatHelperString(const std::string& helper, std::string& final_text)
{
    std::string label_text=helper;
    numLines_ = 0;
    while (!label_text.empty())
    {
        std::string::size_type pos = label_text.find('\n');
        std::string current_sentence;
        if (pos != std::string::npos)
            current_sentence  = label_text.substr(0,pos+1);
        else
            current_sentence = label_text;
        if (current_sentence.size() > SIZE_TEXT)
        {
            const std::size_t cut = current_sentence.size()/SIZE_TEXT;
            for (std::size_t index_cut=1; index_cut<=cut; index_cut++)
            {
                std::string::size_type numero_char=current_sentence.rfind(' ',SIZE_TEXT*index_cut);
                current_sentence = current_sentence.insert(numero_char+1,1,'\n');
                numLines_++;
            }
        }
        if (pos != std::string::npos) label_text = label_text.substr(pos+1);
        else label_text = "";
        final_text += current_sentence;
        numLines_++;
    }
}

unsigned int QDisplayLinkInfoWidget::numLines(const std::string& str)
{
    std::string::size_type newline_pos;
    unsigned int numlines = 1;
    newline_pos = str.find('\n',0);
    while( newline_pos != std::string::npos )
    {
        numlines++;
        newline_pos = str.find('\n',newline_pos+1);
    }
    return numlines;
}

} // namespace qt

} // namespace gui

} // namespace sofa
