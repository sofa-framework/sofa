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
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/


#include <sofa/core/logging/RichConsoleStyleMessageFormatter.h>

#include <sofa/helper/logging/Message.h>

#include <sofa/core/objectmodel/Base.h>
using sofa::helper::logging::SofaComponentInfo;

#include <sofa/helper/system/console.h>
#include <sofa/type/fixed_array.h>

#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>
#include <boost/token_iterator.hpp>

#ifdef WIN32
#undef min
#undef max
#endif

#include <algorithm>

namespace sofa::helper::logging
{

/////////////////////////////// STATIC ELEMENT SPECIFIC TO RichConsoleStyleMessage /////////////////
typedef boost::tokenizer<boost::char_separator<char> > tokenizer;

///
/// \brief simpleFormat a text containing our markdown 'tags'
/// \param jsize size of the line prefix to fill with space (for left side alignment)
/// \param text  the text to format
/// \param line_length number of column to render to to
/// \param wrapped the destination stream where to write the formatted text.
///
void simpleFormat(size_t jsize, const std::string& text, size_t line_length,
                  std::ostream& wrapped)
{
    //TODO(dmarchal): All that code is a mess...need to be done for real.

    /// space and * are separator that are returned in the token flow
    /// while "\n" is a 'hidden' separator.
    static boost::char_separator<char> sep("\n", "* '");

    std::string emptyspace(jsize, ' ') ;

    tokenizer tokens(text, sep) ;

    int numspaces = 0 ;
    bool beginOfLine = false ;
    bool isInItalic = false ;

    size_t space_left = line_length;
    for (tokenizer::iterator tok_iter = tokens.begin();tok_iter != tokens.end(); ++tok_iter)
    {
        const std::string& word = *tok_iter;
        if(word=="'" || word=="*")
        {
            if(isInItalic)
            {
                isInItalic=false;
                wrapped << console::Style::Reset;
                wrapped << "'";
                continue;
            }
            else
            {
                isInItalic=true;

                if(numspaces==1){
                    if(!beginOfLine){
                        wrapped << " ";
                        numspaces = 0;
                        space_left--;
                    }else{
                        wrapped << console::Style::Reset << emptyspace ;
                    }
                }

                wrapped << "'";
                wrapped << console::Style::Italic;
                wrapped << console::Style::Underline;
                continue;
            }
        }else if(word==" ")
        {
            if(numspaces==1)
            {
                wrapped << "\n"  ;
                numspaces=0;
                space_left = line_length;
                beginOfLine=true;
                continue;
            }else
            {
                numspaces=1;
                continue;
            }
        }else{
            if(numspaces==1){
                if(!beginOfLine){
                    wrapped << " ";
                    numspaces = 0;
                    space_left--;
                }else{
                    wrapped << console::Style::Reset;
                    wrapped << emptyspace ;
                    if(isInItalic){
                        wrapped << console::Style::Italic;
                        wrapped << console::Style::Underline;
                    }
                }
            }
        }

        if (space_left < word.length() + 1)
        {
            if(word.length()>line_length)
            {
                std::string first;
                size_t curidx=0;
                size_t endidx=std::min(word.length(), space_left-1);

                while(curidx < word.length())
                {
                    first=word.substr(curidx,endidx);

                    if(beginOfLine){
                        wrapped << console::Style::Reset;
                        wrapped << emptyspace ;
                        if(isInItalic){
                            wrapped << console::Style::Italic;
                            wrapped << console::Style::Underline;
                        }
                    }
                    beginOfLine=false;
                    wrapped << first ;

                    curidx+=endidx;
                    endidx=std::min(word.length()-curidx, line_length-1);

                    if(curidx < word.length())
                    {
                        wrapped << "\n" ;
                        beginOfLine=true;
                    }
                }
                space_left = line_length - first.length();
            }
            else
            {
                wrapped << "\n";
                wrapped << console::Style::Reset;
                wrapped << emptyspace ;
                if(isInItalic){
                    wrapped << console::Style::Italic;
                    wrapped << console::Style::Underline;
                }
                wrapped << word ;
                space_left = line_length-word.length();
            }
        }
        else
        {
            if(beginOfLine){
                wrapped << console::Style::Reset;
                wrapped << emptyspace ;
                if(isInItalic){
                    wrapped << console::Style::Italic;
                    wrapped << console::Style::Underline;
                }
            }
            beginOfLine=false;
            wrapped << word;
            space_left -= word.length() ;
        }
    }
}



////////////////////////////// RichConsoleStyleMessageFormatter Implementation /////////////////////

void RichConsoleStyleMessageFormatter::formatMessage(const Message& m, std::ostream& out)
{
    size_t psize = getPrefixText(m.type()).size() ;

    setColor(out, m.type()) << getPrefixText(m.type());

    const SofaComponentInfo* nfo = dynamic_cast<SofaComponentInfo*>(m.componentInfo().get()) ;
    if( nfo != nullptr )
    {
        const std::string& classname= nfo->sender();
        const std::string& name = nfo->name();
        psize +=classname.size()+name.size()+5 ;
        out << console::Foreground::Normal::Blue << "[" << classname << "(" << name << ")] ";
    }
    else
    {
        psize +=m.sender().size()+3 ;
        out << console::Foreground::Normal::Blue << "[" << m.sender()<< "] ";
    }

    out << console::Style::Reset;

    /// Format & align the text and write the result into 'out'.
    simpleFormat(psize , m.message().str(), console::getColumnCount()-psize, out) ;

    if(m_showFileInfo && m.fileInfo()){
        std::stringstream buf;
        const std::string emptyspace(psize, ' ') ;
        buf << "Emitted from '" << m.fileInfo()->filename << "' line " << m.fileInfo()->line ;
        out << "\n" << console::Style::Reset << emptyspace ;
        simpleFormat(psize , buf.str(), console::getColumnCount()-psize, out) ;
    }

    ///Restore the console rendering attribute.
    out << console::Style::Reset;
    out << std::endl ;
}
} // sofa::helper::logging

