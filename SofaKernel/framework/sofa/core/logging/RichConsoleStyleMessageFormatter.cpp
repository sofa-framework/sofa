/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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


#include "RichConsoleStyleMessageFormatter.h"

#include <sofa/helper/logging/Message.h>

#include <sofa/core/objectmodel/Base.h>
using sofa::helper::logging::SofaComponentInfo ;

#include <sofa/helper/system/console.h>
#include <sofa/helper/fixed_array.h>

#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>
#include <boost/token_iterator.hpp>

namespace sofa
{

namespace helper
{

namespace logging
{

namespace richconsolestylemessageformater
{
/////////////////////////////// STATIC ELEMENT SPECIFIC TO RichConsoleStyleMessage /////////////////
typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
helper::fixed_array<std::string,Message::TypeCount> s_messageTypePrefixes;
helper::fixed_array<Console::ColorType,Message::TypeCount> s_messageTypeColors;

int initColors(){
    s_messageTypePrefixes[Message::Advice]      = "[SUGGESTION] ";
    s_messageTypePrefixes[Message::Info]        = "[INFO]    ";
    s_messageTypePrefixes[Message::Deprecated]  = "[DEPRECATED] ";
    s_messageTypePrefixes[Message::Warning]     = "[WARNING] ";
    s_messageTypePrefixes[Message::Error]       = "[ERROR]   ";
    s_messageTypePrefixes[Message::Fatal]       = "[FATAL]   ";
    s_messageTypePrefixes[Message::TEmpty]      = "[EMPTY]   ";

    s_messageTypeColors[Message::Advice]       = Console::BRIGHT_GREEN;
    s_messageTypeColors[Message::Info]       = Console::BRIGHT_GREEN;
    s_messageTypeColors[Message::Deprecated] = Console::BRIGHT_YELLOW;
    s_messageTypeColors[Message::Warning]    = Console::BRIGHT_CYAN;
    s_messageTypeColors[Message::Error]      = Console::BRIGHT_RED;
    s_messageTypeColors[Message::Fatal]      = Console::BRIGHT_PURPLE;
    s_messageTypeColors[Message::TEmpty]     = Console::DEFAULT_COLOR;
    return 1;
}

int s_isInited = initColors();

///
/// \brief simpleFormat a text containing our markdown 'tags'
/// \param jsize size of the line prefix to fill with space (for left side alignment)
/// \param text  the text to format
/// \param line_length number of column to render to to
/// \param wrapped the destination stream where to write the formatted text.
///
void simpleFormat(int jsize, const std::string& text, size_t line_length,
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
                wrapped << Console::DEFAULT_CODE ;
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
                        wrapped << Console::DEFAULT_CODE << Console::DEFAULT_COLOR << emptyspace ;
                    }
                }

                wrapped << "'";
                wrapped << Console::ITALIC ;
                wrapped << Console::UNDERLINE ;
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
                    wrapped << Console::DEFAULT_CODE;
                    wrapped << Console::DEFAULT_COLOR;
                    wrapped << emptyspace ;
                    if(isInItalic){
                        wrapped << Console::ITALIC;
                        wrapped << Console::UNDERLINE;
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
                        wrapped << Console::DEFAULT_CODE;
                        wrapped << Console::DEFAULT_COLOR;
                        wrapped << emptyspace ;
                        if(isInItalic){
                            wrapped << Console::ITALIC;
                            wrapped << Console::UNDERLINE;
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
                wrapped << Console::DEFAULT_CODE;
                wrapped << Console::DEFAULT_COLOR;
                wrapped << emptyspace ;
                if(isInItalic){
                    wrapped << Console::ITALIC;
                    wrapped << Console::UNDERLINE;
                }
                wrapped << word ;
                space_left = line_length-word.length();
            }
        }
        else
        {
            if(beginOfLine){
                wrapped << Console::DEFAULT_CODE;
                wrapped << Console::DEFAULT_COLOR;
                wrapped << emptyspace ;
                if(isInItalic){
                    wrapped << Console::ITALIC;
                    wrapped << Console::UNDERLINE;
                }
            }
            beginOfLine=false;
            wrapped << word;
            space_left -= word.length() ;
        }
    }
}



////////////////////////////// RichConsoleStyleMessageFormatter Implementation /////////////////////
RichConsoleStyleMessageFormatter::RichConsoleStyleMessageFormatter(){
    m_showFileInfo=false;
}

void RichConsoleStyleMessageFormatter::formatMessage(const Message& m, std::ostream& out)
{
    int psize = s_messageTypePrefixes[m.type()].size() ;

    out << s_messageTypeColors[m.type()] << s_messageTypePrefixes[m.type()];

    SofaComponentInfo* nfo = dynamic_cast<SofaComponentInfo*>(m.componentInfo().get()) ;
    if( nfo != nullptr )
    {
        const std::string& classname= nfo->sender();
        const std::string& name = nfo->name();
        psize +=classname.size()+name.size()+5 ;
        out << Console::BLUE << "[" << classname << "(" << name << ")] ";
    }
    else
    {
        psize +=m.sender().size()+3 ;
        out << Console::BLUE << "[" << m.sender()<< "] ";
    }

    out << Console::DEFAULT_COLOR;

    /// Format & align the text and write the result into 'out'.
    simpleFormat(psize , m.message().str(), Console::getColumnCount()-psize, out) ;

    if(m_showFileInfo && m.fileInfo()){
        std::stringstream buf;
        std::string emptyspace(psize, ' ') ;
        buf << "Emitted from '" << m.fileInfo()->filename << "' line " << m.fileInfo()->line ;
        out << "\n" << Console::DEFAULT_CODE << Console::DEFAULT_COLOR << emptyspace ;
        simpleFormat(psize , buf.str(), Console::getColumnCount()-psize, out) ;
    }

    ///Restore the console rendering attribute.
    out << Console::DEFAULT_COLOR;
    out << Console::DEFAULT_CODE;
    out << std::endl ;
}

} // richconsolestylmessageformatter
} // logging
} // helper
} // sofa

