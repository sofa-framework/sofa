/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* This component is open-source                                               *
*                                                                             *
* Contributors:                                                               *
*    - damien.marchal@univ-lille1.fr                                          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/


#include "RichConsoleStyleMessageFormatter.h"
#include "Message.h"

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

helper::fixed_array<std::string,Message::TypeCount> s_messageTypePrefixes;
helper::fixed_array<Console::ColorType,Message::TypeCount> s_messageTypeColors;
RichConsoleStyleMessageFormatter RichConsoleStyleMessageFormatter::s_instance;

RichConsoleStyleMessageFormatter::RichConsoleStyleMessageFormatter()
{
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
}

typedef boost::tokenizer<boost::char_separator<char> > tokenizer;



void simpleFormat(int jsize, const std::string& text, size_t line_length,
                  std::ostream& wrapped)
{
    /// space and * are separator that are returned in the token flow
    /// while "\n" is a 'hidden' separator.
    static boost::char_separator<char> sep("\n", "* '");

    std::istringstream words(text) ;
    std::string emptyspace(jsize, ' ') ;

    tokenizer tokens(text, sep) ;

    int numspaces = 0 ;
    bool beginOfLine = false ;
    bool isInItalic = false ;

    size_t space_left = line_length;
    for (tokenizer::iterator tok_iter = tokens.begin();tok_iter != tokens.end(); ++tok_iter)
    {
        const std::string& word = *tok_iter;
        if(word=="'")
        {
            if(isInItalic)
            {
                isInItalic=false;
                wrapped << Console::DEFAULT_CODE ;
            }
            else
            {
                isInItalic=true;
                wrapped << Console::ITALIC ;
            }
        }
        if(word==" ")
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
                    wrapped << emptyspace ;
                }
            }
        }

        if (space_left < word.length() + 1)
        {
            if(word.length()>line_length)
            {
                std::string first;
                size_t curidx=0;
                size_t endidx=std::min(word.length(), space_left);

                while(curidx < word.length())
                {
                    first=word.substr(curidx,endidx);

                    if(beginOfLine)
                        wrapped << emptyspace ;

                    beginOfLine=false;
                    wrapped << first ;

                    curidx+=endidx;
                    endidx=std::min(word.length()-curidx, line_length);

                    if(curidx < word.length())
                    {
                        wrapped << '\n' ;
                        beginOfLine=true;
                    }
                }
                space_left = line_length - first.length();
            }
            else
            {
                wrapped << "\n";
                wrapped << emptyspace ;
                wrapped << word ;
                space_left = line_length-word.length();
            }
        }
        else
        {
            if(beginOfLine)
                wrapped << emptyspace ;

            beginOfLine=false;
            wrapped << word;
            space_left -= word.length() ;
        }
    }
}

void RichConsoleStyleMessageFormatter::formatMessage(const Message& m, std::ostream& out)
{
    std::stringstream tmp ;
    std::ostringstream formatted;
    int psize = s_messageTypePrefixes[m.type()].size() ;

    out << s_messageTypeColors[m.type()] << s_messageTypePrefixes[m.type()];

    if (!m.sender().empty()){
        ///The +3 is for the extra '[', and '] '
        psize +=m.sender().size()+3 ;
        out << Console::BLUE << "[" << m.sender() << "] ";
    }

    out << Console::DEFAULT_COLOR;
    simpleFormat(psize , m.message().str(), Console::getColumnCount()-psize, out) ;
    out << std::endl ;
}

} // richconsolestylmessageformatter
} // logging
} // helper
} // sofa
