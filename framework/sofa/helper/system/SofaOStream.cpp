/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
 *                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
 *                              SOFA :: Framework                              *
 *                                                                             *
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/

#include <sofa/helper/system/SofaOStream.h>
#include <sstream>
#include <iostream>

namespace sofa
{

namespace helper
{

namespace system
{

SofaOStream::SofaOStream(const bool &output):outputConsole(output)
{
    serr = new std::ostringstream();
    sout = new std::ostringstream();
}

void SofaOStream::processStream(std::ostream& out)
{
    if (out == *serr)
    {
        *serr << "\n";
        // if (isOutputConsole())
        std::cerr<< "WARNING[" << nameComponent << "(" << nameClass << ")]: "<<serr->str();
        warnings += serr->str();
        serr->str("");
    }
    else if (out == *sout)
    {
        *sout << "\n";
        if (outputConsole) std::cout<< "[" << nameComponent << "(" << nameClass << ")]: "<< sout->str();
        outputs += sout->str();
        sout->str("");
    }
}

std::string SofaOStream::getWarnings() const
{
    return warnings;
}

std::string SofaOStream::getOutputs() const
{
    return outputs;
}

void SofaOStream::clearWarnings() {warnings.clear();}
void SofaOStream::clearOutputs() {outputs.clear();}
}
}
}

