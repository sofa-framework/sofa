/***************************************************************************
                              AbortException.h
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2004/08/11 14:05:24 $
    Version           : $Revision: 1.4 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef ABORT_EXCEPTION_H
#define ABORT_EXCEPTION_H

#include <string>
#include <exception>

/** Exception class to handle abortion in the xmlReading
  * Particularly useful to handle constructor's abortion.
  *
  * @author Emmanuel Promayon
  * $Revision: 1.4 $
  */
class AbortException : public std::exception  {
public:
    /// default constructor: give the reason for the exception
    AbortException(std::string s) {
        reason = s;
    }
    virtual ~AbortException() throw() {}
    ;

    /// get the detailed reason from the exception
    virtual const char* what() {
        return reason.c_str();
    }
private:
    std::string reason;
};

#endif
