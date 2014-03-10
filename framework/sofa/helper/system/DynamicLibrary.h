/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_SYSTEM_DYNAMICLIBRARY_H
#define SOFA_HELPER_SYSTEM_DYNAMICLIBRARY_H

#include <sofa/helper/helper.h>
#include <boost/shared_ptr.hpp>
#include <iostream>


namespace sofa
{
namespace helper
{
namespace system
{


/**
   This class provides wrappers around dynamic library facilities, which are
   system-specific.
*/
class SOFA_HELPER_API DynamicLibrary
{
public:

    /**
       A handle to a dynamic library.
    */
    class Handle {
        friend class DynamicLibrary;
    public:
        Handle();
        Handle(const Handle& that);
        /// Check if the handle is valid, i.e. if load() was successful.
        bool isValid() const;
    private:
        void * m_realHandle;
        boost::shared_ptr<std::string> m_filename;
        Handle(const std::string& filename, void *handle);
    };

    /// Load a dynamic library
    ///
    /// @return a handle, that must be unloaded with unload().
    /// Use Handle::isValid() to know if the loading was successful.
    static Handle load(const std::string& filename);

    /// Unload a dynamic library loaded with load().
    ///
    /// @return 0 on success, and nonzero on error.
    static int unload(Handle handle);

    /// Get the address of a symbol
    ///
    /// @return a pointer to the symbol if it was found, or NULL on error.
    static void * getSymbolAddress(Handle handle, const std::string& symbol);

    /// Get the message for the most recent error that occurred from load(),
    /// unload() or getSymbolAddress().
    ///
    /// @return the error message, or an empty string if no errors have occurred
    /// since initialization or since it was last called.
    static std::string getLastError();

    /// System-specific file extension for a dynamic library (e.g. ".so")
    static const std::string extension;

private:
    static std::string m_lastError;
    static void fetchLastError();
};


}

}

}

#endif // SOFA_HELPER_SYSTEM_DYNAMICLIBRARY_H
