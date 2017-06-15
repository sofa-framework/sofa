/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_HELPER_SYSTEM_DYNAMICLIBRARY_H
#define SOFA_HELPER_SYSTEM_DYNAMICLIBRARY_H

#include <sofa/helper/helper.h>
#include <memory>
#include <iostream>


namespace sofa
{
namespace helper
{
namespace system
{


/**
   @brief Wrapper around the dynamic library facilities of the operating system.
*/
class SOFA_HELPER_API DynamicLibrary
{
public:

    /// A handle to a dynamic library.
    class SOFA_HELPER_API Handle {
        friend class DynamicLibrary;
    public:
        /// Default constructor: invalid handle.
        Handle();
        /// Copy constructor.
        Handle(const Handle& that);
        /// Check if the handle is valid, i.e. if load() was successful.
        bool isValid() const;
        /// Get the filename of the library.
        const std::string& filename() const;
    private:
        void * m_realHandle;
        std::shared_ptr<std::string> m_filename;
        Handle(const std::string& filename, void *handle);
    };

    /// @brief Load a dynamic library.
    ///
    /// @param filename The library to load.
    /// @return A handle, that must be unloaded with unload().
    /// Use Handle::isValid() to know if the loading was successful.
    static Handle load(const std::string& filename);

    /// @brief Unload a dynamic library loaded with load().
    ///
    /// @param handle The handle of a library.
    /// @return 0 on success, and nonzero on error.
    static int unload(Handle handle);

    /// @brief Get the address of a symbol.
    ///
    /// @param handle The handle of a library.
    /// @param symbol The symbol to look for.
    /// @return A pointer to the symbol if it was found, or NULL on error.
    static void * getSymbolAddress(Handle handle, const std::string& symbol);

    /// @brief Get the message for the most recent error that occurred from load(), unload() or getSymbolAddress().
    ///
    /// @return The error message, or an empty string if no errors have occurred
    /// since initialization or since it was last called.
    static std::string getLastError();

    /// System-specific file extension for a dynamic library (e.g. "so").
    static const std::string extension;

    /// System-specific file prefix for a dynamic library (e.g. "lib").
    static const std::string prefix;

private:
    static std::string m_lastError;
    static void fetchLastError();
};


}

}

}

#endif // SOFA_HELPER_SYSTEM_DYNAMICLIBRARY_H
