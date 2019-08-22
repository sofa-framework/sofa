# -*- coding: utf-8 -*-
#/******************************************************************************
#*       SOFA, Simulation Open-Framework Architecture, development version     *
#*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
#*                                                                             *
#* This library is free software; you can redistribute it and/or modify it     *
#* under the terms of the GNU Lesser General Public License as published by    *
#* the Free Software Foundation; either version 2.1 of the License, or (at     *
#* your option) any later version.                                             *
#*                                                                             *
#* This library is distributed in the hope that it will be useful, but WITHOUT *
#* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
#* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
#* for more details.                                                           *
#*                                                                             *
#* You should have received a copy of the GNU Lesser General Public License    *
#* along with this library; if not, write to the Free Software Foundation,     *
#* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
#*******************************************************************************
#*                              SOFA :: Framework                              *
#*                                                                             *
#* Authors: damien.marchal@univ-lille1.fr Copyright (C) CNRS                   *
#*                                                                             *
#* Contact information: contact@sofa-framework.org                             *
#******************************************************************************/

import traceback
import imp
import types
import gc

def mimport(modulename, filename):
    f = open(filename, 'r')
    return imp.load_module(modulename, f, filename, (modulename, 'r', imp.PY_SOURCE))

def mreload(modulepath):
        try:
           #print("============== Updating a module ==========================")
           #print("LiveCoding: updating module from: "+modulepath)
           m=mimport(modulepath, modulepath)

           for obj in gc.get_objects():
              try:
                      if isinstance(obj, object) and hasattr(obj, "__class__") and hasattr(m, str(obj.__class__.__name__) ) and  hasattr(obj, "__module__") and str(obj.__module__) in ["__main__", str(m.__file__)]:
                         obj.__class__=getattr(m, str(obj.__class__.__name__))

                         if hasattr(obj, "onRecompile"):
                                obj.onRecompile()
                         continue

                      if isinstance(obj, types.ModuleType):

                        if not hasattr(m, "__file__"):
                                continue;
                        if not hasattr(obj, "__file__"):
                                continue;

                        if obj.__file__.rstrip('c') == m.__file__:
                                for f in dir(obj):
                                        if isinstance(getattr(obj,f), types.FunctionType):
                                                #print("Patching function: "+f.__name__)
                                                setattr(obj, f, getattr(m, f))

              except:
                       traceback.print_exc()
           #print("================ End update ======================================")
        except:
           traceback.print_exc()

def onReimpAFile(filename):
        mreload(filename)

