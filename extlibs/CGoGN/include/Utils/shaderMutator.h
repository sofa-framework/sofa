/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef _CGoGN_SHADERMUTATOR_H_
#define _CGoGN_SHADERMUTATOR_H_

#undef tolower
#undef toupper

#include "Utils/cgognStream.h"
#include <boost/regex.hpp>
#include <string>
#include <sstream>

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

class CGoGN_UTILS_API ShaderMutator
{
	
public:

	/// enum used to choose which shader src type to modify
	enum shaderSrcType { VERTEX_SHADER, FRAGMENT_SHADER, GEOMETRY_SHADER };

	/**
	 * constructor
	 * @param vertShaderSrc vertex shader source to store
	 * @param fragShaderSrc fragment shader source to store
	 * @param geomShaderSrc geometry shader source to store (optional)
	 */
	 ShaderMutator(const std::string& shaderName, const std::string& vertShaderSrc, const std::string& fragShaderSrc, const std::string& geomShaderSrc = "");
	 
	 /**
	  * checks if a variable is declared in the shader source
	  * @param srcType shader source to use
	  * @param variableName variable to search for
	  * @return true if the variable was declared
	  */
	 bool containsVariableDeclaration(shaderSrcType srcType, const std::string& variableName);
	 
	 /**
	  * sets or changes shading language version in the shader source
	  * - only if the current version is lower
	  * @param srcType shader source to use
	  * @param version version to set (110, 120, 150...)
	  * @return true if the version was set or changed
	  */
	 bool setMinShadingLanguageVersion(shaderSrcType srcType, int version);

	 /**
	  * changes int constant value in the shader source
	  * @param srcType shader source to use
	  * @param newVal new constant value
	  * @return true on success
	  */
	 bool changeIntConstantValue(shaderSrcType srcType, const std::string& constantName, int newVal);

	 /**
	  * changes float constant value in the shader source
	  * @param srcType shader source to use
	  * @param newVal new constant value
	  * @return true on success
	  */
	 bool changeFloatConstantValue(shaderSrcType srcType, const std::string& constantName, float newValue);

	 /**
	  * inserts code before main function in the shader source
	  * @param srcType shader source to use
	  * @param insertedCode source code to insert into shader
	  * @return true on success
	  */
	 bool insertCodeBeforeMainFunction(shaderSrcType srcType, const std::string& insertedCode);
	 
	 /**
	  * inserts code at the beginning of main function in the shader source
	  * @param srcType shader source to use
	  * @param insertedCode source code to insert into shader
	  * @return true on success
	  */
	 bool insertCodeAtMainFunctionBeginning(shaderSrcType srcType, const std::string& insertedCode);
	 
	 /**
	  * inserts code at the end of main function in the shader source
	  * @warning takes the number of opening and closing braces of main function into account
	  * @param srcType shader source to use
	  * @param insertedCode source code to insert into shader
	  * @return true on success
	  */
	 bool insertCodeAtMainFunctionEnd(shaderSrcType srcType, const std::string& insertedCode);
	 
	 /// returns the modified vertex shader source code
	 std::string getModifiedVertexShaderSrc() { return m_vShaderMutation; }
	 
	 /// returns the modified fragment shader source code
	 std::string getModifiedFragmentShaderSrc() { return m_fShaderMutation; }
	 
	 /// returns the modified geometry shader source code
	 std::string getModifiedGeometryShaderSrc() { return m_gShaderMutation; }
	
private:

	/// processed shader name stored for log purpose
	std::string m_shaderName;
	
	/// modified version of the original vertex shader source code
	std::string m_vShaderMutation;
	
	/// modified version of the original fragment shader source code
	std::string m_fShaderMutation;
	
	/// modified version of the original geometry shader source code
	std::string m_gShaderMutation;
	
	/**
	 * checks if the given position is commented
	 * @param pos position in the source (like in a string)
	 * @param src source to analyze
	 * @return true if the given position is commented
	 */
	bool srcIsCommented(size_t pos, const std::string& src);
	
	/**
	 * checks if the given position is one-line commented
	 * @param pos position in the source (like in a string)
	 * @param src source to analyze
	 * @return true if the given position is one-line commented
	 */
	bool srcIsOneLineCommented(size_t pos, const std::string& src);
	
	/**
	  * checks if a variable is declared
	  * @param variableName variable to search for
	  * @param src source to analyze
	  * @return true if the variable was declared
	 */
	bool srcContainsVariableDeclaration(const std::string& variableName, std::string& src);
	
	/**
	 * sets or changes shading language version if the current version is lower
	 * @param version version to set (110, 120, 150...)
	 * @param modifiedSrc shader source code to modify
	 * @return true if the version was set or changed
	 */
	bool srcSetMinShadingLanguageVersion(int version, std::string& modifiedSrc);

	/**
	 * changes int constant value
	 * @param newVal new constant value
	 * @param constantName constant name as it is declared
	 * @param modifiedSrc shader source code to modify
	 * @return true on success
	 */
	bool srcChangeIntConstantValue(int newVal, const std::string& constantName, std::string& modifiedSrc);

	/**
	 * changes float constant value
	 * @param newVal new constant value
	 * @param constantName constant name as it is declared
	 * @param modifiedSrc shader source code to modify
	 * @return true on success
	 */
	bool srcChangeFloatConstantValue(float newVal, const std::string& constantName, std::string& modifiedSrc);

	/**
	 * inserts code before main function
	 * @param insertedCode source code to insert
	 * @param modifiedSrc shader source code to modify
	 * @return true on success
	 */
	bool srcInsertCodeBeforeMainFunction(const std::string& insertedCode, std::string& modifiedSrc);
	
	/**
	 * inserts code at the beginning of main function
	 * @param insertedCode source code to insert
	 * @param modifiedSrc shader source code to modify
	 * @return true on success
	 */
	bool srcInsertCodeAtMainFunctionBeginning(const std::string& insertedCode, std::string& modifiedSrc);

	/**
	 * inserts code at the end of main function
	 * @param insertedCode source code to insert
	 * @param modifiedSrc shader source code to modify
	 * @return true on success
	 */
	bool srcInsertCodeAtMainFunctionEnd(const std::string& insertedCode, std::string& modifiedSrc);
	
};

} // namespace Utils

} // namespace CGoGN

#endif
