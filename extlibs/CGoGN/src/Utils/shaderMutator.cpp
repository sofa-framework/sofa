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

#include "Utils/shaderMutator.h"

namespace CGoGN
{

namespace Utils
{


/***********************************************
 *
 * 		Public Section
 *
 ***********************************************/


ShaderMutator::ShaderMutator(const std:: string& shaderName, const std::string& vertShaderSrc, const std::string& fragShaderSrc, const std::string& geomShaderSrc)
{
	// Store the shader name
	m_shaderName = shaderName;

	// Store the shader source codes
	m_vShaderMutation = vertShaderSrc;
	m_fShaderMutation = fragShaderSrc;
	m_gShaderMutation = geomShaderSrc;
}

bool ShaderMutator::containsVariableDeclaration(shaderSrcType srcType, const std::string& variableName)
{
	bool result = false;

	switch (srcType)
	{
		case VERTEX_SHADER :
			result = srcContainsVariableDeclaration(variableName, m_vShaderMutation);
			break;

		case FRAGMENT_SHADER :
			result = srcContainsVariableDeclaration(variableName, m_fShaderMutation);
			break;

		case GEOMETRY_SHADER :
			result = srcContainsVariableDeclaration(variableName, m_gShaderMutation);
			break;
	}

	return result;
}

bool ShaderMutator::setMinShadingLanguageVersion(shaderSrcType srcType, int version)
{
	bool result = false;

	switch (srcType)
	{
		case VERTEX_SHADER :
			result = srcSetMinShadingLanguageVersion(version, m_vShaderMutation);
			break;

		case FRAGMENT_SHADER :
			result = srcSetMinShadingLanguageVersion(version, m_fShaderMutation);
			break;

		case GEOMETRY_SHADER :
			result = srcSetMinShadingLanguageVersion(version, m_gShaderMutation);
			break;
	}

	return result;
}

bool ShaderMutator::changeIntConstantValue(shaderSrcType srcType, const std::string& constantName, int newVal)
{
	switch (srcType)
	{
		case VERTEX_SHADER :
			if (!srcChangeIntConstantValue(newVal, constantName, m_vShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::changeIntConstantValue : "
				<< "Unable to change int constant value in vertex shader of "
				<< m_shaderName
				<< ". Constant declaration not found"
				<< CGoGNendl;

				return false;
			}
			break;

		case FRAGMENT_SHADER :
			if (!srcChangeIntConstantValue(newVal, constantName, m_fShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::changeIntConstantValue : "
				<< "Unable to change int constant value in fragment shader of "
				<< m_shaderName
				<< ". Constant declaration not found"
				<< CGoGNendl;

				return false;
			}
			break;

		case GEOMETRY_SHADER :
			if (!srcChangeIntConstantValue(newVal, constantName, m_gShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::changeIntConstantValue : "
				<< "Unable to change int constant value in geometry shader of "
				<< m_shaderName
				<< ". Constant declaration not found"
				<< CGoGNendl;

				return false;
			}
			break;
	}

	return true;
}

bool ShaderMutator::changeFloatConstantValue(shaderSrcType srcType, const std::string& constantName, float newVal)
{
	switch (srcType)
	{
		case VERTEX_SHADER :
			if (!srcChangeFloatConstantValue(newVal, constantName, m_vShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::changeFloatConstantValue : "
				<< "Unable to change float constant value in vertex shader of "
				<< m_shaderName
				<< ". Constant declaration not found"
				<< CGoGNendl;

				return false;
			}
			break;

		case FRAGMENT_SHADER :
			if (!srcChangeFloatConstantValue(newVal, constantName, m_fShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::changeFloatConstantValue : "
				<< "Unable to change float constant value in fragment shader of "
				<< m_shaderName
				<< ". Constant declaration not found"
				<< CGoGNendl;

				return false;
			}
			break;

		case GEOMETRY_SHADER :
			if (!srcChangeFloatConstantValue(newVal, constantName, m_gShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::changeFloatConstantValue : "
				<< "Unable to change float constant value in geometry shader of "
				<< m_shaderName
				<< ". Constant declaration not found"
				<< CGoGNendl;

				return false;
			}
			break;
	}

	return true;
}

bool ShaderMutator::insertCodeBeforeMainFunction(shaderSrcType srcType, const std::string& insertedCode)
{
	switch (srcType)
	{
		case VERTEX_SHADER :
			if (!srcInsertCodeBeforeMainFunction(insertedCode, m_vShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::insertCodeBeforeMainFunction : "
				<< "Unable to insert source code in vertex shader of "
				<< m_shaderName
				<< ". You should check if the shader has a main function declaration"
				<< CGoGNendl;

				return false;
			}
			break;

		case FRAGMENT_SHADER :
			if (!srcInsertCodeBeforeMainFunction(insertedCode, m_fShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::insertCodeBeforeMainFunction : "
				<< "Unable to insert source code in fragment shader of "
				<< m_shaderName
				<< ". You should check if the shader has a main function declaration"
				<< CGoGNendl;

				return false;
			}
			break;

		case GEOMETRY_SHADER :
			if (!srcInsertCodeBeforeMainFunction(insertedCode, m_gShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::insertCodeBeforeMainFunction : "
				<< "Unable to insert source code in geometry shader of "
				<< m_shaderName
				<< ". You should check if the shader has a main function declaration"
				<< CGoGNendl;

				return false;
			}
			break;
	}

	return true;
}

bool ShaderMutator::insertCodeAtMainFunctionBeginning(shaderSrcType srcType, const std::string& insertedCode)
{
	switch (srcType)
	{
		case VERTEX_SHADER :
			if (!srcInsertCodeAtMainFunctionBeginning(insertedCode, m_vShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::insertCodeAtMainFunctionBeginnning : "
				<< "Unable to insert source code in vertex shader of "
				<< m_shaderName
				<< ". You should check if the shader has a main function declaration"
				<< CGoGNendl;

				return false;
			}
			break;

		case FRAGMENT_SHADER :
			if (!srcInsertCodeAtMainFunctionBeginning(insertedCode, m_fShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::insertCodeAtMainFunctionBeginnning : "
				<< "Unable to insert source code in fragment shader of "
				<< m_shaderName
				<< ". You should check if the shader has a main function declaration"
				<< CGoGNendl;

				return false;
			}
			break;

		case GEOMETRY_SHADER :
			if (!srcInsertCodeAtMainFunctionBeginning(insertedCode, m_gShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::insertCodeAtMainFunctionBeginnning : "
				<< "Unable to insert source code in geometry shader of "
				<< m_shaderName
				<< ". You should check if the shader has a main function declaration"
				<< CGoGNendl;

				return false;
			}
			break;
	}

	return true;
}

bool ShaderMutator::insertCodeAtMainFunctionEnd(shaderSrcType srcType, const std::string& insertedCode)
{
	switch (srcType)
	{
		case VERTEX_SHADER :
			if (!srcInsertCodeAtMainFunctionEnd(insertedCode, m_vShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::insertCodeAtMainFunctionEnd : "
				<< "Unable to insert source code in vertex shader of "
				<< m_shaderName
				<< ". You should check if the shader has a main function declaration "
				<< "and as many '{' as '}' in main"
				<< CGoGNendl;

				return false;
			}
			break;

		case FRAGMENT_SHADER :
			if (!srcInsertCodeAtMainFunctionEnd(insertedCode, m_fShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::insertCodeAtMainFunctionEnd : "
				<< "Unable to insert source code in fragment shader of "
				<< m_shaderName
				<< ". You should check if the shader has a main function declaration "
				<< "and as many '{' as '}' in main"
				<< CGoGNendl;

				return false;
			}
			break;

		case GEOMETRY_SHADER :
			if (!srcInsertCodeAtMainFunctionEnd(insertedCode, m_gShaderMutation))
			{
				CGoGNerr
				<< "ERROR - "
				<< "ShaderMutator::insertCodeAtMainFunctionEnd : "
				<< "Unable to insert source code in geometry shader of "
				<< m_shaderName
				<< ". You should check if the shader has a main function declaration "
				<< "and as many '{' as '}' in main"
				<< CGoGNendl;

				return false;
			}
			break;
	}

	return true;
}


/***********************************************
 *
 * 		Private Section
 *
 ***********************************************/


bool ShaderMutator::srcIsCommented(size_t pos, const std::string& src)
{
	// Verify that the given position is not out of the source
	if (pos >= src.length())
	{
		CGoGNerr
		<< "ERROR - "
		<< "ShaderMutator::srcIsCommented : "
		<< "Given position is out of range"
		<< CGoGNendl;
		return false;
	}
	
	// Look backward in the source to see if there is any comment symbol (// or /* */)
	
	// First look for one-line comments
	if (srcIsOneLineCommented(pos, src))
		return true;
	
	// Now look for multi-line comments 
	for (size_t i = pos; i > 0; i--)
	{	
		if (src[i] == '/')
		{
			// End of multi-line comment
			if (src[i-1] == '*')
			{
				// Verify that the end of multi-line comment is not one-line commented !
				if (!srcIsOneLineCommented(i, src))
					return false;
			}
		}
		else if (src[i] == '*')
		{
			// Beginning of multi-line comment
			if (src[i-1] == '/')
			{
				// Verify that the beginning of multi-line comment is not one-line commented !
				if (!srcIsOneLineCommented(i, src))
					return true;
			}
		}
	}
	
	// No one-line or multi-line comments were found
	return false;
	
}

bool ShaderMutator::srcIsOneLineCommented(size_t pos, const std::string& src)
{
	// Verify that the given position is not out of the source
	if (pos >= src.length())
	{
		CGoGNerr
		<< "ERROR - "
		<< "ShaderMutator::srcIsOneLineCommented : "
		<< "Given position is out of range"
		<< CGoGNendl;
		return false;
	}
	
	// Look backward in the source to see if there is any "//"
	for (size_t i = pos; i > 0; i--)
	{
		// As soon as a '\n' is found, any other "//" will not affect this line anymore
		if (src[i] == '\n')
			return false;
		// Else if a '/' is found, look if it is followed by another
		else if (src[i] == '/')
			if (src[i-1] == '/')
				return true;
	}
	
	// No one-line comments were found
	return false;
}

bool ShaderMutator::srcContainsVariableDeclaration(const std::string& variableName, std::string& src)
{
	// Regular expression for variable declaration
	// <',' OR white-space[1 or more times]> <variableName> <',' OR ';' OR white-space>
	boost::regex var_re("(,|\\s+)" + variableName + "(,|;|\\s)");
	
	// Matches results
	boost::match_results <std::string::iterator> matches;
	
	// Search for the first expression that matches and isn't commented
	std::string::iterator start = src.begin();
	std::string::iterator end = src.end();
	while (regex_search(start, end, matches, var_re, boost::format_first_only))
	{
		// Start position of the match
		size_t startPosition = std::distance(src.begin(), matches[0].first);
		
		// Finish if the matched variable is the good one (i.e. not commented)
		if (!srcIsCommented(startPosition, src))
			return true;
		// Else continue to search for it after last match
		else
			start = matches[0].second;
	}
	
	// At this point no correct match was found
	return false;
}

bool ShaderMutator::srcSetMinShadingLanguageVersion(int version, std::string& modifiedSrc)
{
	// Regular expression for shading language version
	// <#version> <white-space>[1 or more times] <digit>[1 or more times]
	boost::regex version_re("#version\\s+(\\d+)");

	// Matches results
	boost::match_results <std::string::iterator> matches;

	// Build the version string
	std::string versionStr;
	std::stringstream ss;
	ss << version;
	versionStr = ss.str();

	// Search for the first expression that matches and isn't commented
	std::string::iterator start = modifiedSrc.begin();
	std::string::iterator end = modifiedSrc.end();
	while (regex_search(start, end, matches, version_re, boost::format_first_only))
	{
		// Start position of the match
		size_t startPosition = std::distance(modifiedSrc.begin(), matches[0].first);

		// Change the version number if the matched "#version ..." is the good one (i.e. not commented)
		if (!srcIsCommented(startPosition, modifiedSrc))
		{
			// The submatch Match[1] should be the version number
			std::string oldVersion(matches[1].first, matches[1].second);
			int oldVersionValue = atoi(oldVersion.c_str());
			size_t oldVersionLength = oldVersion.length();
			size_t oldVersionPosition = std::distance(modifiedSrc.begin(), matches[1].first);

			// Replace the old version value only if it is lower than 'version'
			if (oldVersionValue < version)
			{
				modifiedSrc.replace(oldVersionPosition, oldVersionLength, versionStr);
				return true;
			}
			else
				return false;
		}
		// Else continue to search for it after last match
		else
		{
			start = matches[0].second;
		}
	}

	// At this point no correct match was found : insert directly the "#version ..." line
	std::string versionLineStr = "#version " + versionStr + "\n";
	modifiedSrc.insert(0, versionLineStr);

	return true;
}

bool ShaderMutator::srcChangeIntConstantValue(int newVal, const std::string& constantName, std::string& modifiedSrc)
{
	// Regular expression for constant expression
	// <#define> <white-space>[1 or more times] <constant name> <white-space>[1 or more times] <digit>[1 or more times]
	boost::regex const_re("#define\\s+" + constantName + "\\s+(\\d+)");

	// Matches results
	boost::match_results <std::string::iterator> matches;

	// Build the constant value string
	std::string newValStr;
	std::stringstream ss;
	ss << newVal;
	newValStr = ss.str();

	// Search for the first expression that matches and isn't commented
	std::string::iterator start = modifiedSrc.begin();
	std::string::iterator end = modifiedSrc.end();
	while (regex_search(start, end, matches, const_re, boost::format_first_only))
	{
		// Start position of the match
		size_t startPosition = std::distance(modifiedSrc.begin(), matches[0].first);

		// Change the constant value if the matched "#define ..." is the good one (i.e. not commented)
		if (!srcIsCommented(startPosition, modifiedSrc))
		{
			// The submatch Match[1] should be the old constant value
			std::string oldValStr(matches[1].first, matches[1].second);
			size_t oldValLength = oldValStr.length();
			size_t oldValPosition = std::distance(modifiedSrc.begin(), matches[1].first);

			// Replace the old constant value
			modifiedSrc.replace(oldValPosition, oldValLength, newValStr);
			return true;
		}
		// Else continue to search for it after last match
		else
		{
			start = matches[0].second;
		}
	}

	// At this point no correct match was found
	return false;
}

bool ShaderMutator::srcChangeFloatConstantValue(float newVal, const std::string& constantName, std::string& modifiedSrc)
{
	// Regular expression for constant expression
	// <#define> <white-space>[1 or more times] <constant name> <white-space>[1 or more times]
	// <digit>[1 or more times] <.>[0 or 1 time] <digit>[0 or more times]
	boost::regex const_re("#define\\s+" + constantName + "\\s+(\\d+\\.?\\d*)");

	// Matches results
	boost::match_results <std::string::iterator> matches;

	// Build the constant value string
	std::string newValStr;
	std::stringstream ss;
	ss << newVal;
	newValStr = ss.str();

	// Search for the first expression that matches and isn't commented
	std::string::iterator start = modifiedSrc.begin();
	std::string::iterator end = modifiedSrc.end();
	while (regex_search(start, end, matches, const_re, boost::format_first_only))
	{
		// Start position of the match
		size_t startPosition = std::distance(modifiedSrc.begin(), matches[0].first);

		// Change the constant value if the matched "#define ..." is the good one (i.e. not commented)
		if (!srcIsCommented(startPosition, modifiedSrc))
		{
			// The submatch Match[1] should be the old constant value
			std::string oldValStr(matches[1].first, matches[1].second);
			size_t oldValLength = oldValStr.length();
			size_t oldValPosition = std::distance(modifiedSrc.begin(), matches[1].first);

			// Replace the old constant value
			modifiedSrc.replace(oldValPosition, oldValLength, newValStr);
			return true;
		}
		// Else continue to search for it after last match
		else
		{
			start = matches[0].second;
		}
	}

	// At this point no correct match was found
	return false;
}

bool ShaderMutator::srcInsertCodeBeforeMainFunction(const std::string& insertedCode, std::string& modifiedSrc)
{
	// Regular expression for main function
	// <void> <white-space>[1 or more times] <main> <white-space>[0 or more times] <'('>
	boost::regex main_re("(void)\\s+(main)\\s*\\(");
	
	// Matches results
	boost::match_results <std::string::iterator> matches;
	
	// Search for the first expression that matches and isn't commented
	std::string::iterator start = modifiedSrc.begin();
	std::string::iterator end = modifiedSrc.end();
	while (regex_search(start, end, matches, main_re, boost::format_first_only))
	{
		// Start position of the match
		size_t startPosition = std::distance(modifiedSrc.begin(), matches[0].first);
		
		// Insert and finish if the matched "main" is the good one (i.e. not commented)
		if (!srcIsCommented(startPosition, modifiedSrc))
		{
			modifiedSrc.insert(startPosition, insertedCode);
			return true;
		}
		// Else continue to search for it after last match
		else
		{
			start = matches[0].second;
		}
	}
	
	// At this point no correct match was found
	return false;
}

bool ShaderMutator::srcInsertCodeAtMainFunctionBeginning(const std::string& insertedCode, std::string& modifiedSrc)
{
	// Regular expression for main function
	// <void> <white-space>[1 or more times] <main> <white-space>[0 or more times]
	// <'('> <white-space>[0 or more times] <')'>
	// <white-space>[0 or more times] <'{'>
	boost::regex main_re("(void)\\s+(main)\\s*\\(\\s*\\)\\s*\\{");
	
	// Matches results
	boost::match_results <std::string::iterator> matches;
	
	// Search for the first expression that matches and isn't commented
	std::string::iterator start = modifiedSrc.begin();
	std::string::iterator end = modifiedSrc.end();
	while (regex_search(start, end, matches, main_re, boost::format_first_only))
	{
		// Start position of the match
		size_t startPosition = std::distance(modifiedSrc.begin(), matches[0].first);
		
		// End position of the match
		size_t endPosition = std::distance(modifiedSrc.begin(), matches[0].second);
		
		// Insert and finish if the matched "main" is the good one (i.e. not commented)
		if (!srcIsCommented(startPosition, modifiedSrc))
		{
			modifiedSrc.insert(endPosition, insertedCode);
			return true;
		}
		// Else continue to search for it after last match
		else
		{
			start = matches[0].second;
		}
	}
	
	// At this point no correct match was found
	return false;
}

bool ShaderMutator::srcInsertCodeAtMainFunctionEnd(const std::string& insertedCode, std::string& modifiedSrc)
{
	// Regular expression for main function
	// <void> <white-space>[1 or more times] <main> <white-space>[0 or more times]
	// <'('> <white-space>[0 or more times] <')'>
	// <white-space>[0 or more times] <'{'>
	boost::regex main_re("(void)\\s+(main)\\s*\\(\\s*\\)\\s*\\{");
	
	// Matches results
	boost::match_results <std::string::iterator> matches;
	
	// Search for the first expression that matches and isn't commented
	std::string::iterator start = modifiedSrc.begin();
	std::string::iterator end = modifiedSrc.end();
	size_t mainFirstBracePos = 0;  // The aim is first to find this position
	while (regex_search(start, end, matches, main_re, boost::format_first_only) && (mainFirstBracePos == 0))
	{
		// Start position of the match
		size_t startPosition = std::distance(modifiedSrc.begin(), matches[0].first);
		
		// End position of the match
		size_t endPosition = std::distance(modifiedSrc.begin(), matches[0].second);
		
		// Get the main first brace position if the matched "main" is the good one (i.e. not commented)
		if (!srcIsCommented(startPosition, modifiedSrc))
			mainFirstBracePos = endPosition;
		// Else continue to search for it after last match
		else
			start = matches[0].second;
	}
	
	// If mainFirstBracePos is still zero, no correct match was found
	if (mainFirstBracePos == 0)
		return false;
	
	// Else, it is now possible to count the opening and closing braces till the final closing brace of the main function is reached
	int bracesCounter = 1;  // =1 since the first opening brace is counted in, it will be =0 when the final closing brace is reached
	size_t closestBracePos = mainFirstBracePos;
	size_t closestOpeningBracePos;
	size_t closestClosingBracePos;
	while (bracesCounter != 0)
	{
		closestOpeningBracePos = modifiedSrc.find_first_of('{', closestBracePos + 1);
		// If this brace appears to be commented, try to get the next one
		while ((closestOpeningBracePos != std::string::npos) && srcIsCommented(closestOpeningBracePos, modifiedSrc))
			closestOpeningBracePos = modifiedSrc.find_first_of('{', closestOpeningBracePos + 1);
		
		closestClosingBracePos = modifiedSrc.find_first_of('}', closestBracePos + 1);
		// If this brace appears to be commented, try to get the next one
		while ((closestClosingBracePos != std::string::npos) && srcIsCommented(closestClosingBracePos, modifiedSrc))
			closestClosingBracePos = modifiedSrc.find_first_of('}', closestClosingBracePos + 1);

		// Happens if there is not enough "}" for the corresponding "{"
		if (closestClosingBracePos == std::string::npos)
			return false;

		// Refresh the closest brace position, and increment or decrement the counter
		if (closestClosingBracePos < closestOpeningBracePos)
		{
			closestBracePos = closestClosingBracePos;
			bracesCounter -= 1;
		}
		else
		{
			closestBracePos = closestOpeningBracePos;
			bracesCounter += 1;
		}
	}

	// We should now have the final '}' of the main function
	size_t mainLastBracePos = closestBracePos;
	
	// Insert the source there
	modifiedSrc.insert(mainLastBracePos, insertedCode);

	size_t posPb = modifiedSrc.find_last_of(';');
	if (modifiedSrc.substr(posPb-6,7) == "#endif;")
		modifiedSrc[posPb] = '\n';

	return true;
}

} // namespace Utils

} // namespace CGoGN
