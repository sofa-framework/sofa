/***************************************************************************
                                 Properties.h
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2004/08/11 14:05:24 $
    Version           : $Revision: 1.8 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef PROPERTIES_H
#define PROPERTIES_H

#include <libxml/tree.h>
//#include <libxml/parser.h>

#include "PhysicalModelIO.h"
#include <string>
#include <sstream>
#include <map>
class PhysicalModel;

/** Describes the properties common to all structures and components.
  * $Revision: 1.8 $
 */
class Properties {

public:
    /** A nice simple constructor, with a given name */
    Properties(const std::string n="");
    /** Another nice constructor, with the PM and a name */
    Properties(PhysicalModel *, const std::string n="");
    /** The default destructor */
    virtual ~Properties();

    /// get the name (be careful, this method DOES NOT return a copy, so you got the direct ptr to the name!!!)
    std::string getName() const;

    /// set the name (use the string = operator)
    void setName(std::string);

    /// set the physical model
    void setPhysicalModel(PhysicalModel *);

    /// get the physical model
    PhysicalModel * getPhysicalModel() const;

	/// convert the xml node parameters to data fields
	void domToFields(xmlNodePtr node);

	///fields accessors
	double getDouble(std::string attName);
	int getInt(std::string attName);
	bool getBool(std::string attName);
	std::string getString(std::string attName);
	void get(std::string attName, std::string &attVal);

	///fields affectors
	void set(std::string attName, double val);
	void set(std::string attName, int val);
	void set(std::string attName, bool val);
	void set(std::string attName, std::string val);


protected : 
	std::map<std::string, std::string> fields;

private:
    std::string name;
    PhysicalModel *myPM;

};

inline double Properties::getDouble(std::string attName){
	std::map<std::string, std::string>::iterator it = fields.find(attName);
	if (it != fields.end())
		return atof( it->second.c_str());
	else
		return 0.0;			
}

inline int Properties::getInt(std::string attName){
	std::map<std::string, std::string>::iterator it = fields.find(attName);
	if (it != fields.end())
		return atoi( it->second.c_str());
	else
		return 0;			
}

inline bool Properties::getBool(std::string attName){
	std::map<std::string, std::string>::iterator it = fields.find(attName);

	if(it == fields.end() || it->second =="false" || it->second =="0")
		return false;
	else
		return true;
}

inline std::string Properties::getString(std::string attName){
	std::map<std::string, std::string>::iterator it = fields.find(attName);
	if (it != fields.end())
		return it->second;
	else
		return "";			
}

inline void Properties::get(std::string attName, std::string &attVal){ 
	std::map<std::string, std::string>::iterator it = fields.find(attName);
	if (it != fields.end())
		attVal = it->second;
	else
		attVal = "";
}

inline void Properties::set(std::string attName, double val){ 
	std::ostringstream oss;
	oss << val;
	std::map<std::string, std::string>::iterator it = fields.find(attName);
	if (it != fields.end())
		it->second = oss.str();
	else
		fields.insert(std::pair<std::string, std::string>(attName, oss.str()));
}

inline void Properties::set(std::string attName, int val){ 
	std::ostringstream oss;
	oss << val;
	std::map<std::string, std::string>::iterator it = fields.find(attName);
	if (it != fields.end())
		it->second = oss.str() ;
	else
		fields.insert(std::pair<std::string, std::string>(attName, oss.str()));
}

inline void Properties::set(std::string attName, bool val){ 
	std::ostringstream oss;
	oss << val;
	std::map<std::string, std::string>::iterator it = fields.find(attName);
	if (it != fields.end())
		it->second = oss.str() ;
	else
		fields.insert(std::pair<std::string, std::string>(attName, oss.str()));
}

inline void Properties::set(std::string attName, std::string val){ 
	std::map<std::string, std::string>::iterator it = fields.find(attName);
	if (it != fields.end())
		it->second = val ;
	else
		fields.insert(std::pair<std::string, std::string>(attName, val));
}

inline std::string Properties::getName() const {
    return name;
}

inline void Properties::setName(std::string n) {
    name = n;
}

inline void Properties::setPhysicalModel(PhysicalModel *pm) {
    myPM = pm;
}

inline PhysicalModel * Properties::getPhysicalModel() const {
    return myPM;
}

#endif //PROPERTIES_H
