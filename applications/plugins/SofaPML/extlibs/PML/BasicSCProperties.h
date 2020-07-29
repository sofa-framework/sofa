/***************************************************************************
        BasicSCProperties.h  -  (was StructuralComponentProperties.h)
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:33:21 $
    Version           : $Revision: 1.7 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef BASICSCPROPERTIES_H
#define BASICSCPROPERTIES_H

#include "Properties.h"
#include <RenderingMode.h>

/**
 * A special class to manage the basic structural component properties.
 *
 * You should derive from this class a StucturalComponentProperties class and use
 * it to implement your own custom stuff.
 * This is a pure virtual class.
 *
 * @author Emmanuel Promayon
 */
class BasicSCProperties  : public Properties {
public:
    /** Default color settings
    */
    enum Color {
        DEFAULT,
        RED, /**< full complete flashy red*/
        GREEN,
        BLUE,
        GRAY,
        OTHER
    };

    /** use a Color constant to set the color */
    BasicSCProperties(PhysicalModel *p, const Color c) : Properties(p) {
        alloc();
        setColor(c);
        setMode(RenderingMode::WIREFRAME_AND_SURFACE);
    };
	BasicSCProperties(PhysicalModel * p, xmlNodePtr node);
    BasicSCProperties(PhysicalModel *p, const Color c, const std::string n) : Properties(p,n) {
        alloc();
        setColor(c);
        setMode(RenderingMode::WIREFRAME_AND_SURFACE);
    };
    /** use a double[3] array to set the color */
    BasicSCProperties(PhysicalModel *p, const double * rgb) : Properties(p)  {
        alloc();
        setRGB(rgb);
        setAlpha(1.0);
        setMode(RenderingMode::WIREFRAME_AND_SURFACE);
    };
    BasicSCProperties(PhysicalModel *p, const double * rgb, const std::string n) : Properties(p,n)  {
        alloc();
        setRGB(rgb);
        setAlpha(1.0);
        setMode(RenderingMode::WIREFRAME_AND_SURFACE);
    };
    /** use 3 floats to set the color */
    BasicSCProperties(PhysicalModel *p, const double r, const double g, const double b) : Properties(p)  {
        alloc();
        setRGB(r,g,b);
        setAlpha(1.0);
        setMode(RenderingMode::WIREFRAME_AND_SURFACE);
    };
    BasicSCProperties(PhysicalModel *p, const double r, const double g, const double b, const std::string n) : Properties(p,n)  {
        alloc();
        setRGB(r,g,b);
        setAlpha(1.0);
        setMode(RenderingMode::WIREFRAME_AND_SURFACE);
    };
    /** defaultcolor is gray */
    BasicSCProperties(PhysicalModel *p) : Properties(p)  {
        alloc();
        setColor(DEFAULT);
        setMode(RenderingMode::WIREFRAME_AND_SURFACE);
    }
    ;
    BasicSCProperties(PhysicalModel *p, const std::string n) : Properties(p,n)  {
        alloc();
        setColor(DEFAULT);
        setMode(RenderingMode::WIREFRAME_AND_SURFACE);
    };

    virtual ~BasicSCProperties() {
        delete colorRGBA;
    };

    double getRed() const;
    double getGreen() const;
    double getBlue() const;
    double getAlpha() const;
    double * getRGB() const;
    double * getRGBA() const;
    Color getColor() const;

    void setRed(const double r);
    void setGreen(const double g);
    void setBlue(const double b);
    void setAlpha(const double a);
    void setRGB(const double * rgb);
    void setRGB(const double r, const double g, const double b);
    void setRGBA(const double r, const double g, const double b, const double a);
    void setRGBA(const double * rgba);
    void setColor(Color c);

    void setMode(const RenderingMode::Mode);
    RenderingMode::Mode getMode() const;
    bool isVisible(const RenderingMode::Mode mode) const;
    void setVisible(const RenderingMode::Mode, const bool);

    /// get the string equivalent to the enum rendering mode
    std::string getModeString() const;

    /** print to an output stream in "pseaudo" XML format.
      * This method is to be implemented in the subclass (StructuralComponentProperties).
      * It HAS to call the beginXML() at the beginning and the endXML() at the end.
      */
    virtual void xmlPrint(std::ostream &) =0;

protected:
    /// write the default xml properties (beginning)
    void beginXML(std::ostream &);
    /// write the default xml properties (end)
    void endXML(std::ostream &);

private:
    double *colorRGBA;
    void alloc();
    Color color;
    RenderingMode mode;
};


inline BasicSCProperties::BasicSCProperties(PhysicalModel * p, xmlNodePtr node) :Properties(p)
{
	alloc();

	//search the name attribute
	xmlChar *pname = xmlGetProp(node, (const xmlChar*) "name");
	if(pname)
		setName((char*)pname);

	//get the pointer on color
	xmlNodePtr SCchild = node->xmlChildrenNode;
	while (SCchild && xmlStrcmp(SCchild->name,(const xmlChar*)"color")) SCchild= SCchild->next;
	if (SCchild) {
		//search the color attributes
		xmlChar *pr = xmlGetProp(SCchild, (const xmlChar*) "r");
		xmlChar *pg = xmlGetProp(SCchild, (const xmlChar*) "g");
		xmlChar *pb = xmlGetProp(SCchild, (const xmlChar*) "b");
		xmlChar *pa = xmlGetProp(SCchild, (const xmlChar*) "a");
		if (pr && pg && pb){
			setRGB(atof((char*)pr),atof((char*)pg),atof((char*)pb));
			if (pa)
				setAlpha(atof((char*)pa));
			else
				setAlpha(1.0);
		}
		else setColor(DEFAULT);
	}
	else setColor(DEFAULT);

	//search the renderingmode attribute
	xmlChar *pmode = xmlGetProp(node, (const xmlChar*) "mode");
	if(pmode){
		if(!xmlStrcmp(pmode, (const xmlChar*)"NONE")) setMode(RenderingMode::NONE);
		if(!xmlStrcmp(pmode, (const xmlChar*)"POINTS")) setMode(RenderingMode::POINTS);
		if(!xmlStrcmp(pmode, (const xmlChar*)"POINTS_AND_SURFACE")) setMode(RenderingMode::POINTS_AND_SURFACE);
		if(!xmlStrcmp(pmode, (const xmlChar*)"SURFACE")) setMode(RenderingMode::SURFACE);
		if(!xmlStrcmp(pmode, (const xmlChar*)"WIREFRAME_AND_SURFACE")) setMode(RenderingMode::WIREFRAME_AND_SURFACE);
		if(!xmlStrcmp(pmode, (const xmlChar*)"WIREFRAME_AND_POINTS")) setMode(RenderingMode::WIREFRAME_AND_POINTS);
		if(!xmlStrcmp(pmode, (const xmlChar*)"WIREFRAME")) setMode(RenderingMode::WIREFRAME);
		if(!xmlStrcmp(pmode, (const xmlChar*)"WIREFRAME_AND_SURFACE_AND_POINTS")) setMode(RenderingMode::WIREFRAME_AND_SURFACE_AND_POINTS);
	}
	else setMode(RenderingMode::NONE);

	
	//search the unknown attributes to fill the property fields map
	xmlAttr * attrs = node->properties;
	xmlNodePtr unknownAttrs = xmlNewNode(NULL, (xmlChar*)("unknownAttrs"));;
	while (attrs)
	{
		const xmlChar * pname = attrs->name;
		xmlChar * pval = attrs->children->content;
				
		if (pname && xmlStrcmp(pname, (xmlChar*)"name") && 
					 xmlStrcmp(pname, (xmlChar*)"index") &&
					 xmlStrcmp(pname, (xmlChar*)"mode")){
			xmlSetProp(unknownAttrs, pname, pval);
		}

		attrs = attrs->next;
	}

	//transform the unknown attributes to a property field map
	domToFields(unknownAttrs);
}


// inlines
inline 	double BasicSCProperties::getRed() const {
    return colorRGBA[0];
}
inline double BasicSCProperties::getGreen() const {
    return colorRGBA[1];
}
inline double BasicSCProperties::getBlue() const {
    return colorRGBA[2];
}
inline double BasicSCProperties::getAlpha() const {
    return colorRGBA[3];
}
inline double * BasicSCProperties::getRGB() const {
    return colorRGBA;
}
inline double * BasicSCProperties::getRGBA() const {
    return colorRGBA;
}
inline BasicSCProperties::Color BasicSCProperties::getColor() const {
    return color;
}
// TODO : a test for the color, if a composant is changed it might still be something we now
inline void BasicSCProperties::setRed(const double r) {
    colorRGBA[0]  = r;
    color = OTHER;
}
inline void BasicSCProperties::setGreen(const double g)  {
    colorRGBA[1] = g;
    color = OTHER;
}
inline void BasicSCProperties::setBlue(const double b)  {
    colorRGBA[2] = b;
    color = OTHER;
}
inline void BasicSCProperties::setAlpha(const double a)  {
    colorRGBA[3] = a;
    color = OTHER;
}
inline void BasicSCProperties::setRGB(const double r, const double g, const double b)  {
    setRed(r);
    setGreen(g);
    setBlue(b);
    color = OTHER;
}
inline void BasicSCProperties::setRGB(const double * rgb) {
    setRGB(rgb[0], rgb[1], rgb[2]);
    color = OTHER;
}
inline void BasicSCProperties::setRGBA(const double r, const double g, const double b, const double a)   {
    setRed(r);
    setGreen(g);
    setBlue(b);
    setAlpha(a);
    color = OTHER;
}
inline void BasicSCProperties::setRGBA(const double *rgba) {
    setRGBA(rgba[0], rgba[1], rgba[2], rgba[3]);
    color = OTHER;
}
inline void BasicSCProperties::setColor(Color c) {
    switch (c) {
    case RED:
        setRGBA(1.0, 0.0, 0.0, 1.0);
        break;
    case GREEN:
        setRGBA(0.0, 1.0, 0.0, 1.0);
        break;
    case BLUE:
        setRGBA(0.0, 0.0, 1.0, 1.0);
        break;
    default: // DEFAULT:or GRAY:or OTHER
        setRGBA(0.8, 0.8, 0.8, 1.0);
        break;
    }
    color = c;
}
inline void BasicSCProperties::alloc() {
    colorRGBA = new double[4];
}
inline void BasicSCProperties::setMode(const RenderingMode::Mode m) {
    mode.setMode(m);
}
inline RenderingMode::Mode BasicSCProperties::getMode() const {
    return mode.getMode();
}
inline std::string BasicSCProperties::getModeString() const {
    return mode.getModeString();
}
inline	bool BasicSCProperties::isVisible(const RenderingMode::Mode m) const {
    return mode.isVisible(m);
}
inline	void BasicSCProperties::setVisible(const RenderingMode::Mode m, const bool b) {
    this->mode.setVisible(m, b);
}

// write the default xml properties (beginning)
inline void BasicSCProperties::beginXML(std::ostream & o) {
    // print the name if there is one
    if (getName() != "")
        o<< " name=\"" << getName().c_str() << "\" ";

    // the mode property (if different than default)
    if (mode.getMode() != RenderingMode::NONE) {
        o << " mode=\"" << mode.getModeString() << "\" ";
    }
}

// write the default xml properties (end)
inline void BasicSCProperties::endXML(std::ostream &) {
    // nothing to be done
}


#endif
