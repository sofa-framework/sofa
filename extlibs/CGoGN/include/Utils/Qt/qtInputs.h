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

#ifndef QTINPUTS_H_
#define QTINPUTS_H_

//#include <QApplication>
#include <QDesktopWidget>
//#include <QMainWindow>
#include <QWidget>
//#include <QtGui>
#include <QDialog>
#include <QLabel>

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

namespace QT
{

/**
 * Abstract class for Variable input in dialog window
 */
class Var
{
protected:
	const Var* m_next;
	std::string m_label;

public:
	Var(const Var& v);
	Var();

	const Var* next() const;
	QLabel* label() const;

	virtual QWidget* createInput() const = 0;
	virtual void updateFrom( QWidget* widg) const = 0;
};

/**
 * Class for boolean input in dialog window
 * Use: VarBool(ref_to_val, "label" [,nextVar])
 */
class CGoGN_UTILS_API VarBool : public Var
{
public:
	bool& m_val;

public:
	VarBool(bool& val, const std::string& label);
	VarBool(bool& val, const std::string& label, const Var& var);
	QWidget* createInput() const;
	void updateFrom(QWidget* widg) const;
};

/**
 * Class for integer input in dialog window (with spinbox)
 * Use: VarBool(min,max,ref_to_val, "label" [,nextVar])
 */
class CGoGN_UTILS_API VarInteger : public Var
{
public:
	int m_min;
	int m_max;
	int& m_val;

public:
	VarInteger(int min, int max, int& val, const std::string& label);
	VarInteger(int min, int max, int& val, const std::string& label, const Var& var);
	QWidget* createInput() const;
	void updateFrom(QWidget* widg) const;
};

/**
 * Class for double input in dialog window (with spinbox)
 * Use: VarBool(min,max,ref_to_val, "label" [,nextVar])
 */
class CGoGN_UTILS_API VarDbl : public Var
{
public:
	double m_min;
	double m_max;
	double& m_val;

public:
	VarDbl(double min, double max, double& val, const std::string& label);
	VarDbl(double min, double max, double& val, const std::string& label, const Var& var);
	QWidget* createInput() const;
	void updateFrom(QWidget* widg) const;
};


/**
 * Class for double input in dialog window (with spinbox)
 * Use: VarBool(min,max,ref_to_val, "label" [,nextVar])
 */
class CGoGN_UTILS_API VarFloat : public Var
{
public:
	float m_min;
	float m_max;
	float& m_val;

public:
	VarFloat(float min, float max, float& val, const std::string& label);
	VarFloat(float min, float max, float& val, const std::string& label, const Var& var);
	QWidget* createInput() const;
	void updateFrom(QWidget* widg) const;
};


/**
 * Class for integer input in dialog window (with slider)
 * Use: VarBool(min,max,ref_to_val, "label" [,nextVar])
 */
class CGoGN_UTILS_API VarSlider : public Var
{
public:
	int m_min;
	int m_max;
	int& m_val;

public:
	VarSlider(int min, int max, int& val, const std::string& label);
	VarSlider(int min, int max, int& val, const std::string& label, const Var& var);
	QWidget* createInput() const;
	void updateFrom(QWidget* widg) const;
};

/**
 * Class for combo input (interger in dialog window (with spinbox)
 * Use: VarBool("list_of_item, ref_to_val, "label" [,nextVar])
 *     item are separated by ; in list_of_item
 */
class CGoGN_UTILS_API VarCombo : public Var
{
public:
	const std::string& m_choices;
	int& m_val;

public:
	VarCombo(const std::string& choices, int& v, const std::string& label);
	VarCombo(const std::string& choices, int& v, const std::string& label, const Var& var);
	QWidget* createInput() const;
	void updateFrom(QWidget* widg) const;
};

/**
 * Open a QtDialog for inputs of all chain Var defined
 * &param v1 a Var object (that chain other Var objects
 * Example:
 * {using namespace Utils::QT
 *  inputValues(VarInt(0,20,x, "Entier",
 *				VarBool(b, "Bool",
 * 				VarDbl(0.314,3.14,d,"Double",
 *				VarSlider(10,100,s,"Slider",
 *				VarCombo("Riri;Fifi;Loulou;Donald",c,"Combo"))))) );
 * } // limit scope of using namespace
 */
CGoGN_UTILS_API bool inputValues(const Var& v1, const std::string& title = "input data");


} // namespace QT

} // namespace Utils

} // namespace CGoGN

#endif
