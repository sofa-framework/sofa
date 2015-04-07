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
#define CGoGN_UTILS_DLL_EXPORT 1
#include "Utils/Qt/qtInputs.h"
#include <iostream>

#include<QLayout>
#include<QPushButton>
#include<QCheckBox>
#include<QSpinBox>
#include<QDoubleSpinBox>
#include<QSlider>
#include<QComboBox>


namespace CGoGN
{

namespace Utils
{

namespace QT
{

// class CGoGNDialog
// only used in this cpp so ...

class CGoGNDialog: public QDialog
{
protected:
	std::vector<QWidget*> m_wid;
public:

	CGoGNDialog(std::vector<const Var*>& params, const std::string& title);

	void getResults(std::vector<const Var*>& params);
};

CGoGNDialog::CGoGNDialog(std::vector<const Var*>& params, const std::string& title)
{
	int nbr=0;
	QVBoxLayout *layout = new QVBoxLayout(this);
	QGridLayout *m_layout = new QGridLayout();
	m_layout->setColumnStretch(0,1);
	m_layout->setColumnStretch(1,2);
	layout->addLayout(m_layout);

	// ADD ALL CREATED WIDGETS
	for (std::vector<const Var*>::iterator it = params.begin(); it != params.end(); ++it)
	{
		m_layout->addWidget((*it)->label(),nbr,0);
		m_wid.push_back((*it)->createInput());
		m_layout->addWidget(m_wid.back(),nbr++,1);

	}

	// ADD OK / CANCEL
	QHBoxLayout *layButtons = new QHBoxLayout();
	QPushButton* okbutton = new QPushButton( "OK", this );
	QObject::connect( okbutton, SIGNAL( clicked() ), this, SLOT( accept() ) );
	QPushButton* cancelbutton = new QPushButton( "Cancel", this );
	QObject::connect( cancelbutton, SIGNAL( clicked() ), 	this, SLOT( reject() ) );
	// TO LAYOUT
	layButtons->addWidget(okbutton);
	layButtons->addWidget(cancelbutton);
	layout->addLayout(layButtons);
	setWindowTitle(QString(title.c_str()));
}

void CGoGNDialog::getResults(std::vector<const Var*>& params)
{
	unsigned int nb = (unsigned int)(params.size());
	for (unsigned int i = 0; i < nb; ++i)
		params[i]->updateFrom(m_wid[i]);
}


// class Var

Var::Var(): m_next(NULL)
{}

Var::Var(const Var& v): m_next(&v)
{}

const Var* Var::next() const
{
	return m_next;
}

QLabel* Var::label() const
{
	return new QLabel(QString(m_label.c_str()));
}


// class VarBool

VarBool::VarBool(bool& val, const std::string& label) :
	m_val(val)
{
	m_label = label;
}

VarBool::VarBool(bool& val, const std::string& label, const Var& var) :
	Var(var), m_val(val)
{
	m_label = label;
}

QWidget* VarBool::createInput() const
{
	QCheckBox* check = new QCheckBox();
	check->setChecked(m_val);
	return check;
}

void VarBool::updateFrom( QWidget* widg) const
{
	QCheckBox* check = dynamic_cast<QCheckBox*>(widg);
	m_val = (check->checkState() == Qt::Checked);
}


//class VarInt

VarInteger::VarInteger(int min, int max, int& val, const std::string& label):
		m_min(min),m_max(max), m_val(val)
{
	m_label = label;
}


VarInteger::VarInteger(int min, int max, int& val, const std::string& label, const Var& var):
		Var(var), m_min(min),m_max(max), m_val(val)
{
	m_label = label;
}

QWidget* VarInteger::createInput() const
{
	QSpinBox *spin = new QSpinBox();
	spin->setRange(m_min, m_max);
	spin->setValue(m_val);
	return spin;
}

void VarInteger::updateFrom( QWidget* widg) const
{
	QSpinBox* spin = dynamic_cast<QSpinBox*>(widg);
	m_val = spin->value();
}


// class VarDbl

VarDbl::VarDbl(double min, double max, double& val, const std::string& label) :
	m_min(min), m_max(max), m_val(val)
{
	m_label = label;
}

VarDbl::VarDbl(double min, double max, double& val, const std::string& label, const Var& var) :
	Var(var), m_min(min), m_max(max), m_val(val)
{
	m_label = label;
}

QWidget* VarDbl::createInput() const
{
	QDoubleSpinBox *spin = new QDoubleSpinBox();
	spin->setRange(m_min, m_max);
	spin->setValue(m_val);
	return spin;
}

void VarDbl::updateFrom( QWidget* widg) const
{
	QDoubleSpinBox* spin = dynamic_cast<QDoubleSpinBox*>(widg);
	m_val = spin->value();
}



// class VarFloat

VarFloat::VarFloat(float min, float max, float& val, const std::string& label) :
	m_min(min), m_max(max), m_val(val)
{
	m_label = label;
}

VarFloat::VarFloat(float min, float max, float& val, const std::string& label, const Var& var) :
	Var(var), m_min(min), m_max(max), m_val(val)
{
	m_label = label;
}

QWidget* VarFloat::createInput() const
{
	QDoubleSpinBox *spin = new QDoubleSpinBox();
	spin->setRange(double(m_min), double(m_max));
	spin->setValue(double(m_val));
	return spin;
}

void VarFloat::updateFrom( QWidget* widg) const
{
	QDoubleSpinBox* spin = dynamic_cast<QDoubleSpinBox*>(widg);
	m_val = float(spin->value());
}


// class VarSlider

VarSlider::VarSlider(int min, int max, int& val, const std::string& label):
		m_min(min),m_max(max), m_val(val)
{
	m_label= label;
}

VarSlider::VarSlider(int min, int max, int& val, const std::string& label, const Var& var):
		Var(var),m_min(min),m_max(max), m_val(val)
{
	m_label= label;
}

QWidget* VarSlider::createInput() const
{
	QSlider *slider = new QSlider(Qt::Horizontal);
	slider->setRange(m_min, m_max);
	slider->setSliderPosition(m_val);
	return slider;
}

void VarSlider::updateFrom(QWidget* widg) const
{
	QSlider* slider = dynamic_cast<QSlider*>(widg);
	m_val = slider->value();
}


// class VarCombo

VarCombo::VarCombo(const std::string& choices, int& v, const std::string& label):
		m_choices(choices),m_val(v)
{
	m_label= label;
}

VarCombo::VarCombo(const std::string& choices, int& v, const std::string& label, const Var& var):
		Var(var), m_choices(choices),m_val(v)
{
	m_label= label;
}

QWidget* VarCombo::createInput() const
{
	QComboBox *combo = new QComboBox();

	size_t pos = 0;
	while (pos != std::string::npos)
	{
		size_t pos2 = m_choices.find(';', pos);
		if (pos2!= std::string::npos)
		{
			std::string choice = m_choices.substr(pos, pos2 - pos);
			combo->addItem(QString(choice.c_str()));
			pos = pos2 + 1;
		}
		else
		{
			std::string choice = m_choices.substr(pos, pos2);
			combo->addItem(QString(choice.c_str()));
			pos = pos2;
		}

	}

	combo->setCurrentIndex(m_val);

	return combo;
}

void VarCombo::updateFrom( QWidget* widg) const
{
	QComboBox* combo = dynamic_cast<QComboBox*>(widg);
	m_val = combo->currentIndex();
}

bool inputValues(const Var& v1, const std::string& title)
{
	std::vector<const Var*> params;
	const Var* ptr = &v1;
	while (ptr != NULL)
	{
		params.push_back(ptr);
		ptr = ptr->next();
	}

	CGoGNDialog dialog(params, title);
	int ret = dialog.exec();
	if (ret == QDialog::Accepted)
	{
		dialog.getResults(params);
		return true;
	}
	return false;
}

} // namespace QT

} // namespace Utils

} // namespace CGoGN
