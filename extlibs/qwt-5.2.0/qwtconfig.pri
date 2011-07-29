######################################################################
# qmake internal options
######################################################################

CONFIG           += qt     # Also for Qtopia Core!

# THE REST OF THE CONFIG OPTIONS ARE HANDLED BY SOFA

######################################################################
# If you want to have different names for the debug and release 
# versions you can add a suffix rule below.
######################################################################

DEBUG_SUFFIX        = 
RELEASE_SUFFIX      = 

win32 {
    DEBUG_SUFFIX      = d
}

######################################################################
# Build the static/shared libraries.
# If QwtDll is enabled, a shared library is built, otherwise
# it will be a static library.
######################################################################

!contains(CONFIGSTATIC, static) {
	CONFIG           += QwtDll
}

######################################################################
# QwtPlot enables all classes, that are needed to use the QwtPlot 
# widget. 
######################################################################

CONFIG       += QwtPlot

######################################################################
# QwtWidgets enables all classes, that are needed to use the all other
# widgets (sliders, dials, ...), beside QwtPlot. 
######################################################################

CONFIG     += QwtWidgets

######################################################################
# If you want to display svg imageson the plot canvas, enable the 
# line below. Note that Qwt needs the svg+xml, when enabling 
# QwtSVGItem.
######################################################################

#CONFIG     += QwtSVGItem

######################################################################
# If you have a commercial license you can use the MathML renderer
# of the Qt solutions package to enable MathML support in Qwt.
# So if you want this, copy qtmmlwidget.h + qtmmlwidget.cpp to
# textengines/mathml and enable the line below.
######################################################################

#CONFIG     += QwtMathML

######################################################################
# If you want to build the Qwt designer plugin, 
# enable the line below.
# Otherwise you have to build it from the designer directory.
######################################################################

#CONFIG     += QwtDesigner

######################################################################
# If you want to auto build the examples, enable the line below
# Otherwise you have to build them from the examples directory.
######################################################################

#CONFIG     += QwtExamples
