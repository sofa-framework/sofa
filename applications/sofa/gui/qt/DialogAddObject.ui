<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>DialogAddObject</class>
 <widget class="QDialog" name="DialogAddObject">
  <property name="geometry">
   <rect>
    <x>0</x>
    <y>0</y>
    <width>556</width>
    <height>264</height>
   </rect>
  </property>
  <property name="sizePolicy">
   <sizepolicy hsizetype="Ignored" vsizetype="Preferred">
    <horstretch>0</horstretch>
    <verstretch>0</verstretch>
   </sizepolicy>
  </property>
  <property name="windowTitle">
   <string>Add a scene or an object</string>
  </property>
  <property name="sizeGripEnabled">
   <bool>true</bool>
  </property>
  <layout class="QGridLayout">
   <item row="4" column="0">
    <layout class="QHBoxLayout">
     <item>
      <widget class="QLabel" name="scaleText">
       <property name="text">
        <string>Scale</string>
       </property>
       <property name="wordWrap">
        <bool>false</bool>
       </property>
      </widget>
     </item>
     <item>
      <widget class="QLineEdit" name="scaleValue"/>
     </item>
    </layout>
   </item>
   <item row="1" column="0">
    <layout class="QHBoxLayout">
     <item>
      <widget class="QLabel" name="openFileText">
       <property name="text">
        <string>Open File</string>
       </property>
       <property name="wordWrap">
        <bool>false</bool>
       </property>
      </widget>
     </item>
     <item>
      <widget class="QLineEdit" name="openFilePath"/>
     </item>
     <item>
      <widget class="QPushButton" name="openFileButton">
       <property name="text">
        <string/>
       </property>
      </widget>
     </item>
    </layout>
   </item>
   <item row="5" column="0">
    <layout class="QHBoxLayout">
     <item>
      <spacer name="Horizontal Spacing2">
       <property name="orientation">
        <enum>Qt::Horizontal</enum>
       </property>
       <property name="sizeType">
        <enum>QSizePolicy::Expanding</enum>
       </property>
       <property name="sizeHint" stdset="0">
        <size>
         <width>20</width>
         <height>20</height>
        </size>
       </property>
      </spacer>
     </item>
     <item>
      <widget class="QPushButton" name="buttonOk">
       <property name="text">
        <string>&amp;OK</string>
       </property>
       <property name="shortcut">
        <string/>
       </property>
       <property name="autoDefault">
        <bool>true</bool>
       </property>
       <property name="default">
        <bool>true</bool>
       </property>
      </widget>
     </item>
     <item>
      <widget class="QPushButton" name="buttonCancel">
       <property name="text">
        <string>&amp;Cancel</string>
       </property>
       <property name="shortcut">
        <string/>
       </property>
       <property name="autoDefault">
        <bool>true</bool>
       </property>
      </widget>
     </item>
    </layout>
   </item>
   <item row="2" column="0">
    <layout class="QHBoxLayout">
     <item>
      <widget class="QLabel" name="positionText">
       <property name="text">
        <string>Initial Position</string>
       </property>
       <property name="wordWrap">
        <bool>false</bool>
       </property>
      </widget>
     </item>
     <item>
      <widget class="QLineEdit" name="positionX"/>
     </item>
     <item>
      <widget class="QLineEdit" name="positionY"/>
     </item>
     <item>
      <widget class="QLineEdit" name="positionZ"/>
     </item>
    </layout>
   </item>
   <item row="3" column="0">
    <layout class="QHBoxLayout">
     <item>
      <widget class="QLabel" name="rotationText">
       <property name="text">
        <string>Initial Rotation</string>
       </property>
       <property name="wordWrap">
        <bool>false</bool>
       </property>
      </widget>
     </item>
     <item>
      <widget class="QLineEdit" name="rotationX"/>
     </item>
     <item>
      <widget class="QLineEdit" name="rotationY"/>
     </item>
     <item>
      <widget class="QLineEdit" name="rotationZ"/>
     </item>
    </layout>
   </item>
   <item row="0" column="0">
    <widget class="QGroupBox" name="buttonGroup">
     <property name="title">
      <string>Objects</string>
     </property>
     <layout class="QGridLayout">
      <item row="0" column="0">
       <widget class="QRadioButton" name="custom">
        <property name="text">
         <string>custom</string>
        </property>
       </widget>
      </item>
     </layout>
    </widget>
   </item>
  </layout>
 </widget>
 <layoutdefault spacing="6" margin="11"/>
 <resources/>
 <connections>
  <connection>
   <sender>openFileButton</sender>
   <signal>clicked()</signal>
   <receiver>DialogAddObject</receiver>
   <slot>fileOpen()</slot>
   <hints>
    <hint type="sourcelabel">
     <x>20</x>
     <y>20</y>
    </hint>
    <hint type="destinationlabel">
     <x>20</x>
     <y>20</y>
    </hint>
   </hints>
  </connection>
  <connection>
   <sender>buttonOk</sender>
   <signal>clicked()</signal>
   <receiver>DialogAddObject</receiver>
   <slot>accept()</slot>
   <hints>
    <hint type="sourcelabel">
     <x>20</x>
     <y>20</y>
    </hint>
    <hint type="destinationlabel">
     <x>20</x>
     <y>20</y>
    </hint>
   </hints>
  </connection>
  <connection>
   <sender>buttonCancel</sender>
   <signal>clicked()</signal>
   <receiver>DialogAddObject</receiver>
   <slot>reject()</slot>
   <hints>
    <hint type="sourcelabel">
     <x>20</x>
     <y>20</y>
    </hint>
    <hint type="destinationlabel">
     <x>20</x>
     <y>20</y>
    </hint>
   </hints>
  </connection>
 </connections>
</ui>
