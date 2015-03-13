import QtQuick 2.0
import QtQuick.Layouts 1.1
import QtQuick.Controls 1.3

ColumnLayout {
    id: root
    spacing: 0

    property var dataObject

    property var values: undefined !== dataObject.value ? dataObject.value.split(' ') : []

    RowLayout {
        Layout.fillWidth: true
        spacing: 0

        Text {
            id: minLabel
            Layout.preferredWidth: Math.max(minLabel.implicitWidth, maxLabel.implicitWidth)
            text: "Min"
        }

        TextField {
            id: textField0
            property int index: 0

            Layout.fillWidth: true;

            //validator: DoubleValidator {decimals: 3}
            readOnly: dataObject.readOnly
            enabled: !dataObject.readOnly

            onTextChanged: {
                if(!dataObject.readOnly) {
                    root.values[index] = text;
                    dataObject.value = root.values.join(' ');
                }
            }

            Binding {
                target: textField0
                property: "text"
                value: root.values[textField0.index]
                when: !dataObject.readOnly
            }
        }

        TextField {
            id: textField1
            property int index: 1

            Layout.fillWidth: true;

            //validator: DoubleValidator {decimals: 3}
            readOnly: dataObject.readOnly
            enabled: !dataObject.readOnly

            onTextChanged: {
                if(!dataObject.readOnly) {
                    root.values[index] = text;
                    dataObject.value = root.values.join(' ');
                }
            }

            Binding {
                target: textField1
                property: "text"
                value: root.values[textField1.index]
                when: !dataObject.readOnly
            }
        }

        TextField {
            id: textField2
            property int index: 2

            Layout.fillWidth: true;

            //validator: DoubleValidator {decimals: 3}
            readOnly: dataObject.readOnly
            enabled: !dataObject.readOnly

            onTextChanged: {
                if(!dataObject.readOnly) {
                    root.values[index] = text;
                    dataObject.value = root.values.join(' ');
                }
            }

            Binding {
                target: textField2
                property: "text"
                value: root.values[textField2.index]
                when: !dataObject.readOnly
            }
        }
    }

    RowLayout {
        Layout.fillWidth: true
        spacing: 0

        Text {
            id: maxLabel
            text: "Max"
            Layout.preferredWidth: Math.max(minLabel.implicitWidth, maxLabel.implicitWidth)
        }

        TextField {
            id: textField3
            property int index: 3

            Layout.fillWidth: true;

            //validator: DoubleValidator {decimals: 3}
            readOnly: dataObject.readOnly
            enabled: !dataObject.readOnly

            onTextChanged: {
                if(!dataObject.readOnly) {
                    root.values[index] = text;
                    dataObject.value = root.values.join(' ');
                }
            }

            Binding {
                target: textField3
                property: "text"
                value: root.values[textField3.index]
                when: !dataObject.readOnly
            }
        }

        TextField {
            id: textField4
            property int index: 4

            Layout.fillWidth: true;

            //validator: DoubleValidator {decimals: 3}
            readOnly: dataObject.readOnly
            enabled: !dataObject.readOnly

            onTextChanged: {
                if(!dataObject.readOnly) {
                    root.values[index] = text;
                    dataObject.value = root.values.join(' ');
                }
            }

            Binding {
                target: textField4
                property: "text"
                value: root.values[textField4.index]
                when: !dataObject.readOnly
            }
        }

        TextField {
            id: textField5
            property int index: 5

            Layout.fillWidth: true;

            //validator: DoubleValidator {decimals: 3}
            readOnly: dataObject.readOnly
            enabled: !dataObject.readOnly

            onTextChanged: {
                if(!dataObject.readOnly) {
                    root.values[index] = text;
                    dataObject.value = root.values.join(' ');
                }
            }

            Binding {
                target: textField5
                property: "text"
                value: root.values[textField5.index]
                when: !dataObject.readOnly
            }
        }
    }
}
