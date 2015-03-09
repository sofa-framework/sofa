import QtQuick 2.0
import QtQuick.Controls 1.2

TextField {
    id: root

    property var dataObject

    readOnly: dataObject.readOnly
    enabled: !dataObject.readOnly
    text: undefined !== dataObject.value ? dataObject.value.toString() : ""
    onTextChanged: dataObject.editedValue = text
}
