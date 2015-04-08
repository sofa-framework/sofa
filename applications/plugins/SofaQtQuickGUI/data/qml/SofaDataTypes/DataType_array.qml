import QtQuick 2.0
import QtQuick.Controls 1.3
import QtQuick.Layouts 1.0

Loader {
    id: root

    property var dataObject

    sourceComponent: {
        if(dataObject.properties.static) {
            if((!dataObject.properties.innerStatic && dataObject.value.length <= 7) ||
               (dataObject.properties.innerStatic && 1 === dataObject.value.length && dataObject.properties.cols <= 7))
                return staticSmallArrayView;
            else if(dataObject.properties.innerStatic && dataObject.properties.cols <= 7)
                return staticInStaticTableView;
            else
                return staticArrayView;
        }
        else {
            if(dataObject.properties.innerStatic) {
                if(dataObject.properties.cols <= 7)
                    return staticInDynamicTableView;
                else
                    return dynamicArrayView;
            }
        }

        return dynamicArrayView;
    }

    Component {
        id: staticInStaticTableView
        TableView {
            id: tableView

            Component {
                id: columnComponent
                TableViewColumn {
                    movable: false
                    resizable: false
                    horizontalAlignment: Text.AlignHCenter
                    width: (tableView.width - 14) / tableView.columnCount - 1
                }
            }

            Component.onCompleted: {
                for(var i = 0; i < dataObject.properties.cols; ++i)
                    addColumn(columnComponent.createObject(tableView, {"title": i.toString(), "role": "c" + i.toString()}));
            }

            Connections {
                target: dataObject
                onValueChanged: {
                    listModel.update();

                    dataObject.modified = false;
                }
            }

            model: ListModel {
                id: listModel

                Component.onCompleted: populate();

                property int previousCount: 0

                function populate() {
                    var newCount = dataObject.value.length;
                    if(previousCount < newCount)
                        for(var j = previousCount; j < newCount; ++j) {
                            var values = {};
                            for(var i = previousCount; i < dataObject.properties.cols; ++i)
                                values["c" + i.toString()] = dataObject.value[j][i];

                            append(values);
                        }
                    else if(previousCount > newCount)
                        remove(newCount, previousCount - newCount);

                    previousCount = count;
                }

                function update() {
                    if(count !== dataObject.value.length)
                        populate();

                    for(var j = 0; j < count; ++j) {
                        var values = {};
                        for(var i = previousCount; i < dataObject.properties.cols; ++i)
                            values["c" + i.toString()] = dataObject.value[j][i];

                            set(j, values);
                    }
                }
            }

            function populate() {
                listModel.populate();
            }

            function update() {
                listModel.update();
            }

            itemDelegate: TextInput {
                anchors.fill: parent
                anchors.leftMargin: 6
                anchors.rightMargin: 6
                clip: true
                readOnly: -1 === styleData.row || dataObject.readOnly
                color: styleData.textColor
                horizontalAlignment: TextEdit.AlignHCenter
                inputMethodHints: Qt.ImhFormattedNumbersOnly
                text: {
                    if(-1 !== styleData.row) {
                        var value = dataObject.value[styleData.row][styleData.column];
                        if("string" === typeof(value))
                            return value;
                        else
                            return value.toFixed(3);
                    }

                    return "";
                }
                property int previousRow: -1
                onTextChanged: {
                    if(-1 === styleData.row || dataObject.readOnly)
                        return;

                    if(previousRow !== styleData.row) {
                        previousRow = styleData.row;
                        return;
                    }

                    var oldValue = dataObject.value[styleData.row][styleData.column];
                    if("string" !== typeof(oldValue))
                        oldValue = oldValue.toFixed(3);

                    var value = text;
                    if(value !== oldValue) {
                        dataObject.value[styleData.row][styleData.column] = value;
                        dataObject.modified = true;
                    }
                }
            }
        }
    }

    Component {
        id: staticSmallArrayView
        RowLayout {
            id: rowLayout
            width: parent.width
            spacing: 0
            enabled: !dataObject.readOnly

            property var fields: []
            property bool innerArray: false
            property bool useSpinBox: false

            Component.onCompleted: populate();

            function populate() {
                var values = dataObject.value;
                if(1 === values.length && Array.isArray(values[0]))
                {
                    values = dataObject.value[0];
                    innerArray = true;
                }

                useSpinBox = false;
                if(values.length <= 4 && "string" !== typeof(values[0])) // TODO: WARNING : could be mixed types array (string, number, etc.)
                    useSpinBox = true;

                fields = [];
                for(var i = 0; i < values.length; ++i)
                    if(useSpinBox)
                        fields[i] = spinBoxComponent.createObject(rowLayout, {index: i});
                    else
                        fields[i] = textFieldComponent.createObject(rowLayout, {index: i});

                update();
            }

            function update() {
                var values = dataObject.value;
                if(innerArray)
                    values = dataObject.value[0];

                for(var i = 0; i < values.length; ++i) {
                    if(useSpinBox)
                        fields[i].value = values[i];
                    else
                        fields[i].text = values[i];
                }
            }

            Component {
                id: textFieldComponent

                TextField {
                    Layout.fillWidth: true
                    readOnly: dataObject.readOnly
                    enabled: !dataObject.readOnly

                    property int index
                    onTextChanged: {
                        if(rowLayout.innerArray)
                            dataObject.value[0][index] = text;
                        else
                            dataObject.value[index] = text;

                        dataObject.modified = true;
                    }
                }
            }

            Component {
                id: spinBoxComponent

                SpinBox {
                    Layout.fillWidth: true
                    enabled: !dataObject.readOnly

                    property int index
                    onValueChanged: {
                        if(rowLayout.innerArray)
                            dataObject.value[0][index] = value;
                        else
                            dataObject.value[index] = value;

                        dataObject.modified = true;
                    }

                    decimals: 3
                    minimumValue: -Number.MAX_VALUE
                    maximumValue:  Number.MAX_VALUE
                }
            }

            Connections {
                target: dataObject
                onValueChanged: rowLayout.update();
            }
        }
    }

    Component {
        id: staticArrayView
        TextField {
            id: textField
            readOnly: dataObject.readOnly
            enabled: !dataObject.readOnly
            text: undefined !== dataObject.value ? dataObject.value.toString() : ""

            Binding {
                target: dataObject
                property: "value"
                value: textField.text
            }
        }
    }

    Component {
        id: staticInDynamicTableView

        ColumnLayout {
            spacing: 0

            RowLayout {
                Layout.fillWidth: true

                Text {
                    text: "Size"
                }
                SpinBox {
                    id: rowNumber
                    enabled: !dataObject.readOnly && showEditButton.checked
                    Layout.fillWidth: true
                    value: dataObject.value.length
                    onEditingFinished: {
                        if(value === dataObject.value.length)
                            return;

                        var oldLength = dataObject.value.length;
                        dataObject.value.length = value;
                        for(var j = oldLength; j < dataObject.value.length; ++j) {
                            dataObject.value[j] = [];
                            for(var i = 0; i < dataObject.properties.cols; ++i)
                                dataObject.value[j][i] = 0;
                        }

                        dataObject.modified = true;

                        if(loader.item)
                            loader.item.populate();
                    }

                    minimumValue: 0
                    maximumValue: Number.MAX_VALUE
                }
                Button {
                    id: showEditButton
                    text: "Show / Edit"
                    checkable: true
                }
            }

            Loader {
                id: loader
                Layout.fillWidth: true
                Layout.fillHeight: true
                visible: showEditButton.checked
                active: visible
                sourceComponent: staticInStaticTableView
            }
        }
    }

    Component {
        id: dynamicArrayView

        TextField {
            id: textField
            readOnly: dataObject.readOnly
            enabled: !dataObject.readOnly

            onTextChanged: {
                if(!dataObject.readOnly)
                    if(Array.isArray(dataObject.value))
                        dataObject.value = text.split(' ')
                    else
                        dataObject.value = text
            }

            Binding {
                target: textField
                property: "text"
                value: Array.isArray(dataObject.value) ? dataObject.value.join(' ') : dataObject.value
            }
        }
    }
}
