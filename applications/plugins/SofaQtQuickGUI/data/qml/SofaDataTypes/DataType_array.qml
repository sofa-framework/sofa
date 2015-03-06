import QtQuick 2.0
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.0

Loader {
    id: root

    property var dataObject

    sourceComponent: {
        if(dataObject.properties.static) {
            if(1 === value.length) {
                if(dataObject.properties.innerStatic && dataObject.properties.cols > 7)
                    return staticArrayView;
                else
                    return staticSmallArrayView;
            }
            else if(dataObject.value.length <= 7) {
                if(dataObject.properties.innerStatic)
                    return staticInStaticTableView;
                else
                    return staticSmallArrayView;
            }
            else
                return staticArrayView;
        }
        else {
            if(dataObject.properties.innerStatic) {
                if(dataObject.value.length <= 7)
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
            enabled: !dataObject.readOnly

            Component {
                id: columnComponent
                TableViewColumn {
                    movable: false
                    resizable: false
                    horizontalAlignment: Text.AlignHCenter
                    width: tableView.width / tableView.columnCount - 1
                }
            }

            Component.onCompleted: {
                for(var i = 0; i < dataObject.properties.cols; ++i)
                    addColumn(columnComponent.createObject(tableView, {"title": i.toString(), "role": "c" + i.toString()}));
            }

            property real rowHeight: 18
            implicitHeight: Math.max(2, Math.min(rowCount, 6)) * rowHeight

            model: ListModel {
                Component.onCompleted: populate();

                function populate() {
                    clear();

                    for(var j = 0; j < dataObject.value.length; ++j) {
                        var values = {};
                        for(var i = 0; i < dataObject.value[j].length; ++i)
                            values["c" + i.toString()] = dataObject.value[j][i];

                        append(values);
                    }
                }
            }

            itemDelegate: TextEdit {
                //anchors.centerIn: parent.Center
                readOnly: dataObject.readOnly
                color: styleData.textColor
                horizontalAlignment: TextEdit.AlignHCenter
                //elide: styleData.elideMode
                text: styleData.value
                onTextChanged: dataObject.editedValue = text
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

                //TODO: choose between textField and spinBox
                useSpinBox = false;
                if(values.length <= 4)
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
                        fields[i].value = Number(values[i]);
                    else
                        fields[i].text = Number(values[i]);
                }
            }

            property real rowHeight: 24

            Component {
                id: textFieldComponent

                TextField {
                    Layout.fillWidth: true
                    validator: DoubleValidator {}
                    readOnly: dataObject.readOnly
                    enabled: !dataObject.readOnly

                    property int index
                    onTextChanged: {
                        if(rowLayout.innerArray)
                            dataObject.editedValue[0][index] = Number(text);
                        else
                            dataObject.editedValue[index] = Number(text);

                        dataObject.modified = true;
                    }
                    //Layout.preferredHeight: rowHeight
/*
                    decimals: 3
                    minimumValue: undefined !== dataObject.properties.min ? dataObject.properties.min : -Number.MAX_VALUE
                    maximumValue: undefined !== dataObject.properties.max ? dataObject.properties.max :  Number.MAX_VALUE
                    stepSize: undefined !== dataObject.properties.step ? dataObject.properties.step : 1
*/
                }
            }

            Component {
                id: spinBoxComponent

                SpinBox {
                    Layout.fillWidth: true
                    //Layout.preferredHeight: rowHeight
                    enabled: !dataObject.readOnly

                    property int index
                    onValueChanged: {
                        if(rowLayout.innerArray)
                            dataObject.editedValue[0][index] = value;
                        else
                            dataObject.editedValue[index] = value;

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

            //"Static in Dynamic Array: " + dataObject.value
        }
    }

    Component {
        id: staticArrayView
        TextField {
            readOnly: dataObject.readOnly
            enabled: !dataObject.readOnly
            text: undefined !== dataObject.value ? dataObject.value.toString() : ""
            onTextChanged: dataObject.editedValue = text
        }
    }

    Component {
        id: staticInDynamicTableView

        Column {
            SpinBox {
                id: rowNumber
                value: dataObject.value.length
                onValueChanged: dataObject.editedValue = value
                enabled: !dataObject.readOnly
            }
            TableView {
                enabled: !dataObject.readOnly

                TableViewColumn {
                    id: titleColumn
                    title: "Value"
                    role: "value"
                    movable: false
                    resizable: false
                    //width: tableView.viewport.width - authorColumn.width
                }

                model: ListModel {
                    ListElement {value: "staticInDynamicTableView"}
                    /*Component.onCompleted: populate();

                    function populate() {
                        clear();

                        for(var i = 0; i < dataObject.value.length; ++i)
                            append({"name": dataObject.value[i]});
                    }*/
                }

                itemDelegate: Item {
                    TextField {
                        readOnly: dataObject.readOnly
                        anchors.verticalCenter: parent.verticalCenter
                        //color: styleData.textColor
                        //elide: styleData.elideMode
                        text: styleData.value
                        onTextChanged: dataObject.editedValue = text
                    }
                }

                //rowCount: rowNumber.value
                //columnCount: dataObject.properties.cols
            }
        }
    }

    Component {
        id: dynamicArrayView

        TextField {
            readOnly: dataObject.readOnly
            enabled: !dataObject.readOnly
            text: undefined !== dataObject.value ? dataObject.value.toString() : ""
            onTextChanged: dataObject.editedValue = text
        }
    }
}

