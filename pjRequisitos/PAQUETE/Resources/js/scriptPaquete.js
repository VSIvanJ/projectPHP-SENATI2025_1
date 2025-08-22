function ConsultarPaquete() {
    nombreC=$('#txtConPaq').val();
    if(nombreC===''){
        $('#selectPaquete').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procConsultaPaquete.php',
        type:'POST',
        data:{txtConPaq:nombreC},
        success: function (respuesta) {
            $('#selectPaquete').html(respuesta);
        },
        error: function () {
            $('#selectPaquete').text('ERROR AL INTENTAR CONSULTAR');
        }
    });
}

function ActualizarPaquete() {
    IDA=$('#txtActId').val();
    DesA=$('#txtActDes').val();
    PreA=$('#txtActPre').val();
    if(nombreC==='' || DesA==='' || PreA===''){
        $('#updatePaquete').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procActualizaPaquete.php',
        type:'POST',
        data:{txtActId:IDA,txtActDes:DesA,txtActPre:PreA},
        success: function (respuesta2) {
            $('#updatePaquete').html(respuesta2);
        },
        error: function () {
            $('#updatePaquete').text('ERROR AL INTENTAR ACTUALIZAR');
        }
    });
}

function EliminarPaquete() {
    IDE=$('#txtBID').val();
    if(IDE===''){
        $('#deletePaquete').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procEliminaPaquete.php',
        type:'POST',
        data:{txtBID:IDE},
        success: function (respuesta3) {
            $('#deletePaquete').html(respuesta3);
        },
        error: function () {
            $('#deletePaquete').text('ERROR AL INTENTAR ELIMINAR');
        }
    });
}

function LimpiarConsulta() {
    $('#txtConPaq').val('');
    $('#selectPaquete').text('');
}

function LimpiarActualizar() {
    $('#txtActId').val('');
    $('#txtActDes').val('');
    $('#txtActPre').val('');
    $('#updatePaquete').text('');
}

function LimpiarEliminar() {
    $('#txtBID').val('');
    $('#deletePaquete').text('');
}