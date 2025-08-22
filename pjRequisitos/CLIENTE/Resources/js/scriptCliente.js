function ConsultarCliente() {
    conRC=$('#txtCRC').val();
    if(conRC===''){
        $('#selectCliente').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procConsultaCliente.php',
        type:'POST',
        data:{txtCRC:conRC},
        success: function (respuesta) {
            $('#selectCliente').html(respuesta);
        },
        error: function () {
            $('#selectCliente').text('ERROR AL INTENTAR CONSULTAR');
        }
    });
}

function ActualizarCliente() {
    ArA=$('#txtARC').val();
    AdA=$('#txtADC').val();
    if(ArA==='' || AdA===''){
        $('#updateCliente').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procActualizaCliente.php',
        type:'POST',
        data:{txtARC:ArA,txtADC:AdA},
        success: function (respuesta2) {
            $('#updateCliente').html(respuesta2);
        },
        error: function () {
            $('#updateCliente').text('ERROR AL INTENTAR ACTUALIZAR');
        }
    });
}

function EliminarCliente() {
    EiCl=$('#txtEIC').val();
    ERCl=$('#txtERC').val();
    if(EiCl==='' || ERCl===''){
        $('#deleteCliente').text('Ingrese el(los) dato(s) requeridos');
        return;
    };
    
    $.ajax({
        url:'procEliminaCliente.php',
        type:'POST',
        data:{txtEIC:EiCl,txtERC:ERCl},
        success: function (respuesta3) {
            $('#deleteCliente').html(respuesta3);
        },
        error: function () {
            $('#deleteCliente').text('ERROR AL INTENTAR ELIMINAR');
        }
    });
}

function LimpiarConsultaCli() {
    $('#txtCRC').val('');
    $('#selectCliente').text('');
}

function LimpiarActualizarCli() {
    $('#txtARC').val('');
    $('#txtADC').val('');
    $('#updateCliente').text('');
}

function LimpiarEliminarCli() {
    $('#txtEIC').val('');
    $('#txtERC').val('');
    $('#deleteCliente').text('');
}