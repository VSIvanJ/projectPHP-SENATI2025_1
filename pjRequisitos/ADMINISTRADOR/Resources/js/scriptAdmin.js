function ActualizarEmail() {
    nuevoEmail=$('#txtEmailAdmin').val();
    contra=$('#txtClaveAdmin').val();
    dni=$('#txtDNIAdmin').val();
    if(nuevoEmail==='' || contra==='' || dni===''){
        $('#updateAdmin').text('Ingrese todos los datos');
        return;
    };
    
    $.ajax({
        url:'procActualizarAdmin.php',
        type:'POST',
        data:{txtEmailAdmin:nuevoEmail,txtClaveAdmin:contra,txtDNIAdmin:dni},
        success: function (respuesta) {
            $('#updateAdmin').html(respuesta);
        },
        error: function () {
            $('#updateAdmin').text('ERROR AL INTENTAR ACTUALIZAR');
        }
    });
}

function LimpiarCampos() {
    $('#txtEmailAdmin').val('');
    $('#txtClaveAdmin').val('');
    $('#txtDNIAdmin').val('');
    $('#updateAdmin').text('');
}