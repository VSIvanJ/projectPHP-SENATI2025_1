<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Acciones Soporte</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js" crossorigin="anonymous"></script>
        <script src="Resources/js/scriptSoporte.js" type="text/javascript"></script>
        <link href="../Resources/css/baseCRUD.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>ACCIONES DE SOPORTE</h1>
        <div class="crudTabla">
            <h2>Consultar Personal de Soporte:</h2>
            <label for="id_personal">ID Personal:</label>
            <input type="text" id="txtCIS" name="txtCIS" required>
            <br><br>
            <label for="id_area">ID Área:</label>
            <input type="text" id="txtCAS" name="txtCAS" required>
            <br><br>
            <button onclick="ConsultarSoporte()">CONSULTAR</button>
            <button onclick="LimpiarConsultaS()">LIMPIAR</button>
            <br><br>
            <div id="selectSoporte"></div>
        </div>
        <br><br>
        <!-- comment <div class="crudTabla">
            <h2>Actualizar correo:</h2>
            <label for="id_personal">ID Personal:</label>
            <input type="text" id="txtAIS" name="txtActIdPer" required>
            <br><br>
            <label for="dni">Nº DNI:</label>
            <input type="text" id="txtActDNIPer" name="txtActDNIPer" required minlength="9" maxlength="9">
            <br><br>
            <label for="correo">Nuevo Correo:</label><br>
            <input type="email" id="txtActCor" name="txtActCor" required>
            <br><br>
            <button onclick="ActualizarSoporte()">ACTUALIZAR</button>
            <button onclick="LimpiarActualizarS()">LIMPIAR</button>
            <br><br>
            <div id="updateSoporte"></div>
        </div>-->
        <br><br>
        <div class="crudTabla">
            <h2>Eliminar soporte:</h2>
            <label for="id_soporte">ID Soporte:</label>
            <input type="text" id="txtEIS" name="txtEIS" required>
            <br><br>
            <label for="id_personal">ID Personal:</label>
            <input type="text" id="txtEIPS" name="txtEIPS" required>
            <br><br>
            <button onclick="EliminarSoporte()">ELIMINAR</button>
            <button onclick="LimpiarEliminarS()">LIMPIAR</button>
            <br><br>
            <div id="deleteSoporte"></div>
        </div>
    </body>
</html>