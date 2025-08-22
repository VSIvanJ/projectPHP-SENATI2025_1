<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Acciones Paquete</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js" crossorigin="anonymous"></script>
        <script src="Resources/js/scriptPaquete.js" type="text/javascript"></script>
        <link href="../Resources/css/baseCRUD.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>ACCIONES PARA PAQUETE</h1>
        <div class="crudTabla">
            <h2>Consultar paquetes registrados:</h2>
            <label for="nombre">Ingrese nombre:</label>
            <input type="text" id="txtConPaq" name="txtConPaq" required>
            <br><br>
            <button onclick="ConsultarPaquete()">CONSULTAR</button>
            <button onclick="LimpiarConsulta()">LIMPIAR</button>
            <br><br>
            <div id="selectPaquete"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Actualizar datos de un paquete:</h2>
            <label for="id_paquete">ID Paquete:</label>
            <input type="text" id="txtActId" name="txtActId" required>
            <br><br>
            <label for="descripcion">Nueva Descripción:</label><br>
            <textarea id="txtActDes" name="txtActDes" rows="5" cols="30" maxlength="750" required></textarea>
            <br><br>
            <label for="precio">Precio: S/.</label>
            <input type="text" id="txtActPre" name="txtActPre" required>
            <br><br>
            <button onclick="ActualizarPaquete()">ACTUALIZAR</button>
            <button onclick="LimpiarActualizar()">LIMPIAR</button>
            <br><br>
            <div id="updatePaquete"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Eliminar paquete:</h2>
            <label for="id_paquete">ID Paquete</label>
            <input type="text" id="txtBID" name="txtBID" required>
            <br><br>
            <button onclick="EliminarPaquete()">ELIMINAR</button>
            <button onclick="LimpiarEliminar()">LIMPIAR</button>
            <br><br>
            <div id="deletePaquete"></div>
        </div>
    </body>
</html>