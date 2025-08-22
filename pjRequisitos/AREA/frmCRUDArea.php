<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Acciones Área</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js" crossorigin="anonymous"></script>
        <script src="Resources/js/scriptArea.js" type="text/javascript"></script>
        <link href="../Resources/css/baseCRUD.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>ACCIONES DE ÁREA</h1>
        <div class="crudTabla">
            <h2>Consultar Área:</h2>
            <label for="nombre">Nombre de área:</label>
            <input type="text" id="txtCA" name="txtCA" required>
            <br><br>
            <button onclick="ConsultarArea()">CONSULTAR</button>
            <button onclick="LimpiarConsultaA()">LIMPIAR</button>
            <br><br>
            <div id="selectArea"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Actualizar Pago y Resumen:</h2>
            <label for="id_area">ID Área:</label>
            <input type="text" id="txtAI" name="txtAI" required>
            <br><br>
            <label for="pago_hora">Nuevo Pago por hora: S/.</label>
            <input type="text" id="txtAP" name="txtAP" required>
            <br><br>
            <label for="resumen">Nuevo Resumen:</label>
            <textarea id="txtAD" name="txtAD" rows="5" cols="30" maxlength="750" required></textarea>
            <br><br>
            <button onclick="ActualizarArea()">ACTUALIZAR</button>
            <button onclick="LimpiarActualizarA()">LIMPIAR</button>
            <br><br>
            <div id="updateArea"></div>
        </div>
        <br><br>
        <div class="crudTabla">
            <h2>Eliminar Área:</h2>
            <label for="id_area">ID Área:</label>
            <input type="text" id="txtEIA" name="txtEIA" required>
            <br><br>
            <button onclick="EliminarArea()">ELIMINAR</button>
            <button onclick="LimpiarEliminarA()">LIMPIAR</button>
            <br><br>
            <div id="deleteArea"></div>
        </div>
    </body>
</html>