<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Registro Contrato</title>
        <link href="Resources/css/style-regContrato.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>Registrar Contrato:</h1>
        <div class="registroTabla">
            <form action="procRegContrato.php">
                <label for="id_cliente">ID Cliente:</label>
                <input type="text" id="txtCliCon" name="txtCliCon" required>
                <br><br>
                <label for="id_soporte">ID Soporte:</label>
                <input type="text" id="txtSopCon" name="txtSopCon" required>
                <br><br>
                <label for="contenido">Contenido del Contrato:</label>
                <textarea id="txtConCon" name="txtConCon" rows="15" cols="15" required></textarea>
                <br><br>
                <label for="fecha_inicio">Fecha de Inicio:</label>
                <input type="date" id="fIni" name="fIni" required>
                <br><br>
                <label for="fecha_fin">Fecha de Finalización:</label>
                <input type="date" id="fFin" name="fFin" required>
                <br><br>
                <label for="monto">Monto a Pagar: S/.</label>
                <input type="text" id="txtMonCon" name="txtMonCon" required>
                <br><br>
                <input type="submit">
            </form>
        </div>
    </body>
</html>
