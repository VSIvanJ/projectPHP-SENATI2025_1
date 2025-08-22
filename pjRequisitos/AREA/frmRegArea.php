<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Registrar Área</title>
        <link href="Resources/css/style-regArea.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>Registro de Área:</h1>
        <hr>
        <div class="registroTabla">
            <form action="procRegArea.php" target="_blank">
                <label for="zona">Nombre de la zona:</label>
                <input type="text" id="txtZonaA" name="txtZonaA" required>
                <br><br>
                <label for="horas_trabajo">Horas al día:</label>
                <input type="text" id="txHorasA" name="txHorasA" required>
                <br><br>
                <label for="pago_hora">Pago por hora: S/.</label>
                <input type="text" id="txtPagoA" name="txtPagoA" required>
                <br><br>
                <label for="pago_hora">Resumen de zona:</label>
                <textarea id="txtResumen" name="txtResumen" rows="5" cols="10" required></textarea>
                <br><br>
                <input type="submit">
            </form>
        </div>
    </body>
</html>