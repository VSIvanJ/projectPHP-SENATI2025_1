<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Registro Personal</title>
        <link href="Resources/css/style-regPersonal.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>REGISTRAR PERSONAL:</h1>
        <hr>
        <div class="registroTabla">
            <form action="../PERSONAL/procRegPersonal.php" method="post">
            <label id="nombre">Nombre:</label>
            <input type="text" id="txtNomb" name="txtNomb" required>
            <br><br>
            <label id="apellido">Apellido:</label>
            <input type="text" id="txtApel" name="txtApel" required>
            <br><br>
            <label id="dni">DNI:</label>
            <input type="text" id="txtDni" name="txtDni" required minlength="9" maxlength="9">
            <br><br>
            <label id="correo">Correo electrónico:</label>
            <input type="email" id="txtEmail" name="txtEmail" required>
            <br><br>
            <input type="submit">
            <br><br>
        </form>
        </div>
    </body>
</html>