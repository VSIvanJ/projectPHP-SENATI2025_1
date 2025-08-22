<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Login Empleado</title>
        <link href="Resources/css/style-loginEmp.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>Bienvenido Empleado</h1>
        <div class="login-container">
            <h2>Iniciar Sesión:</h2>
            <form action="procLoginEmpleado.php" method="POST">
                <label for="nombre">Usuario:</label>
                <input type="text" id="txtNEmp" name="txtNEmp" required>
                <br><br>
                <label for="clave">Contraseña:</label>
                <input type="password" id="txtCEmp" name="txtCEmp" required>
                <br><br>
                <button type="submit">Ingresar</button>
            </form>
        </div>
        <footer>
            <p>¿Eres empleado de nivel superior? 
                <a href="../ADMINISTRADOR/frmLoginAdministrador.php">Presiona este mensaje</a>
            </p>
      </footer>
    </body>
</html>
