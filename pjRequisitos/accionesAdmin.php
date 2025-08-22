<?php
session_start();
if (!isset($_SESSION['id_admin'])) {
    header("Location: index.php");
    exit();
}
?>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Acciones Administrador</title>
        <link href="Resources/css/style-acciones.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
        <h1>LISTA DE ACCIONES PARA EL ADMINISTRADOR</h1>
        <hr>
        <nav>
            <ul>
                <li><strong>Personal</strong>
                    <ul>
                        <li><a href="PERSONAL/frmRegPersonal.php" target="_blank">Registrar Personal</a></li>
                        <li><a href="PERSONAL/frmCRUDPersonal.php" target="_blank">Consultar Personal</a></li>
                    </ul>
                </li>

                <li><strong>Clientes</strong>
                    <ul>
                        <li><a href="CLIENTE/frmCRUDCliente.php" target="_blank">Consultar Cliente</a></li>
                    </ul>
                </li>

                <li><strong>Área</strong>
                    <ul>
                        <li><a href="AREA/frmRegArea.php" target="_blank">Registrar Área</a></li>
                        <li><a href="AREA/frmCRUDArea.php" target="_blank">Consultar Área</a></li>
                    </ul>
                </li>

                <li><strong>Personal de Soporte</strong>
                    <ul>
                        <li><a href="SOPORTE/frmRegSoporte.php" target="_blank">Registrar Soporte</a></li>
                        <li><a href="SOPORTE/frmCRUDSoporte.php" target="_blank">Consultar Soporte</a></li>
                    </ul>
                </li>
                
                <li><strong>Crear Cuenta a Soporte</strong>
                    <ul>
                        <li><a href="EMPLEADO/frmRegEmpleado.php" target="_blank">Registrar Cuenta a Soporte</a></li>
                    </ul>
                </li>

                <li><strong>Usuarios</strong>
                    <ul>
                        <li><a href="USUARIO/frmCRUDUsuario.php" target="_blank">Consultar Usuario</a></li>
                    </ul>
                </li>

                <li><strong>Contratos Cerrados</strong>
                    <ul>
                        <li><a href="CONTRATO/frmCRUDContrato.php" target="_blank">Consultar Contratos Cerrados</a></li>
                    </ul>
                </li>

                <li><strong>Peticiones Recibidas</strong>
                    <ul>
                        <li><a href="PETICION/frmCRUDPeticion.php" target="_blank">Consultar Peticiones</a></li>
                    </ul>
                </li>

                <li><strong>Paquetes Ofrecidos</strong>
                    <ul>
                        <li><a href="PAQUETE/frmRegPaquete.php" target="_blank">Registrar Paquete</a></li>
                        <li><a href="PAQUETE/frmCRUDPaquete.php" target="_blank">Consultar Paquete</a></li>
                    </ul>
                </li>
            </ul>
        </nav>
    </body>
</html>