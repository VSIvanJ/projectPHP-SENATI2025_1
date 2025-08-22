<?php

//MANTENER SESION
session_start();

include '../PDO/Conexion.php';
include '../PDO/EmpleadoDAO.php';

$user = $_POST['txtNEmp'];
$clave = $_POST['txtCEmp'];

$lEmp = new EmpleadoDAO();

$logEmp = $lEmp->loginEmpleado($user, $clave);

if($logEmp){
    //Guardar ID de empleado
    $_SESSION['id_empleado'] = $logEmp['id_empleado'];
    sleep(3);
    header("Location: ../accionesSoporte.php");
    exit();
}