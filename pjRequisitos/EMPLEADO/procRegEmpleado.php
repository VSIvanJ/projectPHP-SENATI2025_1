<?php

include '../PDO/Conexion.php';
include '../PDO/EmpleadoDAO.php';

$soporte = $_REQUEST['txtIDSop'];
$user = $_REQUEST['txtUserEmp'];
$clave= $_REQUEST['txtContraEmp'];
$validar=$_REQUEST['txtConfirCE'];

if($clave==$validar){
    //Cifrado de contraseña
    $hash= password_hash($clave, PASSWORD_DEFAULT);
    
    sleep(2);
    $rEmp = new EmpleadoDAO();
    $regEmp = $rEmp->registrarEmpleado($soporte, $user, $hash);
}else{
    echo "Las contraseñas ingresadas no coinciden...Por favor, vuelva a intentarlo";
}