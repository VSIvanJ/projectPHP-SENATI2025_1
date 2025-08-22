<?php

include '../PDO/Conexion.php';
include '../PDO/PersonalDAO.php';

$dni=$_POST['txtConDniPer'];

$cPer = new PersonalDAO();

$consulPer=$cPer->consultarPersonal($dni);

if($consulPer){
    echo "RESULTADO:<br>";
    echo "================== <br>";
    echo "ID Personal: ".$consulPer['id_personal']."<br>";
    echo "ID Empresa: ".$consulPer['id_empresa']."<br>";
    echo "Nombres: ".$consulPer['nombre']."<br>";
    echo "Apellidos: ".$consulPer['apellido']."<br>";
    echo "Nº DNI: ".$consulPer['dni']."<br>";
    echo "Email: ".$consulPer['correo']."<br>";
}