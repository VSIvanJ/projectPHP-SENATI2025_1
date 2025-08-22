<?php

include '../PDO/Conexion.php';
include '../PDO/SoporteDAO.php';

$personal=$_POST['txtCIS'];
$area=$_POST['txtCAS'];

$cSop = new SoporteDAO();

$consulSop=$cSop->consultarSoporte($personal, $area);

if($consulSop){
    echo "RESULTADO:<br>";
    echo "================== <br>";
    echo "ID Soporte: ".$consulSop['id_soporte']."<br>";
    echo "ID Personal: ".$consulSop['id_personal']."<br>";
    echo "ID Área/Zona: ".$consulSop['id_area']." / ".$consulSop['zona']."<br>";
    echo "Sueldo: S/.".$consulSop['sueldo']."<br>";
}