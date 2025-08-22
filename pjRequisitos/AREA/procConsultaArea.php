<?php

include '../PDO/Conexion.php';
include '../PDO/AreaDAO.php';

$zona=$_POST['txtCA'];

$cAr = new AreaDAO();

$consulArea = $cAr->consultarArea($zona);

if($consulArea){
    echo "RESULTADO:<br>";
    echo "================== <br>";
    echo "ID Área: ".$consulArea['id_area']."<br>";
    echo "Zona: ".$consulArea['zona']."<br>";
    echo "Horas de trabajo: ".$consulArea['horas_trabajo']."<br>";
    echo "Pago/hora: S/.".$consulArea['pago_hora']."<br>";
    echo "Descripción: ".$consulArea['resumen']."<br>";
}