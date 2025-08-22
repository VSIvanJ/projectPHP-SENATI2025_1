<?php

include '../PDO/Conexion.php';
include '../PDO/ContratoDAO.php';

$cliente=$_POST['txtCCliC'];
$soporte=$_POST['txtCSC'];

$cCon = new ContratoDAO();

$consulCon = $cCon->consultarContrato($cliente, $soporte);

if($consulCon){
    echo "RESULTADO:<br>";
    echo "================== <br>";
    echo "ID Contrato: ".$consulCon['id_contrato']."<br>";
    echo "ID Cliente/Razón Social: ".$consulCon['id_cliente']." / ".$consulCon['razon_social']."<br>";
    echo "Personal/ID Soporte: ".$consulCon['nombre']." ".$consulCon['apellido']." / ".$consulCon['id_soporte']."<br>";
    echo "Contenido: ".$consulCon['contenido']."<br>";
    echo "Fecha Inicio/Finalización: ".$consulCon['fecha_inicio']." / ".$consulCon['fecha_fin']."<br>";
    echo "Monto: S/.".$consulCon['monto']."<br>";
    echo "Estado del Contrato: ".$consulCon['estado']."<br>";
}