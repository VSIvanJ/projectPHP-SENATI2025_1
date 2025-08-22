<?php

include '../PDO/Conexion.php';
include '../PDO/ClienteDAO.php';

$nruc=$_POST['txtCRC'];

$cCli = new ClienteDAO();

$consulCli = $cCli->consultarCliente($nruc);

if($consulCli){
    echo "RESULTADO:<br>";
    echo "================== <br>";
    echo "ID: ".$consulCli['id_cliente']."<br>";
    echo "ID Usuario: ".$consulCli['id_usuario']."<br>";
    echo "Paquete comprado: ".$consulCli['nombre']."<br>";
    echo "Razón Social: ".$consulCli['razon_social']."<br>";
    echo "Nº RUC: ".$consulCli['ruc']."<br>";
    echo "Departamento: ".$consulCli['departamento']."<br>";
    echo "Dirección Legal: ".$consulCli['direccion']."<br>";
}