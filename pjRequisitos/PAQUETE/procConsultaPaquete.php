<?php

include '../PDO/Conexion.php';
include '../PDO/PaqueteDAO.php';

$nom=$_POST['txtConPaq'];

$cPaq = new PaqueteDAO();

$consultaPaq = $cPaq->consultarPaquete($nom);

if($consultaPaq){
    echo "RESULTADO:<br>";
    echo "================== <br>";
    echo "ID: ".$consultaPaq['id_paquete']."<br>";
    echo "Nombre: ".$consultaPaq['nombre']."<br>";
    echo "Descripción: ".$consultaPaq['descripcion']."<br>";
    echo "Precio: S/.".$consultaPaq['precio']."<br>";
}