<?php

include '../PDO/Conexion.php';
include '../PDO/PaqueteDAO.php';

$nom=$_REQUEST['txtNombreP'];
$descripcion=$_REQUEST['txtDescripcionP'];
$pre=$_REQUEST['txtPrecioP'];

$paq = new PaqueteDAO();

$rpaq = $paq->registrarPaquete($nom, $descripcion, $pre);