<?php

include '../PDO/Conexion.php';
include '../PDO/PaqueteDAO.php';

$idP=$_POST['txtBID'];

$elP = new PaqueteDAO();

$elimPaq = $elP->eliminarPaquete($idP);