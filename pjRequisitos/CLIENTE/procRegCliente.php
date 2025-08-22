<?php

include '../PDO/Conexion.php';
include '../PDO/ClienteDAO.php';

$iduser=$_REQUEST['txtIDUsu'];
$idpaquete=$_REQUEST['txtPaqueteUsu'];
$rsocial=$_REQUEST['txtRSUsu'];
$ruc=$_REQUEST['txtRUCUsu'];
$depart=$_REQUEST['txtDepUsu'];
$direc=$_REQUEST['txtDirL'];

$re= new ClienteDAO();

$regPer=$re->registrarCliente($iduser, $idpaquete, $rsocial, $ruc, $depart, $direc);