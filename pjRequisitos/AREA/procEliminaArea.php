<?php

include '../PDO/Conexion.php';
include '../PDO/AreaDAO.php';

$area=$_POST['txtEIA'];

$eAr = new AreaDAO();

$elimArea = $eAr->eliminarArea($area);