<?php

class PaqueteDAO extends Conexion{
    
    public function registrarPaquete($nom,$descripcion,$pre) {
        try {
            $sql="INSERT INTO PAQUETE (nombre,descripcion,precio) VALUES (:nombre,:descripcion,:precio)";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':nombre',$nom);
            $unir->bindParam(':descripcion',$descripcion);
            $unir->bindParam(':precio',$pre);
            $unir->execute();
            
            echo "Se registró el paquete correctamente.";
            
        } catch (Exception $ex) {
            echo "Error al intentar registrar paquete: ".$ex;
        }
    }
    
    public function consultarPaquete($nom) {
        try {
            $sql="SELECT * FROM PAQUETE WHERE nombre=:nombre";
            $unir=$this->conn->prepare($sql);
            $unir->bindParam(':nombre',$nom);
            $unir->execute();
            $rpta=$unir->fetch(PDO::FETCH_ASSOC);
            
            if($rpta){
                return $rpta;
            }else{
                echo "El paquete ingresado no existe.";
            }
            
        } catch (Exception $ex) {
            echo "Error: ".$ex;
        }
    }
    
    public function actualizarPaquete($des,$pre,$id) {
        try {
            $sql="UPDATE PAQUETE SET descripcion=:descripcion, precio=:precio "
                    . "WHERE id_paquete=:id_paquete";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':descripcion',$des);
            $unir->bindParam(':precio',$pre);
            $unir->bindParam(':id_paquete',$id);
            $unir->execute();
            
            echo "Se actualizaron los datos.";
            
        } catch (Exception $ex) {
            echo "Error: ".$ex;
        }
    }
    
    public function eliminarPaquete($idP) {
        try {
            $sql="DELETE FROM PAQUETE WHERE id_paquete=:id_paquete";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_paquete',$idP);
            $unir->execute();
            
            echo "Se eliminó el paquete de la base de datos.";
            
        } catch (Exception $ex) {
            echo "Error: ".$ex;
        }
    }
    
    public function obtenerPaquetes() {
        try {
            $sql = "SELECT * FROM PAQUETE";
            $stmt = $this->conn->prepare($sql);
            $stmt->execute();
            return $stmt->fetchAll(PDO::FETCH_ASSOC);
            
        } catch (Exception $ex) {
            error_log("ERROR: ".$ex->getMessage());
            echo "Error al obtener paquetes: ".$ex;
            return [];
        }
    }
}
