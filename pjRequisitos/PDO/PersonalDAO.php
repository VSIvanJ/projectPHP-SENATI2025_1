<?php

class PersonalDAO extends Conexion{
    
    public function registrarPersonal($emp,$nom,$ap,$dni,$correo) {
        try {
            $sql="INSERT INTO PERSONAL (id_empresa, nombre, apellido, dni, correo) VALUES (:id_empresa, :nombre, :apellido, :dni, :correo)";
            $unir=$this->conn->prepare($sql);
            
            $unir->bindParam(':id_empresa',$emp);
            $unir->bindParam(':nombre',$nom);
            $unir->bindParam(':apellido',$ap);
            $unir->bindParam(':dni',$dni);
            $unir->bindParam(':correo',$correo);
            $unir->execute();
            
            echo "Se registró al personal";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }

    public function consultarPersonal($dni) {
        try {
            $sql="SELECT * FROM PERSONAL WHERE dni = :dni";
            $unir=$this->conn->prepare($sql);
            $unir->bindParam(':dni',$dni);
            $unir->execute();
            $rpta=$unir->fetch(PDO::FETCH_ASSOC);
            
            if($rpta){
                return $rpta;
            }else{
                echo "El personal ingresado no existe";
            }
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function actualizarPersonal($id,$dni,$correo) {
        try {
            $sql="UPDATE PERSONAL SET correo = :correo WHERE dni = :dni AND id_personal = :id_personal";
            $unir=$this->conn->prepare($sql);
            
            $unir=bindParam(':id_personal',$id);
            $unir=bindParam(':dni',$dni);
            $unir=bindParam(':correo',$correo);
            $unir->execute();
            
            echo "Se actualizaron los datos del personal";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function eliminarPersonal($id,$dni) {
        try {
            $sql="DELETE FROM PERSONAL WHERE id_personal = :id_personal AND dni = :dni";
            $unir=$this->conn->prepare($sql);
            
            $unir=bindParam(':id_personal',$id);
            $unir=bindParam(':dni',$dni);
            $unir->execute();
            
            echo "Se eliminó el personal indicado";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
}
