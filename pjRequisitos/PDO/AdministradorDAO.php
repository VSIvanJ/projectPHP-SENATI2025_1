<?php

class AdministradorDAO extends Conexion{
    
    public function actualizarAdministrador($email,$clave,$dni) {
        try {
            $sql="UPDATE ADMINISTRADOR SET email=:email WHERE clave=:clave AND dni=:dni";
            $unir=$this->conn->prepare($sql);
            $unir->bindParam(':email',$email);
            $unir->bindParam(':clave',$clave);
            $unir->bindParam(':dni',$dni);
            $unir->execute();
            
        } catch (Exception $ex) {
            echo "Error al intentar actualizar: ".$ex;
        }
    }
    
    public function loginAdministrador($user, $clave) {
        try {
            $sql = "SELECT id_admin, clave FROM ADMINISTRADOR WHERE nombre = :nombre";
            $stmt = $this->conn->prepare($sql);
            $stmt->bindParam(':nombre', $user);
            $stmt->execute();

            // Obtener la contraseña almacenada
            $resultado = $stmt->fetch(PDO::FETCH_ASSOC);

            if ($resultado && password_verify($clave, $resultado['clave'])) {
                echo "¡Inicio de sesión exitoso! Bienvenido, $user";
                return [
                    'id_admin' => $resultado['id_admin'],
                    'usuario' => $user
                ];
            } else {
                echo "Usuario y/o contraseña incorrectos. Inténtelo de nuevo.";
                return false;
            }
        } catch (Exception $ex) {
            echo "ERROR: " . $ex->getMessage();
        }
    }
}