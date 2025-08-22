<?php

class UsuarioDAO extends Conexion{
    public function registrarUsuario($user,$email,$hash) {
        try {
            $sql="INSERT INTO USUARIO (usuario,email,clave) VALUES (:usuario,:email,:clave)";
            $stmt=$this->conn->prepare($sql);
            
            $stmt->bindParam(':usuario',$user);
            $stmt->bindParam(':email',$email);
            $stmt->bindParam(':clave',$hash);
            $stmt->execute();
            
            echo "Se registró correctamente...<br>";
            echo "¡Bienvenido! Ya puede cerrar esta ventana";
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }        
    }
    
    public function consultarUsuario($email) {
        try {
            $sql="SELECT * FROM USUARIO WHERE email=:email";
            $stmt=$this->conn->prepare($sql);
            $stmt->bindParam(':email',$email);
            $stmt->execute();
            $rpta=$stmt->fetch(PDO::FETCH_ASSOC);
            
            if($rpta){
                return $rpta;
            }else{
                echo "El email ingresado no existe...";
            }
            
        } catch (Exception $ex) {
            echo "ERROR: ".$ex;
        }
    }
    
    public function loginUsuario($user, $clave) {
        try {
            $sql = "SELECT id_usuario,clave FROM USUARIO WHERE usuario = :usuario";
            $stmt = $this->conn->prepare($sql);
            $stmt->bindParam(':usuario', $user);
            $stmt->execute();

            // Obtener la contraseña almacenada
            $resultado = $stmt->fetch(PDO::FETCH_ASSOC);

            if ($resultado && password_verify($clave, $resultado['clave'])) {
                echo "¡Inicio de sesión exitoso! Bienvenido, $user";
                return [
                    'id_usuario' => $resultado['id_usuario'],
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