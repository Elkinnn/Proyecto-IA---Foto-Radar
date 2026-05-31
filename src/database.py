import sqlite3
from pathlib import Path


DATOS_PRUEBA = [
    ("PBC1234", "Chevrolet", "Blanco", "Ana Molina", "ana.molina@example.com"),
    ("GSK5678", "Kia", "Gris", "Luis Zambrano", "luis.zambrano@example.com"),
    ("TMA9012", "Hyundai", "Rojo", "Carla Vera", "carla.vera@example.com"),
]


def conectar(ruta_bd: str) -> sqlite3.Connection:
    Path(ruta_bd).parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(ruta_bd)
    conexion.row_factory = sqlite3.Row
    return conexion


def inicializar_bd(ruta_bd: str = "data/database/fotorradar.sqlite3") -> None:
    with conectar(ruta_bd) as conexion:
        conexion.execute(
            """
            CREATE TABLE IF NOT EXISTS vehiculos (
                placa TEXT PRIMARY KEY,
                marca TEXT,
                color TEXT,
                propietario TEXT,
                correo TEXT
            )
            """
        )
        conexion.execute(
            """
            CREATE TABLE IF NOT EXISTS eventos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha_hora TEXT,
                placa TEXT,
                velocidad REAL,
                estado TEXT,
                sancion TEXT,
                evidencia TEXT
            )
            """
        )
        conexion.executemany(
            """
            INSERT OR IGNORE INTO vehiculos (placa, marca, color, propietario, correo)
            VALUES (?, ?, ?, ?, ?)
            """,
            DATOS_PRUEBA,
        )


def buscar_vehiculo_por_placa(placa: str, ruta_bd: str = "data/database/fotorradar.sqlite3") -> dict | None:
    with conectar(ruta_bd) as conexion:
        fila = conexion.execute("SELECT * FROM vehiculos WHERE placa = ?", (placa,)).fetchone()
    return dict(fila) if fila else None


def listar_vehiculos(ruta_bd: str = "data/database/fotorradar.sqlite3") -> list[dict]:
    with conectar(ruta_bd) as conexion:
        filas = conexion.execute("SELECT * FROM vehiculos ORDER BY placa").fetchall()
    return [dict(fila) for fila in filas]


def listar_eventos(ruta_bd: str = "data/database/fotorradar.sqlite3", limite: int = 50) -> list[dict]:
    with conectar(ruta_bd) as conexion:
        filas = conexion.execute(
            "SELECT * FROM eventos ORDER BY id DESC LIMIT ?",
            (limite,),
        ).fetchall()
    return [dict(fila) for fila in filas]


def guardar_evento(evento: dict, ruta_bd: str = "data/database/fotorradar.sqlite3") -> int:
    with conectar(ruta_bd) as conexion:
        cursor = conexion.execute(
            """
            INSERT INTO eventos (fecha_hora, placa, velocidad, estado, sancion, evidencia)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                evento["fecha_hora"],
                evento["placa"],
                evento["velocidad"],
                evento["estado"],
                evento["sancion"],
                evento["evidencia"],
            ),
        )
        return int(cursor.lastrowid)
